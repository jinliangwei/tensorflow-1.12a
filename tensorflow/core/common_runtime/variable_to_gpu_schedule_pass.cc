#include "tensorflow/core/common_runtime/variable_to_gpu_schedule_pass.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"

namespace tensorflow {

namespace {

struct PlacedGPUGraph {
  std::vector<int32_t> node_ids;
  std::vector<const Edge*> var_input_edges;
  std::unordered_set<int32_t> gpu_src_ids;
  std::vector<int32_t> scheduled_gpu_src;
};

bool IsNodeAssignedOnGPU(Node *node) {
  if (node->IsSource() || node->IsSink()) return false;
  const auto& assigned_device = node->assigned_device_name();
  CHECK(!assigned_device.empty())
      << node->name()
      << " requested_device_name = " << node->requested_device();
  DeviceNameUtils::ParsedName parsed_device_name;
  DeviceNameUtils::ParseFullName(assigned_device, &parsed_device_name);
  return (parsed_device_name.type == "GPU" ||
          parsed_device_name.type == "gpu");
}

bool IsNodeAssignedOnCPU(Node *node) {
  if (node->IsSource() || node->IsSink()) return false;
  const auto& assigned_device = node->assigned_device_name();
  CHECK(!assigned_device.empty())
      << node->name()
      << " requested_device_name = " << node->requested_device();
  DeviceNameUtils::ParsedName parsed_device_name;
  DeviceNameUtils::ParseFullName(assigned_device, &parsed_device_name);
  return (parsed_device_name.type == "CPU" ||
          parsed_device_name.type == "cpu");
}

void ComputeGPUSrcDeps(Graph *g,
                       const std::unordered_set<int32_t> &gpu_src_ids,
                       std::unordered_map<int32_t, std::vector<int32_t>> *gpu_src_deps,
                       std::unordered_set<int32_t> *ready_gpu_node_ids) {
  LOG(INFO) << __func__;

  size_t num_node_ids = g->num_node_ids();
  std::vector<int32_t> ready_nodes;
  size_t front = 0, back = 0;
  std::vector<int32_t> num_ready_inputs(num_node_ids, 0);
  std::vector<std::unordered_set<int32_t>> deps_on_gpu_src_nodes(num_node_ids);

  for (int node_id = 0; node_id < num_node_ids; node_id++) {
    Node *node = g->FindNodeId(node_id);
    if (!node) continue;
    if (node->num_inputs() == 0) {
      ready_nodes.push_back(node_id);
      ready_gpu_node_ids->insert(node_id);
      back++;
    }

    if (node->IsMerge()) {
      for (const auto &input_edge : node->in_edges()) {
        Node *src = input_edge->src();
        int32_t src_id = src->id();
        if (src->IsNextIteration()) {
          num_ready_inputs[node_id]++;
        }
      }
    }
  }

  while (front != back) {
    int32_t ready_node_id = ready_nodes[front];
    Node *ready_node = g->FindNodeId(ready_node_id);
    bool is_gpu_src_dep = (gpu_src_ids.count(ready_node_id) == 1);
    for (const auto &out_edge : ready_node->out_edges()) {
      Node* dst = out_edge->dst();
      int32_t dst_id = dst->id();
      num_ready_inputs[dst_id]++;
      for (int32_t ready_node_dep_id : deps_on_gpu_src_nodes[ready_node_id]) {
        deps_on_gpu_src_nodes[dst_id].insert(ready_node_dep_id);
      }
      if (is_gpu_src_dep) deps_on_gpu_src_nodes[dst_id].emplace(ready_node_id);
      if (num_ready_inputs[dst_id] == dst->num_inputs()) {
        ready_nodes.push_back(dst_id);
        if (deps_on_gpu_src_nodes[dst_id].empty()) {
          ready_gpu_node_ids->insert(dst_id);
        }
        back++;
      }
    }
    front++;
  }

  for (int32_t src_id = 0; src_id < g->num_node_ids(); src_id++) {
    if (!deps_on_gpu_src_nodes[src_id].empty()) {
      (*gpu_src_deps)[src_id].insert((*gpu_src_deps)[src_id].end(),
                                     deps_on_gpu_src_nodes[src_id].begin(),
                                     deps_on_gpu_src_nodes[src_id].end());

      LOG(INFO) << __func__
                << " node " << g->FindNodeId(src_id)->name()
                << " node.id = " << src_id
                << " num_deps = " << (*gpu_src_deps)[src_id].size();
    }
  }
}

void ExhaustOneInput(Node *gpu_src_input,
                     std::unordered_set<int32_t> *ready_node_ids,
                     std::set<int32_t> *pending_node_ids) {
  LOG(INFO) << __func__ << " begin";
  std::vector<Node*> node_queue;
  node_queue.push_back(gpu_src_input);
  ready_node_ids->emplace(gpu_src_input->id());
  int32_t front = 0, back = 1;
  while (front != back) {
    Node* curr_node = node_queue[front];
    LOG(INFO) << __func__ << " curr_node.name = " << curr_node->name();
    for (const Edge* out_edge : curr_node->out_edges()) {
      Node *dst = out_edge->dst();
      LOG(INFO) << __func__ << " dst.name = " << dst->name();
      if (!IsNodeAssignedOnGPU(dst) || dst->IsSource() || dst->IsSink()) continue;
      bool pending = false;
      for (const Edge* in_edge : dst->in_edges()) {
        int32_t src_id = in_edge->src()->id();
        if (ready_node_ids->count(src_id) == 0) {
          pending_node_ids->emplace(dst->id());
          LOG(INFO) << __func__ << " emplace pending node "
                    << dst->name()
                    << " missing src.name = " << in_edge->src()->name();
          pending = true;
          break;
        }
      }

      if (!pending) {
        node_queue.push_back(dst);
        ready_node_ids->emplace(dst->id());
        pending_node_ids->erase(dst->id());
        back++;
      }
    }
    front++;
  }
}

Node* FindFirstReadyGPUSrcFromGPUSrc(int32_t gpu_src_id,
                                     Graph *g,
                                     const std::set<int32_t> &ready_gpu_src_ids,
                                     const std::unordered_set<int32_t> &ready_node_ids,
                                     const std::unordered_set<int32> &gpu_src_id_set,
                                     const std::unordered_map<int32_t, std::vector<int32_t>> &gpu_src_deps) {
  LOG(INFO) << __func__ << " gpu_src = " << g->FindNodeId(gpu_src_id)->name();
  if (ready_gpu_src_ids.count(gpu_src_id) == 1) return g->FindNodeId(gpu_src_id);
  auto dep_iter = gpu_src_deps.find(gpu_src_id);
  if (dep_iter == gpu_src_deps.end()) return nullptr;

  for (int32_t dep : dep_iter->second) {
    LOG(INFO) << __func__ << " dep_gpu_src = " << g->FindNodeId(dep)->name();
    auto result = FindFirstReadyGPUSrcFromGPUSrc(dep, g,
                                                 ready_gpu_src_ids,
                                                 ready_node_ids,
                                                 gpu_src_id_set,
                                                 gpu_src_deps);
    if (result != nullptr) return result;
  }
  return nullptr;
}

Node* FindFirstReadyGPUSrcFromPending(int32_t pending_node_id,
                                      Graph *g,
                                      const std::set<int32_t> &ready_gpu_src_ids,
                                      const std::unordered_set<int32_t> &ready_node_ids,
                                      const std::unordered_set<int32> &gpu_src_id_set,
                                      const std::unordered_map<int32_t, std::vector<int32_t>> &gpu_src_deps) {
  Node* pending_node = g->FindNodeId(pending_node_id);
  LOG(INFO) << __func__ << " pending_node = "
            << pending_node->name();

  Node* result = nullptr;
  for (const Edge* in_edge : pending_node->in_edges()) {
    Node* src = in_edge->src();
    int32_t src_id = src->id();
    if (gpu_src_id_set.count(src_id) == 1) {
      result = FindFirstReadyGPUSrcFromGPUSrc(src_id, g,
                                              ready_gpu_src_ids,
                                              ready_node_ids,
                                              gpu_src_id_set,
                                              gpu_src_deps);
      if (result != nullptr) return result;
      else continue;
    }
    if (ready_node_ids.count(src_id) == 1) continue;
    LOG(INFO) << __func__ << " src_id = " << src_id;
    result = FindFirstReadyGPUSrcFromPending(src_id, g,
                                             ready_gpu_src_ids,
                                             ready_node_ids,
                                             gpu_src_id_set,
                                             gpu_src_deps);
    if (result != nullptr) return result;
    else continue;
  }
  return nullptr;
}

void ScheduleGPUInputs(Graph *g,
                       std::unordered_map<std::string, PlacedGPUGraph> *placed_gpu_graphs,
                       const std::unordered_set<int32_t> &gpu_src_ids,
                       const std::unordered_map<int32_t, std::vector<int32_t>> &gpu_src_deps,
                       std::unordered_set<int32_t> *ready_node_ids) {
  LOG(INFO) << __func__
            << " ready_node_ids.size() = "
            << ready_node_ids->size();

  if (gpu_src_ids.empty()) return;

  std::set<int32_t> ready_gpu_src;
  std::unordered_map<int32_t, std::vector<int32_t>> gpu_src_to_descendants;
  std::vector<int32_t> gpu_src_num_ready_inputs(g->num_node_ids(), 0);
  for (int32_t gpu_src_id : gpu_src_ids) {
    auto dep_iter = gpu_src_deps.find(gpu_src_id);
    if (dep_iter == gpu_src_deps.end()) {
      ready_gpu_src.insert(gpu_src_id);
      continue;
    }

    CHECK(dep_iter->second.size() > 0);
    for (int32_t dep : dep_iter->second) {
      gpu_src_to_descendants[dep].push_back(gpu_src_id);
    }
  }

  std::set<int32_t> pending_node_ids;
  LOG(INFO) << __func__ << " ready_gpu_src.size() = " << ready_gpu_src.size();
  while (!ready_gpu_src.empty()) {
    int32_t curr_gpu_src_id = *(ready_gpu_src.begin());
    LOG(INFO) << __func__ << " curr_gpu_src_id = " << curr_gpu_src_id;
    Node* curr_gpu_src = g->FindNodeId(curr_gpu_src_id);
    const auto &assigned_device_name = curr_gpu_src->assigned_device_name();

    (*placed_gpu_graphs)[assigned_device_name].scheduled_gpu_src.push_back(curr_gpu_src_id);
    ready_gpu_src.erase(curr_gpu_src_id);

    LOG(INFO) << __func__ << " curr_gpu_src = " << curr_gpu_src->name();
    ExhaustOneInput(curr_gpu_src,
                    ready_node_ids,
                    &pending_node_ids);

    while (!pending_node_ids.empty()) {
      auto desc_iter = gpu_src_to_descendants.find(curr_gpu_src->id());
      if (desc_iter != gpu_src_to_descendants.end()) {
        for (int32_t desc_id : desc_iter->second) {
          gpu_src_num_ready_inputs[desc_id] += 1;
          auto dep_iter = gpu_src_deps.find(desc_id);
          CHECK(dep_iter != gpu_src_deps.end());
          if (gpu_src_num_ready_inputs[desc_id] == dep_iter->second.size()) {
            ready_gpu_src.emplace(desc_id);
          }
        }
      }

      int32_t pending_node_id = *(pending_node_ids.begin());
      LOG(INFO) << __func__ << " pending_node_id = " << pending_node_id;
      curr_gpu_src = FindFirstReadyGPUSrcFromPending(pending_node_id,
                                                     g,
                                                     ready_gpu_src,
                                                     *ready_node_ids,
                                                     gpu_src_ids,
                                                     gpu_src_deps);
      LOG(INFO) << __func__ << " due to pending node, curr_gpu_src = " << curr_gpu_src->name();
      CHECK(curr_gpu_src != nullptr);
      (*placed_gpu_graphs)[assigned_device_name].scheduled_gpu_src.push_back(curr_gpu_src->id());
      ready_gpu_src.erase(curr_gpu_src->id());

      ExhaustOneInput(curr_gpu_src,
                      ready_node_ids,
                      &pending_node_ids);
    }
    LOG(INFO) << __func__ << " ready_gpu_src.size() = " << ready_gpu_src.size();
  }
}
}

Status
VariableToGPUSchedulePass::Run(const GraphOptimizationPassOptions& options) {
  LOG(INFO) << __func__ << " from VariableToGPUSchedulePass";
  if (options.partition_graphs != nullptr) {
    return errors::Internal(
        "Variable placement should happen before partitioning.");
  }

  if (options.graph == nullptr) {
    return Status::OK();
  }

  Graph* g = options.graph->get();
  std::unordered_set<int32_t> gpu_src_ids;
  std::unordered_map<std::string, PlacedGPUGraph> placed_gpu_graphs;

  for (Node* node: g->nodes()) {
    if (IsNodeAssignedOnGPU(node)) {
      const auto& assigned_device = node->assigned_device_name();
      placed_gpu_graphs[assigned_device].node_ids.push_back(node->id());
      for (const Edge* input_edge : node->in_edges()) {
        Node *src = input_edge->src();
        if (IsNodeAssignedOnCPU(src)) {
          gpu_src_ids.insert(src->id());
          placed_gpu_graphs[assigned_device].gpu_src_ids.insert(src->id());
          placed_gpu_graphs[assigned_device].var_input_edges.push_back(input_edge);
          LOG(INFO) << __func__ << " from VariableToGPUSchedulePass "
                    << " device = " << assigned_device
                    << " gpu_src = " << src->name()
                    << " gpu_node = " << input_edge->dst()->name();
        }
      }
    }
  }

  // deps from a GPU source node to other source nodes for that GPU
  std::unordered_map<int32_t, std::vector<int32_t>> gpu_src_deps;
  std::unordered_set<int32_t> ready_node_ids;
  ComputeGPUSrcDeps(g, gpu_src_ids, &gpu_src_deps, &ready_node_ids);
  ScheduleGPUInputs(g, &placed_gpu_graphs, gpu_src_ids, gpu_src_deps, &ready_node_ids);
  return Status::OK();
}

//REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
//                      VariableToGPUSchedulePass);
}
