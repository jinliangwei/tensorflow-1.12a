#include "tensorflow/core/common_runtime/variable_placement_pass.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"

namespace tensorflow {

namespace {

void PlaceNodeOnCPU(Node* node) {
  if (!node->requested_device().empty()) {
    const auto &requested_device = node->requested_device();
    DeviceNameUtils::ParsedName parsed_device_name;
    DeviceNameUtils::ParseFullName(requested_device, &parsed_device_name);
    parsed_device_name.type = "CPU";
    parsed_device_name.has_type = true;
    auto new_requested_device = DeviceNameUtils::ParsedNameToString(parsed_device_name);
    node->set_requested_device(new_requested_device);
  } else {
    DeviceNameUtils::ParsedName parsed_device_name;
    parsed_device_name.type = "CPU";
    parsed_device_name.has_type = true;
    auto new_requested_device = DeviceNameUtils::ParsedNameToString(parsed_device_name);
    node->set_requested_device(new_requested_device);
  }
}

void ComputeTopologicalOrder(Graph *g,
                             std::vector<int32_t> *topo_order,
                             std::vector<int32_t> *last_parent) {
  size_t num_node_ids = g->num_node_ids();
  std::vector<int32_t> ready_nodes;
  size_t front = 0, back = 0;
  std::vector<int32_t> num_ready_inputs(num_node_ids, 0);
  for (int node_id = 0; node_id < num_node_ids; node_id++) {
    Node *node = g->FindNodeId(node_id);
    if (!node) continue;
    if (node->num_inputs() == 0) {
      ready_nodes.push_back(node_id);
      back++;
      (*topo_order)[node_id] = 0;
      (*last_parent)[node_id] = -1;
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

    for (const auto &out_edge : ready_node->out_edges()) {
      Node* dst = out_edge->dst();
      int32_t dst_id = dst->id();
      num_ready_inputs[dst_id]++;
      if (num_ready_inputs[dst_id] == dst->num_inputs()) {
        ready_nodes.push_back(dst_id);
        back++;
        (*topo_order)[dst_id] = (*topo_order)[ready_node_id] + 1;
        (*last_parent)[dst_id] = ready_node_id;
      }
    }
    front++;
  }
}

std::string NewName(Graph *g,
                    const std::string &base, const std::string &append) {
  return g->NewName(strings::StrCat(base, "/", append));
}

int32_t GetNodeNthAncestor(int32_t node_id,
                           const std::vector<int32_t>& last_parent,
                           int32_t n) {
  int32_t curr_node_id = node_id;
  for (int32_t i = 0; i < n && curr_node_id >= 0; i++) {
    curr_node_id = last_parent[curr_node_id];
  }
  return curr_node_id;
}

}

Status
VariablePlacementPass::Run(const GraphOptimizationPassOptions& options) {
  return Status::OK();
  if (options.partition_graphs != nullptr) {
    return errors::Internal(
        "Variable placement should happen before partitioning.");
  }

  if (options.graph == nullptr) {
    return Status::OK();
  }

  Graph* g = options.graph->get();

  std::vector<int32_t> topo_order(g->num_node_ids());
  std::vector<int32_t> last_parent(g->num_node_ids());
  ComputeTopologicalOrder(g, &topo_order, &last_parent);

  struct NodeTopoOrder {
    const Edge* edge;
    int32_t topo_order;
    NodeTopoOrder(const Edge* edge,
                  int32_t topo_order):
        edge(edge),
        topo_order(topo_order) { }
  };

  std::unordered_map<int32_t,
                     std::vector<NodeTopoOrder>> node_id_to_child_topo_order;
  std::vector<int32_t> node_to_child_topo_order;

  for (Node *node : g->nodes()) {
    if (node->IsVariable()
        || node->type_string() == "VarHandleOp") {
      //|| node->IsConstant()) {
      // Place variables on CPU: same job, same task, same replica
      PlaceNodeOnCPU(node);
      // Place all fanout nodes with a reference edge on the same CPU
      //for (const auto &out_edge : node->out_edges()) {
      //  if (out_edge->IsControlEdge()) continue;
      //  int32_t output_idx = out_edge->src_output();
      //  DataType output_type = node->output_type(output_idx);
      //  if (!IsRefType(output_type)) continue;

      //  Node* dst = out_edge->dst();
      //  PlaceNodeOnCPU(dst);

        // partition the child nodes of dst based on their
        // topological order and replicate dst to cluster
        // child nodes

        /*
        if (!dst->IsIdentity()) continue;
        if (dst->out_edges().size() == 0) continue;

        auto& child_topo_order = node_id_to_child_topo_order[dst->id()];
        LOG(INFO) << __func__ << " node.name = " << node->name()
                  << " dst.name = " << dst->name()
                  << " dst.num_out_edges = " << dst->out_edges().size();
        for (const Edge* dst_out_edge : dst->out_edges()) {
          Node* dst_child = dst_out_edge->dst();
          int32_t dst_child_id = dst_child->id();
          int32_t dst_child_order = topo_order[dst_child_id];
          child_topo_order.emplace_back(dst_out_edge, dst_child_order);
        }

        std::sort(child_topo_order.begin(), child_topo_order.end(),
                  [](const NodeTopoOrder& a,
                     const NodeTopoOrder& b) {
                    return a.topo_order < b.topo_order;
                  });

        std::vector<int32_t> partition_idx;
        partition_idx.push_back(0);

        for (auto idx = 1; idx < child_topo_order.size(); idx++) {
          if (child_topo_order[idx].topo_order > (child_topo_order[idx - 1].topo_order + 1)) {
            LOG(INFO) << " topo_order diff = "
                      << child_topo_order[idx].topo_order - child_topo_order[idx - 1].topo_order;
            partition_idx.push_back(idx);
          }
        }

        if (partition_idx.size() > 1) {
          LOG(INFO) << __func__ << " partition the child nodes of node "
                    << dst->name()
                    << " node type = " << dst->type_string()
                    << " num_partitions = " << partition_idx.size();
          for (size_t offset = 1; offset < partition_idx.size(); offset++) {
            int32_t partition_start = partition_idx[offset];
            int32_t partition_end = (offset == (partition_idx.size() - 1))
                                    ? child_topo_order.size() - 1
                                    : partition_idx[offset + 1] - 1;

            std::string appendix("copy_");
            appendix += std::to_string(offset - 1);
            NodeBuilder node_builder(NewName(g, dst->name(), appendix),
                                     "RefIdentity", g->op_registry());
            node_builder.Input(node, output_idx);
            node_builder.Device(dst->requested_device());
            Node* new_node;
            CHECK(node_builder.Finalize(g, &new_node).ok());
            LOG(INFO) << __func__ << " add node "
                      << new_node->name();

            for (int32_t i = partition_start; i <= partition_end; i++) {
              const Edge* child_edge = child_topo_order[i].edge;
              Node* child_node = child_edge->dst();
              int32_t child_input_idx = child_edge->dst_input();
              g->AddEdge(new_node, child_edge->src_output(),
                         child_node, child_input_idx);
              g->RemoveEdge(child_edge);
            }
          }
          } */
      //}
    }
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      VariablePlacementPass);
}
