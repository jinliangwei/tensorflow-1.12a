#include "tensorflow/core/common_runtime/variable_placement_pass.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"

namespace tensorflow {

namespace {

void PlaceNodeOnCPU(Node* node) {
  /* const NodeDef& def = node->def();
  auto iter = def.attr().find("value");
  if (iter != def.attr().end()) {
    AttrValue attr_tensor = iter->second;
    size_t num_elements = 1;
    TensorShapeProto tensor_shape = attr_tensor.tensor().tensor_shape();

    for (auto dim : tensor_shape.dim()) {
      num_elements *= dim.size();
    }
    if (num_elements < 1024) {
      LOG(INFO) << __func__
                << " do not place " << node->name()
                << " on CPU because number of elements is too small, num_elements = "
                << num_elements;
      return;
    }
  } else {
    iter = def.attr().find("shape");
    if (iter != def.attr().end()) {
      TensorShapeProto tensor_shape = iter->second.shape();
      size_t num_elements = 1;

      for (auto dim : tensor_shape.dim()) {
        num_elements *= dim.size();
      }

      if (num_elements < 1024) {
        LOG(INFO) << __func__
                  << " do not place " << node->name()
                  << " on CPU because number of elements is too small, num_elements = "
                  << num_elements;
        return;
      }
    }
  }

  LOG(INFO) << __func__ << " " << node->name()
            << " type: " << node->type_string()
            << " IsConstant: " << node->IsConstant();
  */

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

}

Status
VariablePlacementPass::Run(const GraphOptimizationPassOptions& options) {
  if (options.partition_graphs != nullptr) {
    return errors::Internal(
        "Variable placement should happen before partitioning.");
  }

  if (options.graph == nullptr) {
    return Status::OK();
  }

  Graph* g = options.graph->get();

  for (Node *node : g->nodes()) {
    if (node->IsVariable()
        || node->type_string() == "VarHandleOp"
        || node->IsConstant()) {
      // Place variables on CPU: same job, same task, same replica
      PlaceNodeOnCPU(node);
    }
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      VariablePlacementPass);
}
