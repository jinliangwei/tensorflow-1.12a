/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Op kernels used to swap data in and out of GPU memory.

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

class CopyFromGpuToHostKernel : public AsyncOpKernel {
 public:
  explicit CopyFromGpuToHostKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    //LOG(INFO) << __func__ << " from CopyFromGpuToHostKernel";
    const Tensor& input = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, !ctx->input_alloc_attr(0).on_host(),
        errors::Internal("The input tensor to the _CopyFromGpuToHost kernel "
                         "must reside on the device."),
        done);

    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_gpu_compatible(true);
    alloc_attrs.set_on_host(true);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, input.shape(), &output, alloc_attrs),
        done);

    ctx->op_device_context()->CopyDeviceTensorToCPU(
        &input, "CopyFromGpuToHost", static_cast<Device*>(ctx->device()),
        output, [ctx, done](const Status& s) {
          ctx->SetStatus(s);
          done();
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_CopyFromGpuToHost").Device(DEVICE_GPU).HostMemory("output"),
    CopyFromGpuToHostKernel);

class CopyFromHostToGpuKernel : public AsyncOpKernel {
 public:
  explicit CopyFromHostToGpuKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    //LOG(INFO) << __func__ << " from CopyFromHostToGpuKernel";
    const Tensor& input = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, ctx->input_alloc_attr(0).on_host(),
        errors::Internal("The input tensor to the _CopyFromHostToGpu kernel "
                         "must reside on the host."),
        done);

    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input.shape(), &output),
                         done);

    ctx->op_device_context()->CopyCPUTensorToDevice(
        &input, static_cast<Device*>(ctx->device()), output,
        [ctx, done](const Status& s) {
          ctx->SetStatus(s);
          done();
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_CopyFromHostToGpu").Device(DEVICE_GPU).HostMemory("input"),
    CopyFromHostToGpuKernel);

class CopyFromGpuToHostAndClearKernel : public AsyncOpKernel {
 public:
  explicit CopyFromGpuToHostAndClearKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    //LOG(INFO) << __func__ << " from CopyFromGpuToHostAndClearKernel";
    mutex_lock l(*ctx->input_ref_mutex(0));
    Tensor lhs = ctx->mutable_input(0, true);
    OP_REQUIRES_ASYNC(
        ctx, !ctx->input_alloc_attr(0).on_host(),
        errors::Internal("The input tensor to the _CopyFromGpuToHostAndClear kernel "
                         "must reside on the device."),
        done);

    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_gpu_compatible(true);
    alloc_attrs.set_on_host(true);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, lhs.shape(), &output, alloc_attrs),
        done);

    ctx->op_device_context()->CopyDeviceTensorToCPU(
        &lhs, "CopyFromGpuToHostAndClear", static_cast<Device*>(ctx->device()),
        output, [ctx, done](const Status& s) {
          ctx->SetStatus(s);
          done();
        });

    ctx->clear_ref_input(0, true);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_CopyFromGpuToHostAndClear").Device(DEVICE_GPU).HostMemory("output"),
    CopyFromGpuToHostAndClearKernel);

class CopyFromHostToGpuAndAssignKernel : public AsyncOpKernel {
 public:
  explicit CopyFromHostToGpuAndAssignKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    //LOG(INFO) << __func__ << " from CopyFromHostToGpuAndAssignKernel";
    mutex_lock l(*ctx->input_ref_mutex(0));
    Tensor lhs = ctx->mutable_input(0, true);
    const Tensor& rhs = ctx->input(1);

    OP_REQUIRES_ASYNC(
        ctx, ctx->input_alloc_attr(1).on_host(),
        errors::Internal("The input tensor to the _CopyFromHostToGpuAndAssign kernel "
                         "must reside on the host."),
        done);

    PersistentTensor copy;
    Tensor* copyTensor = nullptr;

    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_persistent(rhs.dtype(), rhs.shape(), &copy,
                                      &copyTensor, attr));

    ctx->clear_recorded_memory();

    ctx->op_device_context()->CopyCPUTensorToDevice(
        &lhs, static_cast<Device*>(ctx->device()), copyTensor,
        [ctx, done](const Status& s) {
          ctx->SetStatus(s);
          done();
        });
    ctx->replace_ref_input(0, *copyTensor, true);
    ctx->set_output_ref(0, ctx->input_ref_mutex(0), copyTensor);
  }
};

//REGISTER_KERNEL_BUILDER(
//    Name("_CopyFromHostToGpuAndAssign").Device(DEVICE_GPU).HostMemory("input"),
//    CopyFromHostToGpuAndAssignKernel);

}  // namespace
}  // namespace tensorflow
