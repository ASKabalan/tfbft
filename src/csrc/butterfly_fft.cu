#include "collectives/butterfly_comm_index.hpp"
#include "collectives/collective_ops.hpp"
#include "common/ffi_helper.hpp"
#include "extensions.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <nccl.h>

namespace ffi = xla::ffi;

template <typename T> __global__ void AddElementKernel(T *x, T *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * 2;
  }
}

template <DataType dtype>
ffi::Error ButterFlyForward(cudaStream_t stream, Buffer<dtype> x, NORM iNorm,
                            Result<dtype> y, CommIterator &comms) {
  CCO::CollectiveOps ops;
  CCO::ReductionOp reduc_op{CCO::ReducType::SUM};
  auto stageComm = comms.next();
  assertm(stageComm.has_value(),
          "[INTERNAL] Distributed FFT called without Buttefly comms");

  // Get the Stage comm and the N (normalization factor for this stage)
  auto [N, comm] = stageComm.value();
  int device_count;
  NCCLCHECK(ncclCommCount(comm, &device_count));
  assertm(device_count == 2, "In each step the communication is pairwise");

  // First step A + B
  ops.reduce(x.typed_data(), y->typed_data(), x.element_count(), reduc_op, 0,
             comm, stream);
  // Second step A - B
  ops.reduce(y->typed_data(), y->typed_data(), y->element_count(), reduc_op, 1,
             comm, stream);

  // apply the twiddle factor

  // Continue with the rest of the stages
  while ((stageComm = comms.next())) {
    auto [N, comm] = stageComm.value();
    ops.reduce(y->typed_data(), y->typed_data(), y->element_count(), reduc_op,
               0, comm, stream);
    ops.reduce(y->typed_data(), y->typed_data(), y->element_count(), reduc_op,
               1, comm, stream);
  }
  return ffi_with_cuda_error_check();
}

template <DataType dtype>
ffi::Error ButterFlyBackward(cudaStream_t stream, Buffer<DataType::F32> x,
                             NORM iNorm, Result<dtype> y, CommIterator &comms) {

  CCO::CollectiveOps ops;
  CCO::ReductionOp reduc_op{CCO::ReducType::SUM};
  comms.reset();
  auto comm = comms.prev();
  assertm(comm.has_value(),
          "[INTERNAL] Distributed IFFT called without Buttefly comms");

  // ops.allreduce(x.typed_data(), y->typed_data(), x.element_count(), reduc_op,
  //               comm.value(), stream);
  //
  // while ((comm = comms.prev())) {
  //   ops.allreduce(y->typed_data(), y->typed_data(), x.element_count(),
  //   reduc_op,
  //                 comm.value(), stream);
  // }
  return ffi_with_cuda_error_check();
}

template <DataType dtype>
ffi::Error ButterFlyFFT(cudaStream_t stream, Buffer<DataType::F32> x,
                        int64_t iDirection, int64_t iAxis, int64_t iNorm,
                        Result<dtype> y) {

  AXIS axis = static_cast<AXIS>(iAxis);
  DIRECTION direction = static_cast<DIRECTION>(iDirection);
  NORM norm = static_cast<NORM>(iNorm);

  assertm(axis == AXIS::X, "Only X axis is supported");

  ncclComm_t comm = CCO::NCCLOps::get_comm();
  const int &rank = CCO::NCCLOps::get_rank();
  const int &size = CCO::NCCLOps::get_size();

  auto butterfly_comm = ButterflyCommIndex::get_or_create_comms(comm);
  CCO::ReductionOp reduc_op{CCO::ReducType::SUM};

  switch (direction) {
  case DIRECTION::FORWARD:
    return ButterFlyForward(stream, x, norm, y, butterfly_comm);
  case DIRECTION::INVERSE:
    return ButterFlyBackward(stream, x, norm, y, butterfly_comm);
  default:
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("Un recongnized FFT direction "));
  }
}

template <DataType dtype>
ffi::Error TestFunc(cudaStream_t stream, Buffer<dtype> x, Result<dtype> y) {
  int blockSize = 256;
  int numBlocks = (x.element_count() + blockSize - 1) / blockSize;
  AddElementKernel<<<numBlocks, blockSize, 0, stream>>>(
      x.typed_data(), y->typed_data(), x.element_count());
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ButterFlyFFTHandlerF32,
                              ButterFlyFFT<DataType::F32>,
                              ffi::Ffi::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Arg<Buffer<DataType::F32>>() // x
                                  .Attr<int64_t>("direction")
                                  .Attr<int64_t>("norm")
                                  .Attr<int64_t>("axis")
                                  .Ret<Buffer<DataType::F32>>() // y
);
