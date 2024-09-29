#include "collectives/butterfly_comm_index.hpp"
#include "collectives/collective_ops.hpp"
#include "common/ffi_helper.hpp"
#include "extensions.h"
#include "matx.h"
#include "matx/core/make_tensor.h"
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
ffi::Error ButterFlyFFT(cudaStream_t stream, Buffer<DataType::F32> x,
                        int64_t iDirection, int64_t iAxis, int64_t iNorm,
                        Result<dtype> y) {

  CCO::CollectiveOps ops;
  AXIS axis = static_cast<AXIS>(iAxis);
  DIRECTION direction = static_cast<DIRECTION>(iDirection);
  NORM norm = static_cast<NORM>(iNorm);

  assertm(axis == AXIS::X, "Only X axis is supported");

  ncclComm_t comm = CCO::NCCLOps::get_comm();
  const int &rank = CCO::NCCLOps::get_rank();
  const int &size = CCO::NCCLOps::get_size();
  matx::index_t dims[3]{1, 1, 1};
  Dimensions buffer_dims = x.dimensions();
  std::copy(buffer_dims.begin(), buffer_dims.end(), dims);

  auto butterfly_comm = ButterflyCommIndex::get_or_create_comms(comm);
  auto tensor_s = matx::make_tensor(x.typed_data(), dims);
  auto tensor_d = matx::make_tensor(y->typed_data(), dims);
  CCO::ReductionOp reduc_op{CCO::ReducType::SUM};
  
  auto comm_iter = 

  for (ncclComm_t comm : butterfly_comm) {
    ops.allreduce(x.typed_data(), y->typed_data(), x.element_count(), reduc_op,
                  comm, stream);
  }

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
