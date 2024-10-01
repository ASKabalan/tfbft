
#include "collectives/collective_ops.hpp"
#include "extensions.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>


namespace ffi = xla::ffi;

ffi::Error AllReduceNCCLImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> x,
                       ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {

  CCO::CollectiveOps ops;
  CCO::ReductionOp op(CCO::ReducType::SUM);
  ncclComm_t comm = CCO::NCCLOps::get_comm();
  ops.allreduce(x.typed_data() , y->typed_data(), x.element_count(), op, comm , stream);

  return ffi::Error::Success();
}

ffi::Error AllReduceMPIImpl(cudaStream_t stream,ffi::Buffer<ffi::DataType::F32> x,
                       ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {

  CCO::CollectiveOps ops;
  CCO::ReductionOp op(CCO::ReducType::SUM);
  MPI_Comm mpi_comm = CCO::MPIOps::get_comm();
  ops.allreduce(x.typed_data() , y->typed_data(), x.element_count(), op, mpi_comm);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllReduceNCCLF32, AllReduceNCCLImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>() // x
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllReduceMPIF32, AllReduceMPIImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>() // x
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);


