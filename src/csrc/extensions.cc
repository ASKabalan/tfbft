#include "collectives/collective_ops.hpp"
#include "gpu_ops.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <type_traits>

namespace ffi = xla::ffi;
namespace nb = nanobind;

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllReduceNCCL, AllReduceNCCLImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>() // x
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllReduceMPI, AllReduceMPIImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>() // x
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

nb::dict Registrations() {
  nb::dict d;
  d["all_reduce_nccl"] = EncapsulateFfiCall(AllReduceNCCL);
  d["all_reduce_mpi"] = EncapsulateFfiCall(AllReduceMPI);
  return d;
}

NB_MODULE(butterfly_fft_lib, m) { m.def("registrations", &Registrations); }
