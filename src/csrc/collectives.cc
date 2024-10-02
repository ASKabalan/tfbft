
#include "extensions.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include "collectives/nccl_ops.hpp"
#include "collectives/mpi_ops.hpp"

namespace ffi = xla::ffi;

ffi::Error AllReduceNCCLImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> x, ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {
    ncclComm_t comm = CCO::NCCLOps::get_comm();
    int rank = CCO::NCCLOps::get_rank();

    int color = rank % 2;
    ncclComm_t split_comm;
    ncclCommSplit(comm, color, rank, &split_comm, nullptr);

    int local_rank;
    int local_count;

    ncclCommUserRank(split_comm, &local_rank);
    ncclCommCount(split_comm, &local_count);

    std::cout << "NCCL Rank: " << rank << " NCCL Local Rank: " << local_rank << " NCCL Local Count: " << local_count << std::endl;

    ncclAllReduce(x.untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, split_comm, stream);

    return ffi::Error::Success();
}

ffi::Error AllReduceMPIImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> x, ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {
    MPI_Comm mpi_comm = CCO::MPIOps::get_comm();
    // ops.allreduce(x.typed_data(), y->typed_data(), x.element_count(), op,
    //               mpi_comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllReduceNCCLF32, AllReduceNCCLImpl,
                              ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                      .Arg<ffi::Buffer<ffi::DataType::F32>>()  // x
                                      .Ret<ffi::Buffer<ffi::DataType::F32>>()  // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllReduceMPIF32, AllReduceMPIImpl,
                              ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                      .Arg<ffi::Buffer<ffi::DataType::F32>>()  // x
                                      .Ret<ffi::Buffer<ffi::DataType::F32>>()  // y
);
