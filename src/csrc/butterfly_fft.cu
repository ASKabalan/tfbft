#include "extensions.h"
#include <cute/tensor.hpp>
// #include <cute/layout.hpp>
// #include <cute/stride.hpp>
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "arithmetics.cuh"
#include "collectives/butterfly_comm_index.hpp"
#include "common/ffi_helper.hpp"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
#include <nccl.h>
#include "collectives/nccl_ops.hpp"

namespace ffi = xla::ffi;
using namespace cute;

template <DataType dtype>
ffi::Error ButterFlyForward(cudaStream_t stream, Buffer<dtype> x, NORM iNorm, Result<dtype> y, CommIterator &comms) {
    const int &device_rank = CCO::NCCLOps::get_rank();
    const int &device_count = CCO::NCCLOps::get_size();
    const int global_axis_size = x.dimensions()[0] * device_count;
    // Get the cute Tensor from the XLA Buffer
    Dimensions dims = x.dimensions();
    assertm(dims.size() == 3, "Only 3D tensors are supported for now");
    // Create a static rank 3 tensor from the buffer
    auto tensor_shape = make_shape(dims[0], dims[1], dims[2]);
    Tensor tensor_S = make_tensor(make_gmem_ptr(x.typed_data()), make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(make_gmem_ptr(y->typed_data()), make_layout(tensor_shape));
    using TensorType = typename decltype(tensor_S)::value_type;
    // Create the kernel configuration and the tiled tensor
    auto block_shape = Shape<_64, _4, _4>{};
    Layout thread_layout = make_layout(Shape<_64, _4, _4>{});  // 256 threads one thread per element
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);
    // Grid dim is the result of the division of the tensor shape by the block Shape
    // Thread layout is the shape of the block to make sure that each thread is responsible for one element
    dim3 gridDim(size<1>(tiled_tensor_S), size<2>(tiled_tensor_S), size<3>(tiled_tensor_S));
    dim3 blockDim(shape<0>(thread_layout), shape<1>(thread_layout), shape<2>(thread_layout));
    // Get the first stage comm .. this is called in a distributed setup and the first step
    // Means that we have atleast 2 devices in the comm .. otherwise this is an error

    auto stageComm = comms.next();
    assertm(stageComm.has_value(), "[INTERNAL] Distributed FFT called without Buttefly comms");
    // Get the Stage comm and the N (normalization factor for this stage)
    auto [N, comm] = stageComm.value();
    int stage_rank;
    NCCLCHECK(ncclCommUserRank(comm, &stage_rank));
    // First step A + B
    ncclGroupStart();
    ncclReduce(x.untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 0, comm, stream);

    // Second step A - B
    // First add -2 to B to prepare for => -2B + (A + B) = A - B
    if (stage_rank == 1) {
        Multiply<<<gridDim, blockDim, 0, stream>>>(tiled_tensor_S, tiled_tensor_D, (TensorType)-2, thread_layout);
    }
    // ncclReduce(y->untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 1, comm, stream);
    std::cout << "Reduced " << std::endl;
    // Apply the twiddle factor
    // Check if current device holds the A or B
    if (stage_rank == 1) {
        // ApplyTwiddle<<<gridDim, blockDim, 0, stream>>>(tiled_tensor_S, global_axis_size, N, device_rank, device_count, thread_layout);
    }

    // Continue with the rest of the stages
    while ((stageComm = comms.next())) {
        auto [N, comm] = stageComm.value();

        int stage_rank;
        int stage_count;

        NCCLCHECK(ncclCommUserRank(comm, &stage_rank));
        NCCLCHECK(ncclCommCount(comm, &stage_count));

        std::cout << "For the next stage N " << N << " stage rank " << stage_rank << " stage count " << stage_count << std::endl;
    }
    ncclGroupEnd();

    return ffi_with_cuda_error_check();
}

template <DataType dtype>
ffi::Error ButterFlyBackward(cudaStream_t stream, Buffer<DataType::F32> x, NORM iNorm, Result<dtype> y, CommIterator &comms) {
    comms.reset();
    auto comm = comms.prev();
    assertm(comm.has_value(), "[INTERNAL] Distributed IFFT called without Buttefly comms");

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
ffi::Error ButterFlyFFT(cudaStream_t stream, Buffer<DataType::F32> x, int64_t iDirection, int64_t iAxis, int64_t iNorm, Result<dtype> y) {
    AXIS axis = static_cast<AXIS>(iAxis);
    DIRECTION direction = static_cast<DIRECTION>(iDirection);
    NORM norm = static_cast<NORM>(iNorm);

    assertm(axis == AXIS::X, "Only X axis is supported");

    ncclComm_t comm = CCO::NCCLOps::get_comm();

    auto butterfly_comm = ButterflyCommIndex::get_or_create_comms(comm);

    switch (direction) {
        case DIRECTION::FORWARD:
            return ButterFlyForward(stream, x, norm, y, butterfly_comm);
        case DIRECTION::INVERSE:
            return ButterFlyBackward(stream, x, norm, y, butterfly_comm);
        default:
            return ffi::Error(XLA_FFI_Error_Code_INTERNAL, std::string("Un recongnized FFT direction "));
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ButterFlyFFTHandlerF32, ButterFlyFFT<DataType::F32>,
                              ffi::Ffi::Bind()
                                      .Ctx<FFI_Stream_Type>()
                                      .Arg<Buffer<DataType::F32>>()  // x
                                      .Attr<int64_t>("direction")
                                      .Attr<int64_t>("norm")
                                      .Attr<int64_t>("axis")
                                      .Ret<Buffer<DataType::F32>>()  // y
);
