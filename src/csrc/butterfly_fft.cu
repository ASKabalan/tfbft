#include "extensions.h"
#include <cute/tensor.hpp>
#include <cute/tensor.hpp>
#include "cute/tensor_impl.hpp"
#include "cute/stride.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include <cute/numeric/complex.hpp>
#include <cutlass/complex.h>
#include "arithmetics.cuh"
#include "collectives/butterfly_comm_index.hpp"
#include "collectives/nccl_ops.hpp"
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/complex.hpp>
// #include <cutlass/complex.h>
#include "arithmetics.cuh"
#include "common/ffi_helper.hpp"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
// #include <nccl.h>

namespace ffi = xla::ffi;
using namespace cute;
template <typename T>
using cuteComplex = cutlass::complex<T>;

int compute_b_rank(int device_rank, int stage_size) {
    return device_rank - stage_size * (1 + (device_rank / stage_size) / 2);
}

template <DataType dtype>
ffi::Error ButterFlyForward(cudaStream_t stream, Buffer<dtype> x, NORM iNorm, Result<dtype> y,
                            CommIterator &comms) {
    static_assert(ffi::IsComplexType<dtype>(), "Only complex types are supported");
    using Real = ffi::NativeType<ffi::ToReal(dtype)>;
    using Complex = cuteComplex<Real>;

    const int &device_rank = CCO::NCCLOps::get_rank();
    const int &device_count = CCO::NCCLOps::get_size();
    const int global_axis_size = x.dimensions()[0] * device_count;
    // Get the cute Tensor from the XLA Buffer
    Dimensions dims = x.dimensions();
    assertm(dims.size() == 3, "Only 3D tensors are supported for now");
    // Create a static rank 3 tensor from the buffer
    auto tensor_shape = make_shape(dims[0], dims[1], dims[2]);
    auto x_ptr = reinterpret_cast<Complex *>(x.typed_data());
    auto y_ptr = reinterpret_cast<Complex *>(y->typed_data());
    Tensor tensor_S = make_tensor(x_ptr, make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(y_ptr, make_layout(tensor_shape));
    // Create the kernel configuration and the tiled tensor
    auto block_shape = Shape<_2, _2, _2>{};
    Layout thread_layout = make_layout(Shape<_2, _2, _2>{});  // 256 threads one thread per element
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
    auto [factor, comm] = stageComm.value();
    int stage_rank, stage_size(device_count / factor);
    NCCLCHECK(ncclCommUserRank(comm, &stage_rank));
    // First step A + B
    ncclGroupStart();
    ncclReduce(x.untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 0, comm, stream);
    // Second step A - B
    // First multipy -2 to B to prepare for => -2B + (A + B) = A - B
    if (stage_rank == 1) {
        Multiply<<<gridDim, blockDim, 0, stream>>>(tiled_tensor_S, tiled_tensor_D, Complex{-2, 0},
                                                   thread_layout);
    }
    ncclReduce(y->untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 1, comm, stream);
    // Apply the twiddle factor
    // Check if current device holds the A or B
    // in DIF only the B ranks are twiddled
    if (stage_rank == 1) {
        int b_rank = compute_b_rank(device_rank, stage_size);
        std::cout << "Twiddle for B rank " << b_rank << std::endl;
        ApplyTwiddle<DIRECTION::FORWARD><<<gridDim, blockDim, 0, stream>>>(
                tiled_tensor_S, global_axis_size, factor, b_rank, device_count, thread_layout);
    }

    // Continue with the rest of the stages
    while ((stageComm = comms.next())) {
        auto [factor, comm] = stageComm.value();

        int stage_rank, stage_size(device_count / factor);
        NCCLCHECK(ncclCommUserRank(comm, &stage_rank));
        // First step A + B
        ncclReduce(y->untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 0, comm,
                   stream);
        // Second step -2B + (A + B) = A - B
        if (stage_rank == 1) {
            Multiply<<<gridDim, blockDim, 0, stream>>>(tiled_tensor_D, Complex{-2, 0}, thread_layout);
        }
        ncclReduce(y->untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 1, comm,
                   stream);

        // Apply the twiddle factor
        // Check if current device holds the A or B
        if (stage_rank == 1) {
            int b_rank = compute_b_rank(device_rank, stage_size);
            std::cout << "Twiddle for B rank " << b_rank << std::endl;
            ApplyTwiddle<DIRECTION::FORWARD><<<gridDim, blockDim, 0, stream>>>(
                    tiled_tensor_D, global_axis_size, factor, b_rank, device_count, thread_layout);
        }
    }
    ncclGroupEnd();

    return ffi_with_cuda_error_check();
}

template <DataType dtype>
ffi::Error ButterFlyBackward(cudaStream_t stream, Buffer<dtype> x, NORM iNorm, Result<dtype> y,
                             CommIterator &comms) {
    static_assert(ffi::IsComplexType<dtype>(), "Only complex types are supported");
    using Real = ffi::NativeType<ffi::ToReal(dtype)>;
    using Complex = cuteComplex<Real>;

    const int &device_rank = CCO::NCCLOps::get_rank();
    const int &device_count = CCO::NCCLOps::get_size();
    const int global_axis_size = x.dimensions()[0] * device_count;

    // Get the cute Tensor from the XLA Buffer
    Dimensions dims = x.dimensions();
    assertm(dims.size() == 3, "Only 3D tensors are supported for now");
    auto tensor_shape = make_shape(dims[0], dims[1], dims[2]);
    auto x_ptr = reinterpret_cast<Complex *>(x.typed_data());
    auto y_ptr = reinterpret_cast<Complex *>(y->typed_data());
    Tensor tensor_S = make_tensor(x_ptr, make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(y_ptr, make_layout(tensor_shape));
    // Create the kernel configuration and the tiled tensor
    auto block_shape = Shape<_2, _2, _2>{};
    Layout thread_layout = make_layout(Shape<_2, _2, _2>{});
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);

    dim3 gridDim(size<1>(tiled_tensor_S), size<2>(tiled_tensor_S), size<3>(tiled_tensor_S));
    dim3 blockDim(shape<0>(thread_layout), shape<1>(thread_layout), shape<2>(thread_layout));

    // Reverse the comms iterator to start with the last stage
    comms.reverse();
    auto stageComm = comms.prev();
    assertm(stageComm.has_value(), "[INTERNAL] Distributed IFFT called without Butterfly comms");

    // Start with the last stage
    auto [factor, comm] = stageComm.value();
    int stage_rank, stage_size(device_count / factor);
    NCCLCHECK(ncclCommUserRank(comm, &stage_rank));

    // Step 1: Apply the twiddle factor first (opposite order of the forward FFT)
    if (stage_rank == 1) {
        int b_rank = compute_b_rank(device_rank, stage_size);
        ApplyTwiddle<DIRECTION::INVERSE><<<gridDim, blockDim, 0, stream>>>(
                tiled_tensor_S, global_axis_size, factor, b_rank, device_count, thread_layout);
    }

    // Step 2: Perform A + W * B and A - W * B using NCCL reduction
    ncclGroupStart();
    ncclReduce(x.untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 0, comm, stream);

    if (stage_rank == 1) {
        Multiply<<<gridDim, blockDim, 0, stream>>>(tiled_tensor_S, tiled_tensor_D, Complex{-2, 0},
                                                   thread_layout);
    }
    ncclReduce(y->untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 1, comm, stream);

    // Continue with the remaining stages in reverse order
    while ((stageComm = comms.prev())) {
        auto [factor, comm] = stageComm.value();

        NCCLCHECK(ncclCommUserRank(comm, &stage_rank));
        stage_size = device_count / factor;

        // Step 1: Apply the twiddle factor for this stage
        if (stage_rank == 1) {
            int b_rank = compute_b_rank(device_rank, stage_size);
            ApplyTwiddle<DIRECTION::INVERSE><<<gridDim, blockDim, 0, stream>>>(
                    tiled_tensor_D, global_axis_size, factor, b_rank, device_count, thread_layout);
        }

        // Step 2: Perform A + W * B and A - W * B using NCCL reduction
        ncclReduce(y->untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 0, comm,
                   stream);

        if (stage_rank == 1) {
            Multiply<<<gridDim, blockDim, 0, stream>>>(tiled_tensor_D, Complex{-2, 0}, thread_layout);
        }
        ncclReduce(y->untyped_data(), y->untyped_data(), x.element_count(), ncclFloat, ncclSum, 1, comm,
                   stream);
    }
    ncclGroupEnd();

    return ffi_with_cuda_error_check();
}
template <DataType dtype>
ffi::Error ButterFlyFFT(cudaStream_t stream, Buffer<dtype> x, int64_t iDirection, int64_t iAxis,
                        int64_t iNorm, Result<dtype> y) {
    static_assert(ffi::IsComplexType<dtype>(), "Only complex types are supported");

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(ButterFlyFFTHandlerC64, ButterFlyFFT<DataType::C64>,
                              ffi::Ffi::Bind()
                                      .Ctx<FFI_Stream_Type>()
                                      .Arg<Buffer<DataType::C64>>()  // x
                                      .Attr<int64_t>("direction")
                                      .Attr<int64_t>("norm")
                                      .Attr<int64_t>("axis")
                                      .Ret<Buffer<DataType::C64>>()  // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ButterFlyFFTHandlerC128, ButterFlyFFT<DataType::C128>,
                              ffi::Ffi::Bind()
                                      .Ctx<FFI_Stream_Type>()
                                      .Arg<Buffer<DataType::C128>>()  // x
                                      .Attr<int64_t>("direction")
                                      .Attr<int64_t>("norm")
                                      .Attr<int64_t>("axis")
                                      .Ret<Buffer<DataType::C128>>()  // y
);
