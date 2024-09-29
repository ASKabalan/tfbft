#ifndef FFI_HELPER_H_
#define FFI_HELPER_H_

#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "Failed: CUDA error " << __FILE__ << ":" << __LINE__        \
                << " '" << cudaGetErrorString(e) << "'" << std::endl;          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECKLASTERROR()                                                   \
  do {                                                                         \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "Failed: CUDA error " << __FILE__ << ":" << __LINE__        \
                << " '" << cudaGetErrorString(e) << "'" << std::endl;          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define assertm(exp, msg) assert(((void)msg, exp))

using Error_Type = xla::ffi::Error;
using DataType = xla::ffi::DataType;

template <DataType dtype>
using Buffer = xla::ffi::Buffer<dtype>;
template <DataType dtype, size_t rank = 3>
using Result = xla::ffi::Result<xla::ffi::Buffer<dtype>>;

using Dimensions = xla::ffi::AnyBuffer::Dimensions;
using FFI_Stream_Type = xla::ffi::PlatformStream<cudaStream_t>;
template <DataType dtype> using NativeType = xla::ffi::NativeType<dtype>;

static inline xla::ffi::Error ffi_with_cuda_error_check() {
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                           std::string("CUDA error: ") +
                               cudaGetErrorString(last_error));
  }
  return xla::ffi::Error::Success();
}

#endif // FFI_HELPER_ss
