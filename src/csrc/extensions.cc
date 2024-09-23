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

ffi::Error XlaCallImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> x,
                       ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {
  add_element(x.typed_data(), y->typed_data(), x.element_count(), stream);
  auto dims = x.dimensions();
  std::cout << "size of x = " << dims.size() << std::endl;
  for (auto d : dims) {
    std::cout << "dim = " << d << std::endl;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(XlaCall, XlaCallImpl,
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
  d["xla_call"] = EncapsulateFfiCall(XlaCall);
  return d;
}

NB_MODULE(butterfly_fft_lib, m) { m.def("registrations", &Registrations); }
