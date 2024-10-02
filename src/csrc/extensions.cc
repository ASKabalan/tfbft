#include "extensions.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <type_traits>

namespace nb = nanobind;

template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
    // This check is optional, but it can be helpful for avoiding invalid
    // handlers.
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

nb::dict Registrations() {
    nb::dict d;
    // d["all_reduce_nccl"] = EncapsulateFfiCall(AllReduceNCCLF32);
    // d["all_reduce_mpi"] = EncapsulateFfiCall(AllReduceMPIF32);
    d["butterfly_fft_c64"] = EncapsulateFfiCall(ButterFlyFFTHandlerC64);
    d["butterfly_fft_c128"] = EncapsulateFfiCall(ButterFlyFFTHandlerC128);
    return d;
}

NB_MODULE(butterfly_fft_lib, m) { m.def("registrations", &Registrations); }
