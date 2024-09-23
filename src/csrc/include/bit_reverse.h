
#include <cute/tensor.hpp>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <class Buffer>
ffi::Error bit_reverse_tensor(Buffer& S);



template<> ffi::Error bit_reverse_tensor(ffi::Buffer<ffi::DataType::F16>&);
template<> ffi::Error bit_reverse_tensor(ffi::Buffer<ffi::DataType::F32>&);
template<> ffi::Error bit_reverse_tensor(ffi::Buffer<ffi::DataType::F64>&);
