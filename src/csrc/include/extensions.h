#ifndef GPU_OPS_H_
#define GPU_OPS_H_

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

enum class DIRECTION {
  FORWARD = 1,
  INVERSE = -1
};

enum class NORM {
  NONE = 1,
  ORTHO = 2,
  FORWARD = 3,
  BACKWARD = 4,
};

enum class AXIS {
  X = 1,
  Y = 2,
  Z = 3
};

#define XLA_FFI_DECLARE_HANDLER_TYPES(fn) \
  XLA_FFI_DECLARE_HANDLER_SYMBOL(fn##F32); \
  XLA_FFI_DECLARE_HANDLER_SYMBOL(fn##F64); \
  XLA_FFI_DECLARE_HANDLER_SYMBOL(fn##C64); \
  XLA_FFI_DECLARE_HANDLER_SYMBOL(fn##C128); \


// AllReduces 
XLA_FFI_DECLARE_HANDLER_TYPES(AllReduceNCCL);
XLA_FFI_DECLARE_HANDLER_TYPES(AllReduceMPI);

// FFTs
XLA_FFI_DECLARE_HANDLER_TYPES(ButterFlyFFTHandler);

#endif // GPU_OPS_H_
