import butterfly_fft.butterfly_fft_lib as gpu_ops
import jax.extend as jex

for name, fn in gpu_ops.registrations().items():
    jex.ffi.register_ffi_target(name,
                                fn,
                                platform="CUDA")
