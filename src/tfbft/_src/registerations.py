import nccl_mpi_benchmarks.gpu_ops as gpu_ops
import jax.extend as jex


for name, fn in gpu_ops.registrations().items():
    
    jex.ffi.register_ffi_target(name,
                                fn,
                                platform="CUDA")
