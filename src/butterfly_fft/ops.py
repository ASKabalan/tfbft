import jax
import jax.numpy as jnp
import numpy as np
import jax.extend as jex


def all_reduce_nccl(operand):

    assert operand.dtype == jnp.float32, f"operand should be of type float32"

    out_type = jax.ShapeDtypeStruct(operand.shape, operand.dtype)

    return jex.ffi.ffi_call("all_reduce_nccl",
                            out_type,
                            operand,
                            vectorized=False)

def all_reduce_mpi(operand):

    assert operand.dtype == jnp.float32, f"operand should be of type float32"

    out_type = jax.ShapeDtypeStruct(operand.shape, operand.dtype)

    return jex.ffi.ffi_call("all_reduce_mpi",
                            out_type,
                            operand,
                            vectorized=False)
