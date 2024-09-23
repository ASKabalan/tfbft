import jax
import jax.numpy as jnp
import numpy as np
import jax.extend as jex


def xla_call(operand):

    assert operand.dtype == jnp.float32, f"operand should be of type float32"

    out_type = jax.ShapeDtypeStruct(operand.shape, operand.dtype)

    return jex.ffi.ffi_call("xla_call",
                            out_type,
                            operand,
                            vectorized=False)
