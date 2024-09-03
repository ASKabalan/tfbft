import jax
import jax.numpy as jnp
import numpy as np
import jax.extend as jex
from nccl_mpi_benchmarks import Backend, Mode, Collective


def add_element(a, scaler=1.):

    if a.dtype != jnp.float32:
        raise ValueError("Only float32 is supported")
    if type(scaler) != float:
        raise ValueError("Only float32 is supported")

    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)

    return jex.ffi.ffi_call("add_element",
                            out_type,
                            a,
                            scaler=np.float32(scaler),
                            vectorized=False)


def collective_call(operand , backend : Backend , collective : Collective, mode : Mode):

  assert isinstance(backend, Backend) , f"backend should be instance of Backend ({[b for b in Backend]})"
  assert isinstance(collective, Collective) , f"collective should be instance of Collective ({[c for c in Collective]})"
  assert isinstance(mode, Mode) , f"mode should be instance of Mode ({[m for m in Mode]})"
  assert operand.dtype == jnp.float32 , f"operand should be of type float32"

  out_type = jax.ShapeDtypeStruct(operand.shape, operand.dtype)

  return jex.ffi.ffi_call("collective_call",
                          out_type,
                          operand,
                          backend=backend.value,
                          collective=collective.value,
                          mode=mode.value,
                          vectorized=False)

