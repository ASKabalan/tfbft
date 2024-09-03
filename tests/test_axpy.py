import jax
import nccl_mpi_benchmarks as nmb
import jax.numpy as jnp
from numpy.testing import assert_allclose
import pytest
from conftest import initialize_distributed

# Initialize JAX
initialize_distributed()
rank = jax.process_index()
size = jax.process_count()

@pytest.mark.skipif(jax.device_count() != 1, reason="Test only runs on single device")
def test_axpy():
    a = jnp.array([1, 2, 3], dtype=jnp.float32)

    # Add 1 to each element of the array
    b = nmb.ops.add_element(a, scaler=1.5)

    c = a + 1.5

    assert_allclose(b, c)
    print(f"Original array: {a}")
    print(f"Array after adding 1 to each element With cuda code: {b}")
    print(f"Array after adding 1 to each element With XLA: {c}")

