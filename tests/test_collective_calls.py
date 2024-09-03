import jax
from conftest import initialize_distributed
import nccl_mpi_benchmarks as nmb
import jax.numpy as jnp
from jax.sharding import Mesh , PartitionSpec as P , NamedSharding
from jax.experimental import mesh_utils , multihost_utils
from jax.experimental.shard_map import shard_map
from functools import partial
import pytest
from numpy.testing import assert_array_equal

# Initialize JAX
initialize_distributed()
rank = jax.process_index()
size = jax.process_count()

@pytest.fixture(scope="module", autouse=True)
def create_global_array(global_shape=(32,)):
  import jax.numpy as jnp
  from jax.experimental import mesh_utils
  from jax.sharding import Mesh , PartitionSpec as P , NamedSharding
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  mesh = Mesh(devices, ('gpus',))
  sharding = NamedSharding(mesh , P('gpus',))

  local_shape = (global_shape[0] // jax.process_count(),)
  gspmd_array = jax.make_array_from_callback(global_shape , sharding, lambda _: jnp.ones(local_shape, dtype=jnp.float32) * jax.process_index())

  return gspmd_array , mesh


@pytest.mark.skipif(jax.device_count() < 2, reason="Test only runs on multi device")
def test_collective_calls(create_global_array):
  gspmd_array , mesh = create_global_array

  @partial(shard_map , mesh=mesh, in_specs=P('gpus') , out_specs=P('gpus'),check_rep=False)
  def all_reduce(operand):

    return nmb.ops.collective_call(operand , nmb.Backend.NCCL, nmb.Collective.AllReduce, nmb.Mode.OutOfPlace)


  result = all_reduce(gspmd_array)
  gathered_array = multihost_utils.process_allgather(gspmd_array , tiled=True)
  gathered_result = multihost_utils.process_allgather(result , tiled=True)

  slices = jnp.split(gathered_array , size)
  jnp_result = jnp.tile(sum(slices) , size) 

  print(f"Original array: {gathered_array}")
  print(f"Array after all reduce operation: {gathered_result}")
  print(f"Array after all reduce operation With jnp: {jnp_result}")
  
  assert_array_equal(gathered_result , jnp_result)

  


