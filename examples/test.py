import jax

jax.distributed.initialize()
size = jax.process_count()
rank = jax.process_index()

from jax import numpy as jnp
from butterfly_fft.ops import *
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial
devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ('x', None))
sharding = NamedSharding(mesh, P('x'))

global_shape = (16, 16)
local_shape = (4 ,16)
a = jax.make_array_from_callback(
    shape=global_shape,
    sharding=sharding,
    data_callback=lambda _: jnp.ones(local_shape, dtype=jnp.float32))


@partial(shard_map , mesh=mesh ,in_specs=P('x'), out_specs=P('x') , check_rep=False)
def all_reduce_n(operand):
    return all_reduce_nccl(operand)

@partial(shard_map , mesh=mesh ,in_specs=P('x'), out_specs=P('x') , check_rep=False)
def all_reduce_m(operand):
    return all_reduce_mpi(operand)

a_n = all_reduce_n(a)

a_n = multihost_utils.process_allgather(a_n , tiled=True)
a = multihost_utils.process_allgather(a , tiled=True)

print(f"a_n = \n{a_n}")
print(f"a = \n{a}")
