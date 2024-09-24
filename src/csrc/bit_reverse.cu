
#include <cute/tensor.hpp>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace cute;

template <class TensorS, class ThreadLayout>
__global__ void bit_reverse_kernel(TensorS S, ThreadLayout) {

  using Element = typename TensorS::value_type;

  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int tid_z = threadIdx.z + blockIdx.z * blockDim.z;
  if (tid >= size<0>(S) || tid_y >= size<1>(S) || tid_z >= size<2>(S))
    return;

  unsigned int bit_reversed_tid = __brev(tid) >> (__clz(size<0>(S)) + 1);
  unsigned int bit_reversed_blockIdx =
      bit_reversed_tid / (unsigned int)blockDim.x;
  unsigned int bit_reversed_threadIdx =
      bit_reversed_tid % (unsigned int)blockDim.x;

  if (tid < bit_reversed_tid)
    return;

  if (tid == bit_reversed_tid && blockIdx.x == bit_reversed_blockIdx)
    return;

  Tensor natural_order_tile =
      S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
  Tensor reversed_order_tile =
      S(make_coord(_, _, _), bit_reversed_blockIdx, blockIdx.y, blockIdx.z);

  auto natural_order_threadid =
      threadIdx.x + threadIdx.y * stride<1>(ThreadLayout{}) +
      threadIdx.z * stride<2>(ThreadLayout{}) * stride<1>(ThreadLayout{});
  auto bit_reversed_threadid =
      bit_reversed_threadIdx + threadIdx.y * stride<1>(ThreadLayout{}) +
      threadIdx.z * stride<2>(ThreadLayout{}) * stride<1>(ThreadLayout{});

  Tensor natural_order_thr = local_partition(natural_order_tile, ThreadLayout{},
                                             natural_order_threadid);
  Tensor reversed_order_thr = local_partition(
      reversed_order_tile, ThreadLayout{}, bit_reversed_threadid);
  // Copy from GMEM to RMEM
  Tensor rmem_tensor = make_fragment_like(natural_order_thr);

  copy(natural_order_thr, rmem_tensor);
  copy(reversed_order_thr, natural_order_thr);
  copy(rmem_tensor, reversed_order_thr);
}

template <ffi::DataType T>
ffi::Error bit_reverse_tensor(cudaStream_t stream , ffi::Buffer<T> &buffer)
{ 
  auto dimensions = buffer.dimensions();
  auto shape = make_shape(dimensions[0], dimensions[1], dimensions[2]);
  Tensor tensor_S = make_tensor(make_gmem_ptr(buffer.typed_data()) , make_layout(shape));

  auto block_shape = Shape<_64, _4, _4>{};
  auto thread_layout = make_layout(Shape<_64, _4, _4>{});
  Tensor tiled_tensor = tiled_divide(tensor_S, block_shape);

  dim3 gridDim(size<1>(tiled_tensor), size<2>(tiled_tensor), size<3>(tiled_tensor));
  dim3 blockDim(shape<0>(thread_layout), shape<1>(thread_layout), shape<2>(thread_layout));

  bit_reverse<<<gridDim, blockDim, 0, stream>>>(tiled_tensor, thread_layout);
}


