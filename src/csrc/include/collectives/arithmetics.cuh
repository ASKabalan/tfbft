#include <cute/tensor.hpp>

using namespace cute;

template <class TensorS, class ThreadLayout>
__global__ void Multiplies(TensorS S, typename TensorS::value_type alpha,
                           ThreadLayout) {

  using T = typename TensorS::value_type;

  Tensor tile = S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
  Tensor thr_tile = local_partition(tile, ThreadLayout{}, threadIdx.x);

  transform(thr_tile, thr_tile, [&alpha](T x) { return x * alpha; });
}
