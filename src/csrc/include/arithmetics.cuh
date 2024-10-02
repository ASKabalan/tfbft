#include <cute/tensor.hpp>

using namespace cute;

template <class TensorS, class ThreadLayout>
__global__ void Multiply(TensorS S, typename TensorS::value_type alpha, ThreadLayout) {
    int local_index = threadIdx.x + stride<1>(ThreadLayout{}) * threadIdx.y + stride<1>(ThreadLayout{}) * stride<2>(ThreadLayout{}) * threadIdx.z;
    Tensor local_tile = S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile = local_partition(local_tile, ThreadLayout{}, local_index);

    transform(thr_tile, [&alpha](auto x) { return x * alpha; });
}

template <class TensorS, class TensorD, class ThreadLayout>
__global__ void Multiply(TensorS S, TensorD D, typename TensorS::value_type alpha, ThreadLayout) {
    int local_index = threadIdx.x + stride<1>(ThreadLayout{}) * threadIdx.y + stride<1>(ThreadLayout{}) * stride<2>(ThreadLayout{}) * threadIdx.z;
    Tensor local_tile = S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile_S = local_partition(local_tile, ThreadLayout{}, local_index);

    Tensor local_tile_D = D(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile_D = local_partition(local_tile_D, ThreadLayout{}, local_index);

    Tensor fragment = make_fragment_like(thr_tile_S);

    copy(thr_tile_S, fragment);
    transform(fragment, [&alpha](auto x) { return x * alpha; });
    copy(fragment, thr_tile_D);
}

template <class TensorS, class ThreadLayout>
__global__ void ApplyTwiddle(TensorS S, int dim, int factor, int device_rank, int device_count, ThreadLayout) {
    using Element = typename TensorS::value_type;

    int local_index = threadIdx.x + stride<1>(ThreadLayout{}) * threadIdx.y + stride<1>(ThreadLayout{}) * stride<2>(ThreadLayout{}) * threadIdx.z;
    Tensor local_tile = S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile = local_partition(local_tile, ThreadLayout{}, local_index);
    int N = dim / factor;
    int position_in_b_partition = device_rank % factor;
    // Twiddle factor exp (- 2 pi * k / N)  where is a range from 0 to to (N/2 - 1) and N is dim/factor
    // the range is split among the ranks .. so the first rank will get

    // auto twiddle = Element{0, -1} * Element{0, 2 * M_PI * device_rank * local_index / size};

    // transform(thr_tile, [twiddle](auto x) { return x * twiddle; });
}
