#include <cute/tensor.hpp>

using namespace cute;

template <typename T>
__device__ T cosine(T x);

template <>
__device__ float cosine(float x) {
    return ::cosf(x);
}

template <>
__device__ double cosine(double x) {
    return ::cos(x);
}

template <typename T>
__device__ T sine(T x);

template <>
__device__ float sine(float x) {
    return ::sinf(x);
}

template <>
__device__ double sine(double x) {
    return ::sin(x);
}

template <class TensorS, class ThreadLayout>
__global__ void Multiply(TensorS S, typename TensorS::value_type alpha, ThreadLayout) {
    int local_index = threadIdx.x + stride<1>(ThreadLayout{}) * threadIdx.y + stride<2>(ThreadLayout{}) * threadIdx.z;
    Tensor local_tile = S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile = local_partition(local_tile, ThreadLayout{}, local_index);

    transform(thr_tile, [&alpha](auto x) { return x * alpha; });
}

template <class TensorS, class TensorD, class ThreadLayout>
__global__ void Multiply(TensorS S, TensorD D, typename TensorS::value_type alpha, ThreadLayout) {

    int local_index = threadIdx.x + stride<1>(ThreadLayout{}) * threadIdx.y + stride<2>(ThreadLayout{}) * threadIdx.z;

    Tensor local_tile = S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile_S = local_partition(local_tile, ThreadLayout{}, local_index);

    Tensor local_tile_D = D(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile_D = local_partition(local_tile_D, ThreadLayout{}, local_index);

    transform(thr_tile_S, thr_tile_D, [&alpha](auto x) { return x * alpha; });
}

template <class TensorS, class ThreadLayout>
__global__ void ApplyTwiddle(TensorS S, int dim, int factor, int butterfly_rank, int device_count, ThreadLayout) {
    using Element = typename TensorS::value_type;

    int local_index = threadIdx.x + stride<1>(ThreadLayout{}) * threadIdx.y + stride<2>(ThreadLayout{}) * threadIdx.z;
    Tensor local_tile = S(make_coord(_, _, _), blockIdx.x, blockIdx.y, blockIdx.z);
    Tensor thr_tile = local_partition(local_tile, ThreadLayout{}, local_index);

    int N = (dim * device_count) / factor;
    int kmax = N / 2;
    int k = (threadIdx.x + dim * butterfly_rank) % kmax;
    // Twiddle factor exp (- 2 pi * k / N)  where is a range from 0 to to (N/2 - 1) and N is dim/factor
    // the range is split among the ranks .. so the first rank will get

    // auto twiddle = Element{cosine(2 * M_PI * k / N), -sine(2 * M_PI * k / N)};
    // Apply the twiddle factor
    // transform(thr_tile, [&twiddle](auto x) { return x * twiddle; });
}
