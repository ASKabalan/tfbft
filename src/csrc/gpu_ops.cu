#include "gpu_ops.h"
#include <cuda_runtime.h>

__global__ void add_element_kernel(float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * 2;
  }
}

void add_element(float *x, float *y, int n, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  add_element_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, n);
}

