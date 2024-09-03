#include <cuda_runtime.h>

void add_element(float scaler, float *x, float *y, int n, cudaStream_t stream);
