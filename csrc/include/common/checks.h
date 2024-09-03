#include <nccl.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                    \
    if( e != MPI_SUCCESS ) {                        \
        std::cerr << "Failed: MPI error " << __FILE__ << ":" << __LINE__ << " '" << e << "'" << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                            \
    if( e != cudaSuccess ) {                        \
        std::cerr << "Failed: CUDA error " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(e) << "'" << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                           \
    if (r != ncclSuccess) {                         \
        std::cerr << "Failed: NCCL error " << __FILE__ << ":" << __LINE__ << " '" << ncclGetErrorString(r) << "'" << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)
