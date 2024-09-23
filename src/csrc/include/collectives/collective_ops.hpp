#ifndef COLLECTIVE_OPS_H
#define COLLECTIVE_OPS_H

#include "cuda_runtime.h"
#include "matx.h" // for matx types
#include "mpi.h"
#include "nccl.h"
#include "nccl_ops.hpp"


#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                           \
    if (r != ncclSuccess) {                         \
        std::cerr << "Failed: NCCL error " << __FILE__ << ":" << __LINE__ << " '" << ncclGetErrorString(r) << "'" << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)


namespace CCO {

// Enum to abstract data types for both MPI and NCCL
template <typename T> class ReductionOp {
  enum class Type { SUM, PROD, MIN, MAX, CUSTOM };
  ReductionOp(Type type) : type(type) {}
  ReductionOp(T alpha) : type(Type::CUSTOM), alpha(alpha) {}

private:
  Type type = Type::SUM;
  T alpha;
};

namespace Internal {

template <typename T> MPI_Datatype get_mpi_type(T);
#define REGISTER_MPI_TYPE(Type, MPI_T)                                         \
  template <> MPI_Datatype get_mpi_type<Type>(Type) { return MPI_T; }

REGISTER_MPI_TYPE(int, MPI_INT);
REGISTER_MPI_TYPE(unsigned int, MPI_UNSIGNED);
REGISTER_MPI_TYPE(float, MPI_FLOAT);
REGISTER_MPI_TYPE(double, MPI_DOUBLE);
REGISTER_MPI_TYPE(cuda::std::complex<float>, MPI_COMPLEX);
REGISTER_MPI_TYPE(cuda::std::complex<double>, MPI_DOUBLE_COMPLEX);

template <typename T> ncclDataType_t get_nccl_type();
#define REGISTER_NCCL_TYPE(Type, NCCL_T)                                       \
  template <> ncclDataType_t get_nccl_type<Type>() { return NCCL_T; }

REGISTER_NCCL_TYPE(int, ncclInt);
REGISTER_NCCL_TYPE(unsigned int, ncclUint32);
REGISTER_NCCL_TYPE(float, ncclFloat32);
REGISTER_NCCL_TYPE(double, ncclFloat64);
REGISTER_NCCL_TYPE(cuda::std::complex<float>, ncclFloat32);
REGISTER_NCCL_TYPE(cuda::std::complex<double>, ncclFloat64);
REGISTER_NCCL_TYPE(matxFp16, ncclFloat16);
REGISTER_NCCL_TYPE(__half, nccFloat16);
REGISTER_NCCL_TYPE(matxBf16, nccBfloat16);
REGISTER_NCCL_TYPE(__nv_bfloat16, nccBfloat16);

template <typename T> MPI_Op get_mpi_op(ReductionOp<T> op) {
  switch (op.type) {
  case ReductionOp<T>::Type::SUM:
    return MPI_SUM;
  case ReductionOp<T>::Type::PROD:
    return MPI_PROD;
  case ReductionOp<T>::Type::MIN:
    return MPI_MIN;
  case ReductionOp<T>::Type::MAX:
    return MPI_MAX;
  case ReductionOp<T>::Type::CUSTOM:
    return MPI_NO_OP;
  }
}

template <typename T> ncclRedOp_t get_nccl_op(ReductionOp<T> op) {
  switch (op.type) {
  case ReductionOp<T>::Type::SUM:
    return ncclSum;
  case ReductionOp<T>::Type::PROD:
    return ncclProd;
  case ReductionOp<T>::Type::MIN:
    return ncclMin;
  case ReductionOp<T>::Type::MAX:
    return ncclMax;
  case ReductionOp<T>::Type::CUSTOM:
    return ncclMaxRedOp;
  }
}

} // namespace Internal

//
//

class CollectiveOps {
public:
  virtual ~CollectiveOps() = default;

  // Pure virtual functions for collective operations

  // ********************************
  // MPI Implementation
  // ********************************

  template <typename T>
  void allreduce(T *sendbuf, T *recvbuf, size_t count, ReductionOp<T> op,
                 MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type<T>(count);
    MPI_Op mpi_op = Internal::get_mpi_op<T>(op);
    MPI_Allreduce(sendbuf, recvbuf, count, dtype, mpi_op, comm);
  }

  template <typename T>
  void reduce(T *sendbuf, T *recvbuf, size_t count, ReductionOp<T> op, int root,
              MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type<T>(count);
    MPI_Op mpi_op = Internal::get_mpi_op<T>(op);
    MPI_Reduce(sendbuf, recvbuf, count, dtype, mpi_op, root, comm);
  }

  template <typename T>
  void allgather(T *sendbuf, T *recvbuf, size_t count, MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type<T>(count);
    MPI_Allgather(sendbuf, count, dtype, recvbuf, count, dtype, comm);
  }

  template <typename T>
  void alltoall(T *sendbuf, T *recvbuf, size_t count, MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type<T>(count);
    MPI_Alltoall(sendbuf, count, dtype, recvbuf, count, dtype, comm);
  }

  template <typename T>
  void broadcast(T *buffer, size_t count, int root, MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type<T>(count);
    MPI_Bcast(buffer, count, dtype, root, comm);
  }

  // ********************************
  // NCCL implementation
  // ********************************

  template <typename T>
  void allreduce(T *sendbuf, T *recvbuf, size_t count, ReductionOp<T> op,
                 ncclComm_t comm, cudaStream_t stream) {
    ncclDataType_t dtype = Internal::get_nccl_type<T>();
    ncclRedOp_t nccl_op = Internal::get_nccl_op<T>(op);
    ncclAllReduce(sendbuf, recvbuf, count, dtype, nccl_op, comm, stream);
  }

  template <typename T>
  void reduce(T *sendbuf, T *recvbuf, size_t count, ReductionOp<T> op, int root,
              ncclComm_t comm, cudaStream_t stream) {
    ncclDataType_t dtype = Internal::get_nccl_type<T>();
    ncclRedOp_t nccl_op = Internal::get_nccl_op<T>(op);
    ncclReduce(sendbuf, recvbuf, count, dtype, nccl_op, root, comm, stream);
  }

  template <typename T>
  void allgather(T *sendbuf, T *recvbuf, size_t count, ncclComm_t comm,
                 cudaStream_t stream) {
    ncclDataType_t dtype = Internal::get_nccl_type<T>();
    ncclAllGather(sendbuf, recvbuf, count, dtype, comm, stream);
  }

  template <typename T>
  void alltoall(T *sendbuf, T *recvbuf, size_t count, ncclComm_t comm,
                cudaStream_t stream) {

    const int &rank = NCCLOps::get_rank();
    const int &nranks = NCCLOps::get_size();
    size_t chunk_size = count / nranks;
    ncclDataType_t dtype = Internal::get_nccl_type<T>();
    ncclGroupStart();
    for (int r = 0; r < nranks; r++) {
      void *sendbuf_r = (void *)(sendbuf + r * chunk_size);
      void *recvbuf_r = (void *)(recvbuf + r * chunk_size);
      ncclSend(sendbuf_r, chunk_size, dtype, r, comm, stream);
      ncclRecv(recvbuf_r, chunk_size, dtype, r, comm, stream);
    }
    ncclGroupEnd();
  }

  template <typename T>
  void broadcast(T *buffer, size_t count, int root, ncclComm_t comm,
                 cudaStream_t stream) {
    ncclDataType_t dtype = Internal::get_nccl_type<T>();
    ncclBroadcast(buffer, count, dtype, root, comm, stream);
  }

private:
};

} // namespace CCO
#endif
