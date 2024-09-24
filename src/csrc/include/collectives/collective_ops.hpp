#ifndef COLLECTIVE_OPS_H
#define COLLECTIVE_OPS_H

#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include "nccl_ops.hpp"
#if __has_include("matx.h")
#define MATX_ENABLE_MATX
#include "matx.h" // for matx types
#endif

namespace CCO {

// Enum to abstract data types for both MPI and NCCL
enum class ReducType { SUM, PROD, MIN, MAX, CUSTOM };
template <typename T = float> struct ReductionOp {
  ReductionOp(ReducType type) : type(type) {}
  ReductionOp(T alpha) : type(ReducType::CUSTOM), alpha(alpha) {}
  const ReducType get_type() const { return type; }

private:
  ReducType type = ReducType::SUM;
  T alpha;
};

namespace Internal {

#define REGISTER_MPI_TYPE(Type, MPI_T)                                         \
  static inline MPI_Datatype get_mpi_type(Type) { return MPI_T; }

REGISTER_MPI_TYPE(int, MPI_INT);
REGISTER_MPI_TYPE(unsigned int, MPI_UNSIGNED);
REGISTER_MPI_TYPE(float, MPI_FLOAT);
REGISTER_MPI_TYPE(double, MPI_DOUBLE);
REGISTER_MPI_TYPE(cuda::std::complex<float>, MPI_COMPLEX);
REGISTER_MPI_TYPE(cuda::std::complex<double>, MPI_DOUBLE_COMPLEX);

#define REGISTER_NCCL_TYPE(Type, NCCL_T)                                       \
  static inline ncclDataType_t get_nccl_type(Type) { return NCCL_T; }

REGISTER_NCCL_TYPE(int, ncclInt);
REGISTER_NCCL_TYPE(unsigned int, ncclUint32);
REGISTER_NCCL_TYPE(float, ncclFloat32);
REGISTER_NCCL_TYPE(double, ncclFloat64);
REGISTER_NCCL_TYPE(cuda::std::complex<float>, ncclFloat32);
REGISTER_NCCL_TYPE(cuda::std::complex<double>, ncclFloat64);
REGISTER_NCCL_TYPE(__half, ncclFloat16);
REGISTER_NCCL_TYPE(__nv_bfloat16, ncclBfloat16);
#ifdef MATX_ENABLE_MATX
REGISTER_NCCL_TYPE(matx::matxFp16, ncclFloat16);
REGISTER_NCCL_TYPE(matx::matxBf16, ncclBfloat16);
#endif

template <typename T> MPI_Op get_mpi_op(ReductionOp<T> op) {
  switch (op.get_type()) {
  case ReducType::SUM:
    return MPI_SUM;
  case ReducType::PROD:
    return MPI_PROD;
  case ReducType::MIN:
    return MPI_MIN;
  case ReducType::MAX:
    return MPI_MAX;
  case ReducType::CUSTOM:
    return MPI_NO_OP;
  }
}

template <typename T> ncclRedOp_t get_nccl_op(ReductionOp<T> op) {
  switch (op.get_type()) {
  case ReducType::SUM:
    return ncclSum;
  case ReducType::PROD:
    return ncclProd;
  case ReducType::MIN:
    return ncclMin;
  case ReducType::MAX:
    return ncclMax;
  case ReducType::CUSTOM:
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
    MPI_Datatype dtype = Internal::get_mpi_type(*sendbuf);
    MPI_Op mpi_op = Internal::get_mpi_op(op);
    MPI_Allreduce(sendbuf, recvbuf, count, dtype, mpi_op, comm);
  }

  template <typename T>
  void reduce(T *sendbuf, T *recvbuf, size_t count, ReductionOp<T> op, int root,
              MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type(*sendbuf);
    MPI_Op mpi_op = Internal::get_mpi_op(op);
    MPI_Reduce(sendbuf, recvbuf, count, dtype, mpi_op, root, comm);
  }

  template <typename T>
  void allgather(T *sendbuf, T *recvbuf, size_t count, MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type(*sendbuf);
    MPI_Allgather(sendbuf, count, dtype, recvbuf, count, dtype, comm);
  }

  template <typename T>
  void alltoall(T *sendbuf, T *recvbuf, size_t count, MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type(*sendbuf);
    MPI_Alltoall(sendbuf, count, dtype, recvbuf, count, dtype, comm);
  }

  template <typename T>
  void broadcast(T *buffer, size_t count, int root, MPI_Comm comm) {
    MPI_Datatype dtype = Internal::get_mpi_type(*buffer);
    MPI_Bcast(buffer, count, dtype, root, comm);
  }

  // ********************************
  // NCCL implementation
  // ********************************

  template <typename T>
  void allreduce(T *sendbuf, T *recvbuf, size_t count, ReductionOp<T> op,
                 ncclComm_t comm, cudaStream_t stream) {
    ncclDataType_t dtype = Internal::get_nccl_type(*sendbuf);
    ncclRedOp_t nccl_op = Internal::get_nccl_op(op);
    ncclAllReduce(sendbuf, recvbuf, count, dtype, nccl_op, comm, stream);
  }

  template <typename T>
  void reduce(T *sendbuf, T *recvbuf, size_t count, ReductionOp<T> op, int root,
              ncclComm_t comm, cudaStream_t stream) {
    ncclDataType_t dtype = Internal::get_nccl_type(*sendbuf);
    ncclRedOp_t nccl_op = Internal::get_nccl_op(op);
    ncclReduce(sendbuf, recvbuf, count, dtype, nccl_op, root, comm, stream);
  }

  template <typename T>
  void allgather(T *sendbuf, T *recvbuf, size_t count, ncclComm_t comm,
                 cudaStream_t stream) {
    ncclDataType_t dtype = Internal::get_nccl_type(*sendbuf);
    ncclAllGather(sendbuf, recvbuf, count, dtype, comm, stream);
  }

  template <typename T>
  void alltoall(T *sendbuf, T *recvbuf, size_t count, ncclComm_t comm,
                cudaStream_t stream) {

    const int &rank = NCCLOps::get_rank();
    const int &nranks = NCCLOps::get_size();
    size_t chunk_size = count / nranks;
    ncclDataType_t dtype = Internal::get_nccl_type(*sendbuf);
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
    ncclDataType_t dtype = Internal::get_nccl_type(*buffer);
    ncclBroadcast(buffer, count, dtype, root, comm, stream);
  }

private:
};

} // namespace CCO
#endif
