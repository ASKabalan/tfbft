#ifndef NCCL_OPS_H
#define NCCL_OPS_H

#include <cuda_runtime.h>
#include <iostream>

#include "mpi_ops.hpp"
#include <mpi.h>
#include <nccl.h>

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      std::cerr << "Failed: NCCL error " << __FILE__ << ":" << __LINE__        \
                << " '" << ncclGetErrorString(r) << "'" << std::endl;          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace CCO {

class NCCLOpsImpl {
public:
  static NCCLOpsImpl &instance() {
    static NCCLOpsImpl instance;
    return instance;
  }

  int get_rank() const { return rank; }

  int get_size() const { return size; }

  ncclComm_t get_comm() const { return m_comm; }

  virtual ~NCCLOpsImpl() {
    if (isInitialized) {
      NCCLCHECK(ncclCommDestroy(m_comm));
    }
  }

private:
  NCCLOpsImpl() : rank(-1), size(-1), isInitialized(false) {
    // Initialize NCCL rank and size
    const int &rank = MPIOps::get_rank();
    const int &size = MPIOps::get_size();
    MPI_Comm mpicomm = MPIOps::get_comm();
    ncclUniqueId id;
    if (rank == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpicomm);
    ncclCommInitRank(&m_comm, size, id, rank);
    isInitialized = true;
  }

  // Disallow copying and assignment
  NCCLOpsImpl(const NCCLOpsImpl &) = delete;
  NCCLOpsImpl &operator=(const NCCLOpsImpl &) = delete;

  int rank;
  int size;
  bool isInitialized;
  ncclComm_t m_comm;
};

// Public interface for NCCL operations
namespace NCCLOps {

static const int get_rank() { return NCCLOpsImpl::instance().get_rank(); }

static const int get_size() { return NCCLOpsImpl::instance().get_size(); }

static ncclComm_t get_comm() { return NCCLOpsImpl::instance().get_comm(); }

} // namespace NCCLOps
} // namespace CCO

#endif // NCCL_OPS_H
