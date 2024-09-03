#ifndef NCCL_OPS_H
#define NCCL_OPS_H

#include "common/checks.h" // Assuming you have similar checks for NCCL
#include "mpi_ops.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <nccl.h>
// Singleton implementation for managing NCCL operations
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
    MPIOps mpiOps;
    const int &rank = mpiOps.get_rank();
    const int &size = mpiOps.get_size();
    MPI_Comm mpicomm = mpiOps.get_comm();
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
class NCCLOps {
public:
  NCCLOps() = default;

  ~NCCLOps() = default;

  int get_rank() const { return NCCLOpsImpl::instance().get_rank(); }

  int get_size() const { return NCCLOpsImpl::instance().get_size(); }

  ncclComm_t get_comm() const { return NCCLOpsImpl::instance().get_comm(); }

private:
};

#endif // NCCL_OPS_H
