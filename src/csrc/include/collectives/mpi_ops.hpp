#ifndef MPI_OPS_H
#define MPI_OPS_H

#include <cuda_runtime.h>
#include <iostream>
#include <mpi-ext.h>
#include <mpi.h>
#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      std::cerr << "Failed: MPI error " << __FILE__ << ":" << __LINE__ << " '" \
                << e << "'" << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace CCO {

class MPIOpsImpl {
public:
  static MPIOpsImpl &instance() {
    static MPIOpsImpl instance;
    return instance;
  }

  void mpi_finalize() {
    if (isInitialized && !isMPIalreadyInitialized) {
      MPICHECK(MPI_Finalize());
    }
    isInitialized = false;
  }

  int get_rank() const { return rank; }

  int get_size() const { return size; }

  MPI_Comm get_comm() { return mpi_comm; }

  virtual ~MPIOpsImpl() {
    if (isInitialized) {
      mpi_finalize();
    }
  }

private:
  MPIOpsImpl()
      : rank(-1), size(-1), isInitialized(false),
        isMPIalreadyInitialized(false), mpi_comm(MPI_COMM_WORLD) {

    // Check if MPI has already been initialized
    MPICHECK(MPI_Initialized(&isMPIalreadyInitialized));
    if (!isMPIalreadyInitialized) {
      MPICHECK(MPI_Init(nullptr, nullptr));
    }

    // Initialize MPI rank and size
    MPICHECK(MPI_Comm_rank(mpi_comm, &rank));
    MPICHECK(MPI_Comm_size(mpi_comm, &size));

    if (1 == MPIX_Query_cuda_support()) {
      printf("This MPI library has CUDA-aware support.\n");
    } else {
      printf("This MPI library does not have CUDA-aware support.\n");
    }

    isInitialized = true;
  }

  // Disallow copying and assignment
  MPIOpsImpl(const MPIOpsImpl &) = delete;
  MPIOpsImpl &operator=(const MPIOpsImpl &) = delete;

  int rank;
  int size;
  MPI_Comm mpi_comm;
  bool isInitialized;
  int isMPIalreadyInitialized;
};

// Public interface for MPI operations
namespace MPIOps {

static MPI_Comm get_comm() { return MPIOpsImpl::instance().get_comm(); }

static const int get_rank() { return MPIOpsImpl::instance().get_rank(); }

static const int get_size() { return MPIOpsImpl::instance().get_size(); }

} // namespace MPIOps

} // namespace CCO
#endif // MPI_OPS_H
