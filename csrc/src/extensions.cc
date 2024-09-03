#include "gpu_ops.cuh"
#include "mpi_ops.hpp"
#include "nanobind/nanobind.h"
#include "nccl_ops.hpp"
#include "perfostep.hpp"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <iomanip>
#include <mpi.h>
#include <nccl.h>
#include <sstream>
#include <string>
#include <type_traits>

namespace ffi = xla::ffi;
namespace nb = nanobind;

enum class Backend { NCCL, MPI };

enum class Mode { InPlace, OutOfPlace };

enum class Collective { AllReduce, AllGather, AllToAll, Peer2Peer };

std::string collective_str(const Collective &c) {
  switch (c) {
  case Collective::AllReduce:
    return "AllReduce";
  case Collective::AllGather:
    return "AllGather";
  case Collective::AllToAll:
    return "AllToAll";
  case Collective::Peer2Peer:
    return "Peer2Peer";
  default:
    return "Unknown";
  }
}

std::string backend_str(const Backend &b) {
  switch (b) {
  case Backend::NCCL:
    return "NCCL";
  case Backend::MPI:
    return "MPI";
  default:
    return "Unknown";
  }
}

std::string mode_str(const Mode &m) {
  switch (m) {
  case Mode::InPlace:
    return "InPlace";
  case Mode::OutOfPlace:
    return "OutOfPlace";
  default:
    return "Unknown";
  }
}

std::string doubleToString(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value;
  return out.str();
}

ffi::Error CollectiveImpl(cudaStream_t stream, int64_t backend,
                          int64_t collective, int64_t mode,
                          ffi::Buffer<ffi::DataType::F32> x,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {

  static const char *env = std::getenv("ENABLE_PERFO_STEP");
  static const char *out_file = std::getenv("PERFO_STEP_OUT_FILE");
  int iterations = 1;
  bool enable_perf = env != nullptr && out_file != nullptr;
  if (enable_perf) {
    iterations = 10;
    std::cout << "Will be profiling" << std::endl;
  }

  Backend e_backend = static_cast<Backend>(backend);
  Collective e_collective = static_cast<Collective>(collective);
  Mode e_mode = static_cast<Mode>(mode);
  // MPI inits
  MPIOps mpi_ops;
  const int &rank = mpi_ops.get_rank();
  const int &size = mpi_ops.get_size();
  //  NCCL inits
  NCCLOps nccl_ops;
  ncclComm_t comm = nccl_ops.get_comm();
  size_t chunk_size = x.element_count() / size;
  // Pair ranks permute with Odd ranks
  // So if rank is even, next_rank is rank + 1
  // If rank is odd, next_rank is rank - 1
  int next_rank = rank % 2 == 0 ? (rank + 1) % size : (rank - 1 + size) % size;
  // int prev_rank = rank % 2 == 0 ? (rank - 1 + size) % size : (rank + 1) %
  // size;

  std::string backend_name(backend_str(e_backend).c_str());
  Perfostep perf;
  for (int i = 0; i < iterations; i++) {
    perf.Start(backend_name);
    if (e_backend == Backend::MPI) {
      switch (e_collective) {
      case Collective::AllReduce:
        if (e_mode == Mode::InPlace) {
          MPI_Allreduce(MPI_IN_PLACE, y->untyped_data(), y->element_count(),
                        MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        } else {
          MPI_Allreduce(x.untyped_data(), y->untyped_data(), y->element_count(),
                        MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
        break;
      case Collective::AllGather:
        MPI_Allgather(x.untyped_data(), x.element_count(), MPI_FLOAT,
                      y->untyped_data(), x.element_count(), MPI_FLOAT,
                      MPI_COMM_WORLD);
        break;
      case Collective::AllToAll:
        if (e_mode == Mode::InPlace) {
          MPI_Alltoall(x.untyped_data(), x.element_count(), MPI_FLOAT,
                       y->untyped_data(), x.element_count(), MPI_FLOAT,
                       MPI_COMM_WORLD);
          break;
        } else {
          MPI_Alltoall(MPI_IN_PLACE, 0, MPI_FLOAT, x.untyped_data(),
                       x.element_count(), MPI_FLOAT, MPI_COMM_WORLD);
        }
        break;
      case Collective::Peer2Peer:
        MPI_Sendrecv(x.untyped_data(), x.element_count(), MPI_FLOAT, next_rank,
                     0, y->untyped_data(), y->element_count(), MPI_FLOAT, rank,
                     0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
    } else {
      // NCCL
      switch (e_collective) {
      case Collective::AllReduce:
        ncclAllReduce(x.untyped_data(), y->untyped_data(), x.element_count(),
                      ncclFloat32, ncclSum, comm, stream);
        break;
      case Collective::AllGather:
        ncclAllGather(x.untyped_data(), y->untyped_data(), x.element_count(),
                      ncclFloat32, comm, stream);
        break;
      case Collective::AllToAll:
        ncclGroupStart();
        for (int r = 0; r < size; r++) {
          ncclSend((void *)(x.typed_data() + r * chunk_size), chunk_size,
                   ncclFloat32, r, comm, stream);
          ncclRecv((void *)(y->typed_data() + r * chunk_size), chunk_size,
                   ncclFloat32, r, comm, stream);
        }
        ncclGroupEnd();
        break;
      case Collective::Peer2Peer:
        ncclGroupStart();
        ncclSend(x.untyped_data(), x.element_count(), ncclFloat32, next_rank,
                 comm, stream);
        ncclRecv(y->untyped_data(), y->element_count(), ncclFloat32, next_rank,
                 comm, stream);
        ncclGroupEnd();
        break;
      }
    }
    CUDACHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_ms = perf.Stop();
    if (enable_perf) {
      MPI_Allreduce(MPI_IN_PLACE, &elapsed_ms, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      if (rank == 0) {
        elapsed_ms /= size;
        double elapsed_s = elapsed_ms / 1E3;
        double alg_bandwidth =
            (double)(x.size_bytes() + y->size_bytes()) / 1E9 / elapsed_s;
        double bus_bandwidth;
        std::cout << "Elapsed: " << elapsed_ms << " ms" << std::endl;
        std::cout << "size in bytes: " << x.size_bytes() + y->size_bytes()
                  << std::endl;
        std::cout << "elapsed in s: " << elapsed_s << std::endl;
        switch (e_collective) {
        case Collective::AllReduce:
          bus_bandwidth = alg_bandwidth * (2.0 * (size - 1) / size);
          break;
        case Collective::AllToAll:
        case Collective::AllGather:
          bus_bandwidth = alg_bandwidth * ((double)(size - 1) / size);
          break;
        case Collective::Peer2Peer:
          bus_bandwidth = alg_bandwidth;
          break;
        default:
          bus_bandwidth = alg_bandwidth; // Fallback
        }
        std::string collective_name(collective_str(e_collective).c_str());
        std::string mode_name(mode_str(e_mode).c_str());
        std::string data_size =
            doubleToString((x.size_bytes() + y->size_bytes()) / 1E9) + " GB";

        std::string elapsed_str = doubleToString(elapsed_ms) + " ms";
        std::string alg_bandwidth_str = doubleToString(alg_bandwidth) + " GB/s";
        std::string bus_bandwidth_str = doubleToString(bus_bandwidth) + " GB/s";

        const ColumnNames col = {{"Collective", collective_name},
                                 {"Iteration", std::to_string(i)},
                                 {"Mode", mode_name},
                                 {"Size", data_size},
                                 {"GPUs", std::to_string(size)},
                                 {"Elapsed", elapsed_str},
                                 {"AlgBW", alg_bandwidth_str},
                                 {"BusBW", bus_bandwidth_str}};
        perf.SetColumnNames(col);
        perf.PrintToMarkdown(out_file);
      }
    }
  }
  return ffi::Error::Success();
}

ffi::Error AddElementImpl(cudaStream_t stream, float scaler,
                          ffi::Buffer<ffi::DataType::F32> x,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {

  add_element(scaler, x.typed_data(), y->typed_data(), x.element_count(),
              stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CollectiveCall, CollectiveImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("backend")
                                  .Attr<int64_t>("collective")
                                  .Attr<int64_t>("mode")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>() // x
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(AddElement, AddElementImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<float>("scaler") // scaler
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

nb::dict Registrations() {
  nb::dict d;
  d["collective_call"] = EncapsulateFfiCall(CollectiveCall);
  d["add_element"] = EncapsulateFfiCall(AddElement);
  return d;
}

NB_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  nb::enum_<Backend>(m, "Backend")
      .value("NCCL", Backend::NCCL)
      .value("MPI", Backend::MPI)
      .export_values();

  nb::enum_<Mode>(m, "Mode")
      .value("InPlace", Mode::InPlace)
      .value("OutOfPlace", Mode::OutOfPlace)
      .export_values();

  nb::enum_<Collective>(m, "Collective")
      .value("AllReduce", Collective::AllReduce)
      .value("AllGather", Collective::AllGather)
      .value("AllToAll", Collective::AllToAll)
      .value("Peer2Peer", Collective::Peer2Peer)
      .export_values();
}
