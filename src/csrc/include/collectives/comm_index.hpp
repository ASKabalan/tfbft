#ifndef COMM_INDEX_HPP
#define COMM_INDEX_HPP

#include "common/checks.h" // Assuming you have NCCLCHECK defined here
#include "nccl_ops.hpp" // Include the NCCLOps for accessing the main communicator
#include <cmath>        // For std::log2, std::pow
#include <cstddef>      // For size_t
#include <nccl.h>
#include <stdexcept> // For exceptions
#include <string>    // For std::to_string
#include <unordered_map>
#include <vector>

class comm_index {
public:
  // Singleton instance getter
  static comm_index &instance() {
    static comm_index instance;
    return instance;
  }

  // Function to get or create communicators for FFT
  std::vector<ncclComm_t> get_or_create_comms(size_t array_size) {
    if (comms_db.find(array_size) == comms_db.end()) {
      generate_comms(array_size);
    }
    return comms_db[array_size];
  }

private:
  // Private constructor for singleton pattern
  comm_index() = default;

  // Disallow copying and assignment
  comm_index(const comm_index &) = delete;
  comm_index &operator=(const comm_index &) = delete;

  // Function to generate the required communicators
  void generate_comms(size_t array_size) {

    NCCLOps nccl_ops;
    const int &size = nccl_ops.get_size();
    const int &rank = nccl_ops.get_rank();
    ncclComm_t base_comm = nccl_ops.get_comm();

    int num_stages = std::log2(size);
    std::vector<ncclComm_t> comms_list;

    // Iterate through the number of stages to create communicators
    for (int stage = 0; stage < num_stages; ++stage) {
      int color = calculate_color(rank, stage, size);
      ncclComm_t new_comm;
      ncclResult_t result =
          ncclCommSplit(base_comm, color, rank, &new_comm, nullptr);

      if (result != ncclSuccess) {
        throw std::runtime_error(
            "Failed to create NCCL communicator for stage " +
            std::to_string(stage));
      }

      comms_list.push_back(new_comm);
    }

    // Store the generated communicators in the map
    comms_db[array_size] = comms_list;
  }

  // Function to calculate the color based on device index and stage
  int calculate_color(int device_index, int stage, int num_devices) const {
    int factor = num_devices / std::pow(2, stage + 1);
    int mod_factor = device_index % factor;
    int add_factor = ((device_index / factor) / 2) * factor;
    return mod_factor + add_factor;
  }

  std::unordered_map<size_t, std::vector<ncclComm_t>>
      comms_db; // Unordered map to store communicators for each array size
};

#endif // COMM_INDEX_HPP
