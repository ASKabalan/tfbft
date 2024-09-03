#ifndef COMM_INDEX_H
#define COMM_INDEX_H

#include <vector>
#include <unordered_map>
#include <nccl.h>
#include <cstddef>      // For size_t
#include "nccl_ops.h"   // Include the NCCLOps for accessing the main communicator

class comm_index {
public:
    // Singleton instance getter
    static comm_index& instance();

    // Function to get or create communicators for FFT
    std::vector<ncclComm_t> get_or_create_comms(size_t array_size, int num_devices);

private:
    // Private constructor for singleton pattern
    comm_index() = default;

    // Disallow copying and assignment
    comm_index(const comm_index&) = delete;
    comm_index& operator=(const comm_index&) = delete;

    // Function to generate the required communicators
    void generate_comms(size_t array_size, int num_devices, ncclComm_t base_comm);

    // Function to calculate the color based on device index and stage
    int calculate_color(int device_index, int stage, int num_devices) const;

    std::unordered_map<size_t, std::vector<ncclComm_t>> comms_db;  // Unordered map to store communicators for each array size
};

#endif // COMM_INDEX_H

