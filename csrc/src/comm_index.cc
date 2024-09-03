#include "comm_index.h"
#include "common/checks.h"  // Assuming you have NCCLCHECK defined here
#include <cmath>        // For std::log2, std::pow
#include <stdexcept>    // For exceptions
#include <string>       // For std::to_string

// Singleton instance getter
comm_index& comm_index::instance() {
    static comm_index instance;
    return instance;
}

// Function to get or create communicators for FFT
std::vector<ncclComm_t> comm_index::get_or_create_comms(size_t array_size, int num_devices) {
    if (comms_db.find(array_size) == comms_db.end()) {
        ncclComm_t base_comm = NCCLOps::instance().get_comm();
        generate_comms(array_size, num_devices, base_comm);
    }
    return comms_db[array_size];
}

// Function to generate the required communicators
void comm_index::generate_comms(size_t array_size, int num_devices, ncclComm_t base_comm) {
    int num_stages = std::log2(num_devices);
    std::vector<ncclComm_t> comms_list;

    // Iterate through the number of stages to create communicators
    for (int stage = 0; stage < num_stages; ++stage) {
        for (int device_index = 0; device_index < num_devices; ++device_index) {
            int color = calculate_color(device_index, stage, num_devices);
            ncclComm_t new_comm;
            ncclResult_t result = ncclCommSplit(base_comm, color, device_index, &new_comm, nullptr);

            if (result != ncclSuccess) {
                throw std::runtime_error("Failed to create NCCL communicator for stage " + std::to_string(stage));
            }

            comms_list.push_back(new_comm);
        }
    }

    // Store the generated communicators in the map
    comms_db[array_size] = comms_list;
}

// Function to calculate the color based on device index and stage
int comm_index::calculate_color(int device_index, int stage, int num_devices) const {
    int factor = num_devices / std::pow(2, stage + 1);
    int mod_factor = device_index % factor;
    int add_factor = ((device_index / factor) / 2) * factor;
    return mod_factor + add_factor;
}

