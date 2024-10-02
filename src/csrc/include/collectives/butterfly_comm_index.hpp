#ifndef BUTTERFLY_COMM_INDEX_HPP
#define BUTTERFLY_COMM_INDEX_HPP

#include "nccl_ops.hpp" // Include the NCCLOps for accessing the main communicator
#include <cmath>        // For std::log2, std::pow
#include <cstddef>      // For size_t
#include <nccl.h>
#include <optional>  // For std::optional
#include <unordered_map>
#include <vector>


typedef std::pair<int, ncclComm_t> StageCommPair;
class CommIterator {
  std::vector<StageCommPair> m_comms;
  std::vector<StageCommPair>::iterator m_it;

public:
  // Constructor
  CommIterator(std::vector<StageCommPair> comms)
      : m_comms(comms), m_it(m_comms.begin()) {}

  std::optional<StageCommPair> next() {
    if (m_it != m_comms.end()) {
      return *(m_it++); // Return current element and move to next
    }
    return std::nullopt; // No more elements
  }

  std::optional<StageCommPair> prev() {
    if (m_it != m_comms.begin()) {
      return *(--m_it); // Move to previous element and return
    }
    return std::nullopt; // No more elements
  }

  void reset() { m_it = m_comms.begin(); }

  void reverse() { m_it = m_comms.end(); }
};

class ButterflyCommIndexImpl {
public:
  // Singleton instance getter
  static ButterflyCommIndexImpl &instance() {
    static ButterflyCommIndexImpl instance;
    return instance;
  }

  // Function to get or create communicators for FFT
  CommIterator get_or_create_comms(ncclComm_t base_comm) {
    int ranks_per_comm = 0;
    NCCLCHECK(ncclCommCount(base_comm, &ranks_per_comm));
    if (comms_db.find(ranks_per_comm) == comms_db.end()) {
      generate_comms(base_comm, ranks_per_comm);
    }
    return CommIterator(comms_db[ranks_per_comm]);
  }

private:
  // Private constructor for singleton pattern
  ButterflyCommIndexImpl() = default;

  // Disallow copying and assignment
  ButterflyCommIndexImpl(const ButterflyCommIndexImpl &) = delete;
  ButterflyCommIndexImpl &operator=(const ButterflyCommIndexImpl &) = delete;

  // Function to generate the required communicators
  void generate_comms(ncclComm_t base_comm, size_t size) {

    int rank = 0;
    NCCLCHECK(ncclCommUserRank(base_comm, &rank));
    int num_stages = std::log2(size);
    std::vector<StageCommPair> comms_list;

    // Iterate through the number of stages to create communicators
    for (int stage = 0; stage < num_stages; ++stage) {
      int color = calculate_color(rank, stage, size);
      int normalising_factor = std::pow(2, (stage + 1));
      ncclComm_t new_comm;
      NCCLCHECK(ncclCommSplit(base_comm, color, rank, &new_comm, nullptr));

      comms_list.push_back(std::make_pair(normalising_factor , new_comm));
    }

    // Store the generated communicators in the map
    comms_db[size] = comms_list;
  }

  // Function to calculate the color based on device index and stage
  int calculate_color(int device_index, int stage, int num_devices) const {
    int factor = num_devices / std::pow(2, stage + 1);
    int mod_factor = device_index % factor;
    int add_factor = ((device_index / factor) / 2) * factor;
    return mod_factor + add_factor;
  }

  std::unordered_map<size_t, std::vector<StageCommPair>>
      comms_db; // Unordered map to store communicators for each array size
};

// Public interface for ButterflyCommIndex

namespace ButterflyCommIndex {
static inline CommIterator get_or_create_comms(ncclComm_t base_comm) {
  return ButterflyCommIndexImpl::instance().get_or_create_comms(base_comm);
}

} // namespace ButterflyCommIndex

#endif // COMM_INDEX_HPP
