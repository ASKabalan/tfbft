import numpy as np

def bit_reversal(index, b):
    """Returns the bit-reversed index of 'index' with 'b' bits."""
    return sum(((index >> j) & 1) << (b - 1 - j) for j in range(b))

def bit_reversal_array(array_size):
    """Returns a list of bit-reversed indices for an array of given size."""
    b = int(np.log2(array_size))  # Number of bits needed
    reversed_indices = [bit_reversal(i, b) for i in range(array_size)]
    return reversed_indices

def generate_movement_tuples(array_size, num_devices):
    b = int(np.log2(array_size))  # Number of bits needed
    device_size = array_size // num_devices  # Number of elements per device
    movement_tuples = []
    data_movements = []

    reversed_indices = bit_reversal_array(array_size)

    for i in range(array_size):
        # Determine source device
        source_device = i // device_size
        
        # Get the precomputed bit-reversed index
        reversed_index = reversed_indices[i]
        
        # Determine destination device
        destination_device = reversed_index // device_size
        
        # Calculate offset within the source and destination devices
        offset_within_source = i % device_size
        offset_within_dest = reversed_index % device_size
        
        # Create the tuple (index, source_device, destination_device, offset_within_dest)
        movement_tuples.append((i, source_device, destination_device, offset_within_dest))
        
        # Create the actual data movement as a tuple (source_index, source_device, destination_index, destination_device)
        data_movements.append((i, source_device, reversed_index, destination_device))

        # Print the movement details
        print(f"Offset: {offset_within_source} | Source Device: {source_device} | Target Device: {destination_device} | Offset in Target Device: {offset_within_dest} | Target Index: {reversed_index}")

    return movement_tuples, data_movements

# Example usage
array_size = 16  # Total size of the array
num_devices = 4  # Number of devices

# Generate and print movement tuples and data movements
movement_tuples, data_movements = generate_movement_tuples(array_size, num_devices)

