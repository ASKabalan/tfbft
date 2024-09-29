import numpy as np

def simulate_butterfly_fft_colors(num_devices):
    stages = int(np.log2(num_devices))
    devices = np.arange(num_devices)

    for stage in range(stages):
        # Calculate the current group size based on the stages
        factor = (num_devices // 2**(stage + 1))
        add_factor = ((devices // factor ) // 2) * factor
        # Assign colors based on the current group size
        colors = (devices) % factor   
        print(f"factor: {factor}")
        print(f"Stage {stage + 1}:")
        print("Devices:", devices)
        print("Modulo factor:", colors)
        print(f"add_factor: { add_factor}")
        print(f"Colors: {colors + add_factor}")
        print("")

# Example usage
num_devices = 16  # Number of devices
simulate_butterfly_fft_colors(num_devices)

