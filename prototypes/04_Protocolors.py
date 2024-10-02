import numpy as np
def find_first_and_second_occurrences(lst):
    # Dictionary to keep track of counts and positions
    positions = {}
    first_positions = []  # List to store positions of first occurrences
    second_positions = []  # List to store positions of second occurrences
    
    for i, value in enumerate(lst):
        # Increment count for each value
        if value in positions:
            positions[value]['count'] += 1
        else:
            positions[value] = {'count': 1}
            first_positions.append(i)  # Record first occurrence position
        
        # When the second occurrence is found, record its position
        if positions[value]['count'] == 2:
            second_positions.append(i)
    

    return first_positions, second_positions

def simulate_butterfly_fft_colors(num_devices):
    stages = int(np.log2(num_devices))
    devices = np.arange(num_devices)
    butterfly_colors = []

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
        butterfly_colors.append(colors + add_factor)
        # Get second occurence of number in the list
        a_ranks , b_ranks = find_first_and_second_occurrences(colors + add_factor)
        print(f"a_ranks: {a_ranks}")
        print(f"b_ranks: {b_ranks}")
        stage_size = num_devices // 2**(stage + 1)
        print(f"stage_size: {stage_size}")
        a_position = [rank // stage_size for rank in a_ranks]
        b_position = [rank // stage_size for rank in b_ranks]
        print(f"a_position: {a_position}")
        print(f"b_position: {b_position}")
        a_rank_norm = [rank - stage_size * ((rank // stage_size) // 2) for rank , pos in zip(a_ranks , a_position)]
        b_rank_norm = [rank - stage_size * (1 + (rank // stage_size)   // 2 ) for rank , pos in zip(b_ranks , b_position)]
        print(f"a_rank_norm: {a_rank_norm}")
        print(f"b_rank_norm: {b_rank_norm}")
        print("")

        

    return butterfly_colors

# Example usage
num_devices = 16  # Number of devices
butterfly_colors = simulate_butterfly_fft_colors(num_devices)


