import numpy as np

# Global logging and FFT mode configuration
LOGGING_LEVEL = 'operations'  # Options: 'operations', 'other', 'both', 'none'
FFT_MODE = 'dit'  # Options: 'dit', 'dif', 'both'


def log(message, level='other'):
    if LOGGING_LEVEL == 'both' or LOGGING_LEVEL == level:
        print(message)


def bit_reversal(x):
    N = len(x)
    num_bits = int(np.log2(N))

    # Create an array to hold the bit-reversed output
    reversed_x = np.zeros(N, dtype=x.dtype)

    for i in range(N):
        # Reverse the bits of index i
        reversed_index = int('{:0{width}b}'.format(i, width=num_bits)[::-1], 2)
        # Place the element at the bit-reversed index
        reversed_x[reversed_index] = x[i]

    return reversed_x


def decimation_in_time(x):
    N = len(x)
    num_stages = int(np.log2(N))

    # Iterate over the stages
    for stage in range(num_stages):
        step = 2**(stage + 1)  # FFT size at this stage
        half_step = step // 2  # Half of the FFT size

        log(f"\n=== Stage {stage + 1} ===", 'operations')
        log(f"Step: {step}, Half Step: {half_step}", 'operations')

        # Indices for the current stage's butterfly operations
        k = np.arange(0, N, step).reshape(-1,
                                          1)  # Start indices for each group
        j = np.arange(half_step)  # Offsets within each group

        log(f"k (start indices for groups):\n{k.flatten()}", 'other')
        log(f"j (offsets within groups): {j}", 'other')

        # Compute twiddle factors for this stage
        twiddle = np.exp(-2j * np.pi * j / step)
        log(f"Twiddle factors: {twiddle}", 'other')

        # Perform the butterfly calculations in a vectorized manner
        u = x[k + j]
        t = twiddle * x[k + j + half_step]

        # Logging the operations
        for group_index in range(len(k)):
            for offset in range(half_step):
                u_index = k[group_index, 0] + j[offset]
                t_index = u_index + half_step
                log(
                    f"[OPERATION] x[{u_index}] = x[{u_index}] + W{offset}/{step} * x[{t_index}]",
                    'operations')
                log(
                    f"[OPERATION] x[{t_index}] = x[{u_index}] - W{offset}/{step} * x[{t_index}]",
                    'operations')

        # Update the array in place
        x[k + j] = u + t
        x[k + j + half_step] = u - t

        log(f"x after stage {stage + 1}:\n{x}", 'other')

    return x


def decimation_in_frequency(x):
    N = len(x)
    num_stages = int(np.log2(N))

    # Iterate over the stages in reverse order compared to DIT
    for stage in range(num_stages):
        step = 2**(num_stages - stage)  # FFT size at this stage
        half_step = step // 2  # Half of the FFT size

        log(f"\n=== Stage {stage + 1} ===", 'operations')
        log(f"Step: {step}, Half Step: {half_step}", 'operations')

        # Indices for the current stage's butterfly operations
        k = np.arange(0, N, step).reshape(-1,
                                          1)  # Start indices for each group
        j = np.arange(half_step)  # Offsets within each group

        log(f"k (start indices for groups):\n{k.flatten()}", 'other')
        log(f"j (offsets within groups): {j}", 'other')

        # Compute twiddle factors for this stage
        twiddle = np.exp(-2j * np.pi * j / step)
        log(f"Twiddle factors: {twiddle}", 'other')

        # Perform the butterfly calculations in a vectorized manner
        u = x[k + j]
        t = x[k + j + half_step]

        # Logging the operations
        for group_index in range(len(k)):
            for offset in range(half_step):
                u_index = k[group_index, 0] + j[offset]
                t_index = u_index + half_step
                log(f"[OPERATION] x[{u_index}] = x[{u_index}] + x[{t_index}]",
                    'operations')
                log(
                    f"[OPERATION] x[{t_index}] = (x[{u_index}] - x[{t_index}]) * W{offset}/{step}",
                    'operations')

        # Update the array in place
        x[k + j] = u + t
        x[k + j + half_step] = (u - t) * twiddle

        log(f"x after stage {stage + 1}:\n{x}", 'other')

    return x


def fft_dif(x):
    log("=== Decimation in Frequency ===", 'other')
    x = decimation_in_frequency(x)
    # Apply bit reversal at the end for DIF
    x = bit_reversal(x)
    log(f"\nx after bit reversal:\n{x}", 'other')
    return x


def fft_dit(x):
    log("=== Bit Reversal ===", 'other')
    x = bit_reversal(x)
    log(f"x after bit reversal:\n{x}", 'other')

    log("=== Decimation in Time ===", 'other')
    return decimation_in_time(x)


# Example usage:
res = {}
powers_of_2 = [2**i for i in range(4, 5)]
dit_f_are_close = "NOT_DONE"
dif_f_are_close = "NOT_DONE"

for i in powers_of_2:
    x = np.arange(i) * 1j  # Input array (using simple integers for clarity)
    log(f"Input Array:\n{x}", 'other')

    # Run FFTs based on FFT_MODE
    if FFT_MODE in ['dit', 'both']:
        dit_f = fft_dit(x.copy())
        log(f"Output dit_f:\n{dit_f}", 'other')
        # Compare with numpy's FFT
        fft_numpy = np.fft.fft(x)
        dit_f_are_close = np.allclose(dit_f, fft_numpy)
        print("\nIs dit_f good?", dit_f_are_close)

    if FFT_MODE in ['dif', 'both']:
        dif_f = fft_dif(x.copy())
        log(f"Output dif_f:\n{dif_f}", 'other')
        # Compare with numpy's FFT
        fft_numpy = np.fft.fft(x)
        dif_f_are_close = np.allclose(dif_f, fft_numpy)
        print("Is dif_f good?", dif_f_are_close)

    res[i] = {"dit": dit_f_are_close, "dif": dif_f_are_close}

for key, value in res.items():
    print(f"For N = {key}: dit is {value['dit']} and dif is {value['dif']}")
