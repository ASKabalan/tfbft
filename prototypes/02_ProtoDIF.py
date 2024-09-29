import numpy as np


def bit_reverse(x, n):
    # x: input array
    # n: number of bits
    # return: array with reversed bits
    original_indices = np.arange(len(x))
    reversed_indices = np.array(
        [int(format(i, f'0{n}b')[::-1], 2) for i in original_indices])
    return x[reversed_indices]


a = np.arange(16)


# Compute 16-point FFT using off the shelf FFT
def fft(a):
    return np.fft.fft(a)


def decimation_in_frequency(a):

    a_first_half = a[:8]
    a_second_half = a[8:]

    a_first_half_butterfly = a_first_half + a_second_half
    a_second_half_butterfly = (a_first_half - a_second_half) * np.exp(
        -2j * np.pi / 16 * np.arange(8))

    a = np.concatenate([a_first_half_butterfly, a_second_half_butterfly])

    a_first_quarter = a[:4]
    a_second_quarter = a[4:8]
    a_third_quarter = a[8:12]
    a_fourth_quarter = a[12:]

    a_first_quarter_butterfly = a_first_quarter + a_second_quarter
    a_second_quarter_butterfly = (a_first_quarter - a_second_quarter) * np.exp(
        -2j * np.pi / 8 * np.arange(4))
    a_third_quarter_butterfly = a_third_quarter + a_fourth_quarter
    a_fourth_quarter_butterfly = (a_third_quarter - a_fourth_quarter) * np.exp(
        -2j * np.pi / 8 * np.arange(4))

    a = np.concatenate([
        a_first_quarter_butterfly, a_second_quarter_butterfly,
        a_third_quarter_butterfly, a_fourth_quarter_butterfly
    ])

    a_first_eighth = a[:2]
    a_second_eighth = a[2:4]
    a_third_eighth = a[4:6]
    a_fourth_eighth = a[6:8]
    a_fifth_eighth = a[8:10]
    a_sixth_eighth = a[10:12]
    a_seventh_eighth = a[12:14]
    a_eighth_eighth = a[14:]

    a_first_eighth_butterfly = a_first_eighth + a_second_eighth
    a_second_eighth_butterfly = (a_first_eighth - a_second_eighth) * np.exp(
        -2j * np.pi / 4 * np.arange(2))
    a_third_eighth_butterfly = a_third_eighth + a_fourth_eighth
    a_fourth_eighth_butterfly = (a_third_eighth - a_fourth_eighth) * np.exp(
        -2j * np.pi / 4 * np.arange(2))
    a_fifth_eighth_butterfly = a_fifth_eighth + a_sixth_eighth
    a_sixth_eighth_butterfly = (a_fifth_eighth - a_sixth_eighth) * np.exp(
        -2j * np.pi / 4 * np.arange(2))
    a_seventh_eighth_butterfly = a_seventh_eighth + a_eighth_eighth
    a_eighth_eighth_butterfly = (a_seventh_eighth - a_eighth_eighth) * np.exp(
        -2j * np.pi / 4 * np.arange(2))

    a = np.concatenate([
        a_first_eighth_butterfly, a_second_eighth_butterfly,
        a_third_eighth_butterfly, a_fourth_eighth_butterfly,
        a_fifth_eighth_butterfly, a_sixth_eighth_butterfly,
        a_seventh_eighth_butterfly, a_eighth_eighth_butterfly
    ])

    res = np.zeros(16, dtype=complex)

    res[0] = a[0] + a[1]
    res[1] = a[0] - a[1]
    res[2] = a[2] + a[3]
    res[3] = a[2] - a[3]
    res[4] = a[4] + a[5]
    res[5] = a[4] - a[5]
    res[6] = a[6] + a[7]
    res[7] = a[6] - a[7]
    res[8] = a[8] + a[9]
    res[9] = a[8] - a[9]
    res[10] = a[10] + a[11]
    res[11] = a[10] - a[11]
    res[12] = a[12] + a[13]
    res[13] = a[12] - a[13]
    res[14] = a[14] + a[15]
    res[15] = a[14] - a[15]

    return res


def hybrid_dif_with_fft_atom(a):

    a_first_half = a[:8]
    a_second_half = a[8:]

    a_first_half_butterfly = a_first_half + a_second_half
    a_second_half_butterfly = (a_first_half - a_second_half) * np.exp(
        -2j * np.pi / 16 * np.arange(8))

    a = np.concatenate([a_first_half_butterfly, a_second_half_butterfly])

    a_first_quarter = a[:4]
    a_second_quarter = a[4:8]
    a_third_quarter = a[8:12]
    a_fourth_quarter = a[12:]

    a_first_quarter_butterfly = a_first_quarter + a_second_quarter
    a_second_quarter_butterfly = (a_first_quarter - a_second_quarter) * np.exp(
        -2j * np.pi / 8 * np.arange(4))
    a_third_quarter_butterfly = a_third_quarter + a_fourth_quarter
    a_fourth_quarter_butterfly = (a_third_quarter - a_fourth_quarter) * np.exp(
        -2j * np.pi / 8 * np.arange(4))

    a = np.concatenate([
        a_first_quarter_butterfly, a_second_quarter_butterfly,
        a_third_quarter_butterfly, a_fourth_quarter_butterfly
    ])

    # testing using FFT as atom for 8-point fft
    first_quarter_fft = bit_reverse(np.fft.fft(a[:4]), 2)
    second_quarter_fft = bit_reverse(np.fft.fft(a[4:8]), 2)
    third_quarter_fft = bit_reverse(np.fft.fft(a[8:12]), 2)
    fourth_quarter_fft = bit_reverse(np.fft.fft(a[12:]), 2)

    a_eight_fft = np.concatenate([
        first_quarter_fft, second_quarter_fft, third_quarter_fft,
        fourth_quarter_fft
    ])

    return a_eight_fft


def decimation_in_time_ifft(a):
    res = np.zeros(16, dtype=complex)

    # Stage 1 .. 2-point FFTs

    res[0] = a[0] + a[1]
    res[1] = a[0] - a[1]
    res[2] = a[2] + a[3]
    res[3] = a[2] - a[3]
    res[4] = a[4] + a[5]
    res[5] = a[4] - a[5]
    res[6] = a[6] + a[7]
    res[7] = a[6] - a[7]
    res[8] = a[8] + a[9]
    res[9] = a[8] - a[9]
    res[10] = a[10] + a[11]
    res[11] = a[10] - a[11]
    res[12] = a[12] + a[13]
    res[13] = a[12] - a[13]
    res[14] = a[14] + a[15]
    res[15] = a[14] - a[15]

    a = res

    # Stage 2 .. 4-point FFTs
    a_first_eighth = a[:2]
    a_second_eighth = a[2:4]
    a_third_eighth = a[4:6]
    a_fourth_eighth = a[6:8]
    a_fifth_eighth = a[8:10]
    a_sixth_eighth = a[10:12]
    a_seventh_eighth = a[12:14]
    a_eighth_eighth = a[14:]

    a_first_eighth_butterfly = a_first_eighth + a_second_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))
    a_second_eighth_butterfly = a_first_eighth - a_second_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))
    a_third_eighth_butterfly = a_third_eighth + a_fourth_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))
    a_fourth_eighth_butterfly = a_third_eighth - a_fourth_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))
    a_fifth_eighth_butterfly = a_fifth_eighth + a_sixth_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))
    a_sixth_eighth_butterfly = a_fifth_eighth - a_sixth_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))
    a_seventh_eighth_butterfly = a_seventh_eighth + a_eighth_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))
    a_eighth_eighth_butterfly = a_seventh_eighth - a_eighth_eighth * np.exp(
        +2j * np.pi / 4 * np.arange(2))

    a = np.concatenate([
        a_first_eighth_butterfly, a_second_eighth_butterfly,
        a_third_eighth_butterfly, a_fourth_eighth_butterfly,
        a_fifth_eighth_butterfly, a_sixth_eighth_butterfly,
        a_seventh_eighth_butterfly, a_eighth_eighth_butterfly
    ])

    # Stage 3 .. 8-point FFTs
    #
    a_first_quarter = a[:4]
    a_second_quarter = a[4:8]
    a_third_quarter = a[8:12]
    a_fourth_quarter = a[12:]

    a_first_quarter_butterfly = a_first_quarter + a_second_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))
    a_second_quarter_butterfly = a_first_quarter - a_second_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))
    a_third_quarter_butterfly = a_third_quarter + a_fourth_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))
    a_fourth_quarter_butterfly = a_third_quarter - a_fourth_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))

    a = np.concatenate([a_first_quarter_butterfly, a_second_quarter_butterfly, a_third_quarter_butterfly, a_fourth_quarter_butterfly])

    # Stage 4 .. 16-point FFTs

    a_first_half = a[:8]
    a_second_half = a[8:]

    a_first_half_butterfly = a_first_half + a_second_half * np.exp(+2j * np.pi / 16 * np.arange(8))
    a_second_half_butterfly = a_first_half - a_second_half * np.exp(+2j * np.pi / 16 * np.arange(8))
    #
    res = np.concatenate([a_first_half_butterfly, a_second_half_butterfly])

    return res


def hybrid_dit_with_ifft_atom(a):

    # stage 3 and 4 together in a FFT

    first_quarter_fft = np.fft.ifft(bit_reverse(a[:4], n=2), norm='forward')  
    second_quarter_fft = np.fft.ifft(bit_reverse(a[4:8], n=2), norm='forward')
    third_quarter_fft = np.fft.ifft(bit_reverse(a[8:12], n=2), norm='forward')
    fourth_quarter_fft = np.fft.ifft(bit_reverse(a[12:], n=2), norm='forward')

    
    a = np.concatenate([first_quarter_fft, second_quarter_fft, third_quarter_fft, fourth_quarter_fft])

    # stage 2 8-point FFTs

    a_first_quarter = a[:4]
    a_second_quarter = a[4:8]
    a_third_quarter = a[8:12]
    a_fourth_quarter = a[12:]

    a_first_quarter_butterfly = a_first_quarter + a_second_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))
    a_second_quarter_butterfly = a_first_quarter - a_second_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))
    a_third_quarter_butterfly = a_third_quarter + a_fourth_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))
    a_fourth_quarter_butterfly = a_third_quarter - a_fourth_quarter * np.exp(+2j * np.pi / 8 * np.arange(4))

    a = np.concatenate([a_first_quarter_butterfly, a_second_quarter_butterfly, a_third_quarter_butterfly, a_fourth_quarter_butterfly])

    # stage 1 16-point FFTs

    a_first_half = a[:8]
    a_second_half = a[8:]

    a_first_half_butterfly = a_first_half + a_second_half * np.exp(+2j * np.pi / 16 * np.arange(8))
    a_second_half_butterfly = a_first_half - a_second_half * np.exp(+2j * np.pi / 16 * np.arange(8))

    a = np.concatenate([a_first_half_butterfly, a_second_half_butterfly])

    return a



np.set_printoptions(precision=2)
np.set_printoptions(linewidth=100)

fft_16 = np.fft.fft(a)
ifft_16 = np.fft.ifft(a)
dif = decimation_in_frequency(a)
dif_hybrid = hybrid_dif_with_fft_atom(a)
dit = decimation_in_time_ifft(bit_reverse(a, 4))
dit_hybrid = hybrid_dit_with_ifft_atom(bit_reverse(a, 4))

dif_dit = decimation_in_time_ifft(dif)
dif_dit_hybrid = hybrid_dit_with_ifft_atom(dif)


dit /= 16
dit_hybrid /= 16
dif_dit /= 16
dif_dit_hybrid /= 16

print(f"-"*20)
print(dit)
print(f"-"*20)
print(dit_hybrid)
print(f"-"*20)
print(ifft_16)
print(f"-"*20)
print(f"-"*20)

print(f"Check if fft_16 matches bit-reversed DIF: {np.allclose(fft_16, bit_reverse(dif, 4))}")
print(f"Check if fft_16 matches bit-reversed DIF hybrid: {np.allclose(fft_16, bit_reverse(dif_hybrid, 4))}")
print(f"Check if DIF matches DIF hybrid: {np.allclose(dif, dif_hybrid)}")

print(f"Check if ifft_16 matches DIT: {np.allclose(ifft_16, dit)}")
print(f"Check if ifft_16 matches DIT hybrid: {np.allclose(ifft_16, dit_hybrid)}")
print(f"Check if DIT matches DIT hybrid: {np.allclose(dit, dit_hybrid)}")

print(f"Check if DIF DIT matches 'a': {np.allclose(dif_dit, a)}")
print(f"Check if DIF DIT hybrid matches 'a': {np.allclose(dif_dit_hybrid, a)}")

