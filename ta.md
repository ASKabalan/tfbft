# FFTs

## my implementation

first implementing the algorithm in numpy

### DIT DIF with bit reversal

```python
def decimation_in_time(x):
    N = len(x)
    power_of_twos = 2**np.arange(int(np.log2(N)) - 1, 0, -1)
    print(power_of_twos)
    for stage, p in enumerate(power_of_twos):
        print(f"--" * 20)
        print(f"[DIT] p = {p}")
        print(f"--" * 20)
        splits = np.split(x, p)
        even_splits = splits[:p // 2]
        odd_splits = splits[p // 2:]

        print(f"  splits = {splits}")
        print(f"  even_splits = {even_splits}")
        print(f"  odd_splits = {odd_splits}")

        stage_output_first = []
        stage_output_second = []

        for i in range(p // 2):
            print(f"index = {i}")
            even = even_splits[i]
            odd = odd_splits[i]

            print(f"  even = {even}  , odd = {odd}")
            # KS are IOTA of stages with strides of p
            print(f"  stage = {stage + 1} p = {p} stage * p = {stage * p + 1}")
            ks = np.arange(0, (2**stage * p), p)
            print(f"  ks = {ks}")
            twiddle = np.exp(-2j * np.pi * ks / N)
            stage_output_first.extend((even + twiddle * odd))
            stage_output_second.extend((even - twiddle * odd))

        x = np.concatenate([stage_output_first, stage_output_second])
        print(f"  stage_output = {x}")

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

```

Test it with bit reversal and with a DIF FFT followed by a DIT IFFT (which should not require bit reversal)

### FFTFreq with bit reversal

undersnading the frequency output of the FFTFreq function
find a way to generate frequencies for a bit-reversed output


## Readings

**Original Cooley-Tukey**
https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/
Certainly! Here's the updated list with links, including **cuFFT** and its `cufftXt` and `cufftMp` extensions:

1. **HeFFTe (Highly Efficient FFT for Exascale)**
   - [HeFFTe GitHub Repository](https://github.com/icl-utk-edu/heffte)
   - [HeFFTe Documentation](https://icl.utk.edu/heffte/)

2. **2decomp&FFT**
   - [2decomp&FFT GitHub Repository](https://github.com/2decomp/2decomp_fft)
   - [2decomp&FFT Documentation](http://www.2decomp.org/)

3. **P3DFFT**
   - [P3DFFT Official Website](http://www.sdsc.edu/~pkuroda/p3dfft.html)
   - [P3DFFT GitHub Repository](https://github.com/sdsc/p3dfft)

4. **FFTW-MPI**
   - [FFTW Official Website](http://www.fftw.org/)
   - [FFTW Documentation (Distributed FFTs with MPI)](http://www.fftw.org/fftw3_doc/Distributed_002dmemory-FFTW-with-MPI.html)

5. **PFFT (Parallel FFT)**
   - [PFFT GitHub Repository](https://github.com/mpip/pfft)
   - [PFFT Documentation and Papers](http://www-user.tu-chemnitz.de/~potts/workgroup/pippig/software.php.en)

6. **AccFFT**
   - [AccFFT GitHub Repository](https://github.com/amirgholami/accfft)
   - [AccFFT Documentation](https://amirgholami.org/accfft.html)

7. **cuFFT (NVIDIA's CUDA Fast Fourier Transform library)**
   - **cufftXt** (for multi-GPU support):
     - [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/index.html#distributed-gpu)
   - **cufftMp** (for multi-node support):
     - [cuFFT Multi-node Library GitHub Repository](https://github.com/NVIDIA/cufftMp)
     - [cufftMp Documentation](https://docs.nvidia.com/cuda/cufftmp/index.html)

### Communications

 - Learn how to use cute
 - Use asymetric reductions using NCCL
 - Use ALL 2 ALL in place MPI
 - Use NCCL ALL2ALL and try to do it in place
