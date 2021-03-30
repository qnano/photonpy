import matplotlib.pyplot as plt
import numpy as np

from photonpy.cpp.lib import SMLM

F = np.fft

freq = 0.02
x = np.arange(200)
sig = np.sin(2 * np.pi * freq * x)

with SMLM(debugMode=False) as smlm:
    fft_sig_cuda = smlm.FFT(sig)

    plt.figure()
    plt.plot(sig)

    plt.figure()
    freqrange = F.fftshift(F.fftfreq(len(x)))
    plt.plot(freqrange, np.abs(F.fftshift(F.fft(sig))), label="np.fft")
    plt.plot(freqrange, np.abs(F.fftshift(fft_sig_cuda)), label="CUDA FFT")
    plt.legend()

    sig = fft_sig_cuda

    plt.figure()
    plt.plot(x, np.abs(F.ifft(sig)), label="np.fft")
    plt.plot(x, np.abs(smlm.IFFT(sig)) + 1, label="CUDA FFT")
    plt.legend()
