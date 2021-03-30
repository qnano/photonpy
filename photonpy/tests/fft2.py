# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from photonpy.cpp.lib import SMLM


def crosscorrelation(A, B, smlm: SMLM):
    A_fft = np.fft.fft2(A)
    B_fft = np.fft.fft2(B)
    return np.fft.ifft2(A_fft * np.conj(B_fft))


def crosscorrelation_cuda(A, B, smlm: SMLM):
    return smlm.IFFT2(smlm.FFT2(A) * np.conj(smlm.FFT2(B)))

def debugImage(img,label):
    print(f"debug image: {label}")
    plt.figure()
    plt.imshow(img[0])
    plt.title(label)

F = np.fft

L = 32
freq = [0.05, 0.03]
x = np.arange(L)
X, Y = np.meshgrid(x, x)
img = np.sin(2 * np.pi * (freq[0] * X + freq[1] * Y))

with SMLM(debugMode=False) as smlm:

    smlm.SetDebugImageCallback(debugImage)

    fft_sig_cuda = F.fftshift(smlm.FFT2(img))
    fft_sig = F.fftshift(F.fft2(img))

    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(np.hstack((np.abs(fft_sig), np.abs(fft_sig_cuda))))

    img0 = np.sin(2 * np.pi * (freq[0] * X + freq[1] * Y))
    img1 = np.sin(2 * np.pi * (freq[0] * (X - 5) + freq[1] * (Y + 10)))

    cc0 = crosscorrelation(img0, img1, smlm)
    cc1 = crosscorrelation_cuda(img0, img1, smlm)
    cc0_max = np.unravel_index(np.argmax(cc0), cc0.shape)
    cc1_max = np.unravel_index(np.argmax(cc1), cc1.shape)
    plt.figure()
    plt.plot(cc0[cc0_max[0], :] + 1000)
    plt.plot(cc1[cc0_max[0], :])
    plt.legend()
    print(cc0_max)
    print(cc1_max)

    sig = fft_sig_cuda

    inv = F.ifft2(sig)
    inv_cuda = smlm.IFFT2(sig)

    plt.figure()
    plt.plot(x, inv[L // 2 + 2], label="np.fft")
    plt.plot(x, inv_cuda[L // 2 + 2], label="CUDA FFT")
    plt.legend()

    assert np.sum(np.abs(inv - inv_cuda) ** 2) < 1e-6
