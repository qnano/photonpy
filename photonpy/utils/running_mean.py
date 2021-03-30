# https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
import numpy as np


def running_mean(y_in, x_in=None, N_out=None, sigma=1):
    """
    Returns running mean as a Bell-curve weighted average at evenly spaced
    points. Does NOT wrap signal around, or pad with zeros.

    Arguments:
    y_in -- y values, the values to be smoothed and re-sampled
    x_in -- x values for array

    Keyword arguments:
    N_out -- NoOf elements in resampled array.
    sigma -- 'Width' of Bell-curve in units of param x .
    """
    N_in = np.size(y_in)

    if x_in == None:
        x_in = np.arange(len(y_in))

    if N_out == None:
        N_out = N_in

    # Gaussian kernel
    x_out = np.linspace(np.min(x_in), np.max(x_in), N_out)
    x_in_mesh, x_out_mesh = np.meshgrid(x_in, x_out)
    gauss_kernel = np.exp(-np.square(x_in_mesh - x_out_mesh) / (2 * sigma ** 2))
    # Normalize kernel, such that the sum is one along axis 1
    normalization = np.tile(np.reshape(np.sum(gauss_kernel, axis=1), (N_out, 1)), (1, N_in))
    gauss_kernel_normalized = gauss_kernel / normalization
    # Perform running average as a linear operation
    y_out = gauss_kernel_normalized @ y_in

    return y_out, x_out


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    y, x = np.random.uniform(size=100), np.arange(100)
    y_avg, x_avg = running_mean(y, x, 100, sigma=6)

    plt.figure()
    plt.plot(x, y)
    plt.plot(x_avg, y_avg)
