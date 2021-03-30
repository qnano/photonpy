# -*- coding: utf-8 -*-
import numpy as np


def lsqfit(Z, Y=None, X=None):
    if X is None:
        Y, X = np.indices(Z.shape)

    X = X.flatten()
    Y = Y.flatten()
    A = np.array([X ** 2, Y ** 2, X, Y, X * Y, X * 0 + 1]).T
    B = Z.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return coeff


def lsqeval(Y, X, coeff):
    Z = np.zeros(X.shape)
    for i, c in enumerate([X ** 2, Y ** 2, X, Y, X * Y, X * 0 + 1]):
        Z += coeff[i] * c
    return Z


def quadraticmax(c):
    d = 4 * c[0] * c[1] - c[4] ** 2
    xtop = -(2 * c[1] * c[2] - c[3] * c[4]) / d
    ytop = -(2 * c[0] * c[3] - c[2] * c[4]) / d
    return np.array([xtop, ytop])


def lsqfitmax(Z, Y=None, X=None):
    return quadraticmax(lsqfit(Z, Y, X))


if __name__ == "__main__":
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y, copy=False)
    Z = X ** 2 + (Y - 0.1) ** 2 + np.random.rand(*X.shape) * 0.01

    print(lsqfitmax(Z, Y, X))
