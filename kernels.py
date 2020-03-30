import numpy as np


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=5):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=3.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))
