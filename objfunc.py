import numpy as np


def rastrigin(x, out=None):  # rast.m
    if len(x.shape) == 2:
        axis = 1
    else:
        axis = 0
    x = np.asarray_chkfinite(x)
    if out is None:
        y = np.zeros((x.shape[0]), dtype=x.dtype)
    else:
        y = out
    n = np.size(x, axis)
    y[:] = 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis)
    return y


def rosenbrock(x, out=None):  # rast.m
    if len(x.shape) == 2:
        axis = 1
    else:
        axis = 0
    x = np.asarray_chkfinite(x)
    if out is None:
        y = np.zeros((x.shape[0]), dtype=x.dtype)
    else:
        y = out
    n = np.size(x, axis)
    x0 = x[:, 0:n-1]
    x1 = x[:, 1:n]
    y[:] = np.sum((1 - x0) ** 2 + 100 * (x1 - x0 ** 2) ** 2, axis)
    return y


def schwefel(x, out=None):  # rast.m
    if len(x.shape) == 2:
        axis = 1
    else:
        axis = 0
    x = np.asarray_chkfinite(x)
    if out is None:
        y = np.zeros((x.shape[0]), dtype=x.dtype)
    else:
        y = out
    n = np.size(x, axis)
    y[:] = 418.9829 * n - np.sum(x * np.sin(np.sqrt(abs(x))), axis)
    return y

# def objective(x):  # rast.m
#      x = np.asarray_chkfinite(x)
#      n = len(x)
#      return 10*n + sum( x**2 - 10 * np.cos( 2 * np.pi * x ))


# def objective(x):  # rosen.m
#     x = np.asarray_chkfinite(x)
#     x0 = x[:-1]
#     x1 = x[1:]
#     return sum((1 - x0) ** 2) + 100 * sum((x1 - x0 ** 2) ** 2)

# def objective(x):  # schw.m
#     x = np.asarray_chkfinite(x)
#     n = len(x)
#     return 418.9829*n - sum( x * np.sin(np.sqrt( abs( x ))))
