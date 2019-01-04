
import numpy as np
from scipy.sparse import issparse
from kernelmethods import config

def check_input_arrays(x, y, ensure_dtype=np.number):
    """
    Ensures the inputs are
    1) 1D arrays (not matrices)
    2) with compatible size
    3) of a particular data type
    and hence are safe to operate on.

    Parameters
    ----------
    x : iterable

    y : iterable

    ensure_dtype : dtype

    Returns
    -------
    x : ndarray

    y : ndarray

    """

    x = ensure_ndarray_1D(x, ensure_dtype)
    y = ensure_ndarray_1D(y, ensure_dtype)

    if x.size != y.size:
        raise ValueError('x (n={}) and y (n={}) differ in size! '
                         'They must be of same length'.format(x.size, y.size))

    # sparse to dense
    if issparse(x):
        x = np.array(x.todense())

    if issparse(y):
        y = np.array(y.todense())

    return x, y


def ensure_ndarray_2D(array, ensure_dtype=np.number):
    """Converts the input to a numpy array and ensure it is 1D."""

    return ensure_ndarray_size(array, ensure_dtype=ensure_dtype, ensure_num_dim=2)


def ensure_ndarray_1D(array, ensure_dtype=np.number):
    """Converts the input to a numpy array and ensure it is 1D."""

    return ensure_ndarray_size(array, ensure_dtype=ensure_dtype, ensure_num_dim=1)


def ensure_ndarray_size(array, ensure_dtype=np.number, ensure_num_dim=1):
    """Converts the input to a numpy array and ensure it is of specified dim."""

    if not isinstance(array, np.ndarray):
        array = np.squeeze(np.asarray(array))

    if array.ndim != ensure_num_dim:
        raise ValueError('array must be {}-dimensional! '
                         'It has {} dims with shape {} '
                         ''.format(ensure_num_dim, array.ndim, array.shape))

    if not np.issubdtype(array.dtype, ensure_dtype):
        raise ValueError('input data type {} is not compatible with the required {}'
                         ''.format(array.dtype, ensure_dtype))

    return array


def check_callable(input_func, min_num_args=2):
    """Ensures input func is callable, and can accept a min # args"""

    if not callable(input_func):
        raise TypeError('Input function must be callable!')

    from inspect import signature
    # would not work for C/builtin functions such as numpy.dot
    func_signature = signature(input_func)

    if len(func_signature.parameters) < min_num_args:
        raise TypeError('Input func must accept atleast {} inputs'.format(min_num_args))

    return input_func


def get_callable_name(input_func, name=None):
    """Provide a callable name"""

    if name is None:
        if hasattr(input_func, '__name__'):
            return input_func.__name__
        else:
            return ''
    else:
        return str(name)

_float_eps = np.finfo('float').eps

def _ensure_min_eps(x):
    return  np.maximum(_float_eps, x)

def not_symmetric(matrix):
    """Returns true if matrix is not symmetric."""

    if not np.isclose(matrix, matrix.T).all():
        return True
    else:
        return False

def check_operation_kernel_matrix(operation):
    """Validates whether input is a valid kernel matrix"""

    opr = operation.lower()
    if opr not in config.VALID_KERNEL_MATRIX_OPS:
        raise ValueError('Invalid kernel matrix operation - must be one of:\n{}'
                         ''.format(config.VALID_KERNEL_MATRIX_OPS))

    return opr
