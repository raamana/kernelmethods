
from scipy.sparse import issparse
import numpy as np

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

    x = ensure_ndarray_1d(x, ensure_dtype)
    y = ensure_ndarray_1d(y, ensure_dtype)

    if x.size != y.size:
        raise ValueError('x (n={}) and y (n={}) differ in size! '
                         'They must be of same length'.format(x.size, y.size))

    # sparse to dense
    if issparse(x):
        x = np.array(x.todense())

    if issparse(y):
        y = np.array(y.todense())

    return x, y


def ensure_ndarray_1d(array, ensure_dtype=np.number):
    """Converts the input to a numpy array and ensure it is 1D."""

    if not isinstance(array, np.ndarray):
        array = np.squeeze(np.asarray(array))

    if array.ndim > 1:
        raise ValueError('array must be 1-dimensional! '
                         'It has {} dims with shape {}'.format(array.ndim, array.shape))

    if not np.issubdtype(array.dtype, ensure_dtype):
        raise ValueError('input data type {} is not compatible with the required {}'
                         ''.format(array.dtype, ensure_dtype))

    return array
