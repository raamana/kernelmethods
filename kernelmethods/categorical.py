"""

Module for categorical kernels

Please refer to the following papers and theses for more details:

 - Villegas García, Marco Antonio. "An investigation into new kernels for
   categorical variables." Master's thesis, Universitat Politècnica de Catalunya,
   2013.


"""

import numpy as np

from kernelmethods.base import BaseKernelFunction
from kernelmethods.utils import check_input_arrays
from kernelmethods import config as cfg


class MatchCountKernel(BaseKernelFunction):
    """
    Categorical kernel measuring similarity via the number of matching categorical
    dimensions.

    Parameters
    ----------

    return_perc : bool
        If True, the return value would be normalized by the number of dimensions.

    References
    ----------

    Villegas García, Marco A., "An investigation into new kernels for categorical
    variables." Master's thesis, Universitat Politècnica de Catalunya, 2013.

    """


    def __init__(self,
                 return_perc=True,
                 skip_input_checks=False):
        """Constructor."""

        self.return_perc = return_perc
        if self.return_perc:
            super().__init__('MatchPerc')
        else:
            super().__init__('MatchCount')

        self.skip_input_checks = skip_input_checks


    def __call__(self, vec_c, vec_d):
        """
        Actual implementation of the kernel func.

        Parameters
        ----------

        vec_c, vec_d : array of equal-sized categorical variables

        """

        vec_c, vec_d = _check_categorical_arrays(vec_c, vec_d)

        if not np.issubdtype(vec_c.dtype, cfg.dtype_categorical) or \
            not np.issubdtype(vec_d.dtype, cfg.dtype_categorical):
            raise TypeError('Categorical kernels require str or unicode dtype')

        match_count = np.sum(vec_c==vec_d)

        if self.return_perc:
            return match_count / len(vec_d)
        else:
            return match_count


    def __str__(self):
        """human readable repr"""

        return self.name


def _check_categorical_arrays(x, y):
    """
    Ensures the inputs are
    1) 1D arrays (not matrices)
    2) with compatible size
    3) of categorical data type
    and hence are safe to operate on.

    This is a variation of utils.check_input_arrays() to accommodate the special
    needs for categorical dtype, where we do not have lists of
    originally numbers/bool data to be converted to strings, and assume they are
    categorical.

    Parameters
    ----------
    x : iterable
    y : iterable

    Returns
    -------
    x : ndarray
    y : ndarray
    """

    x = _ensure_type_size(x, ensure_num_dim=1)
    y = _ensure_type_size(y, ensure_num_dim=1)

    if x.size != y.size:
        raise ValueError('x (n={}) and y (n={}) differ in size! '
                         'They must be of same length'.format(x.size, y.size))

    return x, y


def _ensure_type_size(array, ensure_num_dim=1):
    """Checking type and size of arrays"""

    if not isinstance(array, np.ndarray):
        array = np.squeeze(np.asarray(array))

    if array.ndim != ensure_num_dim:
        raise ValueError('array must be {}-dimensional! '
                         'It has {} dims with shape {} '
                         ''.format(ensure_num_dim, array.ndim, array.shape))

    return array
