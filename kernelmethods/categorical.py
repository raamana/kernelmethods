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

        vec_c, vec_d = check_input_arrays(vec_c, vec_d,
                                          ensure_dtype=cfg.dtype_categorical)

        match_count = np.sum(vec_c==vec_d)

        if self.return_perc:
            return match_count / len(vec_d)
        else:
            return match_count


    def __str__(self):
        """human readable repr"""

        return self.name
