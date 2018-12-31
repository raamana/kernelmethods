
import numpy as np
from scipy.sparse import issparse

from kernelmethods.base import BaseKernelFunction
from kernelmethods.utils import check_input_arrays, _ensure_min_eps

# TODO special handling for sparse arrays
#   (e.g. custom dot product during kernel evaluation might be more efficient


class PolyKernel(BaseKernelFunction):
    """Polynomial kernel function"""

    def __init__(self, degree=2, b=0, skip_input_checks=False):
        """
        Constructor

        Parameters
        ----------
        degree : int
            degree to raise the inner product

        b : float
            intercept

        skip_input_checks : bool
            Flag to skip input validation to save time.
            Skipping validation is strongly discouraged for normal use,
            unless you know exactly what you are doing (expert users).

        """

        super().__init__(name='polynomial')

        # TODO implement param check
        self.degree = degree
        self.b = b

        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.number)

        return (self.b + x.dot(y.T)) ** self.degree

    def __str__(self):
        """human readable repr"""

        return "{}(degree={},b={})".format(self.name, self.degree, self.b)

    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


class GaussianKernel(BaseKernelFunction):
    """Polynomial kernel function"""

    def __init__(self, sigma=2.0, skip_input_checks=False):
        """
        Constructor

        Parameters
        ----------
        sigma : float
            bandwidth

        skip_input_checks : bool
            Flag to skip input validation to save time.
            Skipping validation is strongly discouraged for normal use,
            unless you know exactly what you are doing (expert users).

        """

        super().__init__(name='gaussian')

        # TODO implement param check
        # ensuring values of gamma/gamma is eps or larger to avoid zero division
        self.sigma = _ensure_min_eps(sigma)
        self.gamma = _ensure_min_eps(1.0/(2*self.sigma**2))

        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.number)

        return np.exp(-self.gamma * np.linalg.norm(x - y, ord=2)**2)

    def __str__(self):
        """human readable repr"""

        return "{}(sigma={})".format(self.name, self.sigma)

    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


class LinearKernel(BaseKernelFunction):
    """Linear kernel function"""

    def __init__(self, skip_input_checks=False):
        """
        Constructor

        Parameters
        ----------
        skip_input_checks : bool
            Flag to skip input validation to save time.
            Skipping validation is strongly discouraged for normal use,
            unless you know exactly what you are doing (expert users).

        """

        super().__init__(name='LinearKernel')
        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.number)

        return x.dot(y.T)

    def __str__(self):
        """human readable repr"""

        return self.name

    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__
