
import numpy as np
from kernelmethods.base import BaseKernelFunction
from kernelmethods.config import Chi2NegativeValuesException
from kernelmethods.utils import _ensure_min_eps, check_input_arrays


# TODO special handling for sparse arrays
#   (e.g. custom dot product during kernel evaluation might be more efficient


class PolyKernel(BaseKernelFunction):
    """Polynomial kernel function

    Formula::
        K(x, y) = ( b + gamma*<x, y> )^degree

    Parameters
    ----------
    degree : int
        degree to raise the inner product

    gamma : float
        scaling factor

    b : float
        intercept

    skip_input_checks : bool
        Flag to skip input validation to save time.
        Skipping validation is strongly discouraged for normal use,
        unless you know exactly what you are doing (expert users).
    """

    def __init__(self, degree=3, gamma=1.0, b=1.0, skip_input_checks=False):
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
        self.gamma = gamma
        self.b = b

        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.number)

        return (self.b + self.gamma*np.dot(x, y)) ** self.degree

    def __str__(self):
        """human readable repr"""

        return "{}(degree={},gamma={},b={})".format(self.name, self.degree,
                                                    self.gamma, self.b)


class GaussianKernel(BaseKernelFunction):
    """Gaussian kernel function

    Parameters
    ----------
    sigma : float
        bandwidth

    skip_input_checks : bool
        Flag to skip input validation to save time.
        Skipping validation is strongly discouraged for normal use,
        unless you know exactly what you are doing (expert users).

    """

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


class LaplacianKernel(BaseKernelFunction):
    """Laplacian kernel function

    Parameters
    ----------
    gamma : float
        scale factor

    skip_input_checks : bool
        Flag to skip input validation to save time.
        Skipping validation is strongly discouraged for normal use,
        unless you know exactly what you are doing (expert users).

    """

    def __init__(self, gamma=1.0, skip_input_checks=False):
        """
        Constructor

        Parameters
        ----------
        gamma : float
            scale factor

        skip_input_checks : bool
            Flag to skip input validation to save time.
            Skipping validation is strongly discouraged for normal use,
            unless you know exactly what you are doing (expert users).

        """

        super().__init__(name='laplacian')

        self.gamma = gamma

        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.number)

        return np.exp(-self.gamma * np.sum(np.abs(x - y)))

    def __str__(self):
        """human readable repr"""

        return "{}(gamma={})".format(self.name, self.gamma)


class Chi2Kernel(BaseKernelFunction):
    """Chi-squared kernel function

    This kernel is implemented as::

        k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])

    x and y must have non-negative values (>=0).

    As a division is involved, when x+y is 0 or when x+y and x-y are both 0 for a
    particular dimension, the division results in a NaN, which is currently
    being ignored, by summing only non-NaN values. If your feature sets have many
    zeros, you may want investigate the effect of this kernel on your dataset
    carefully to ensure you understand this kernel meets your needs and
    expectations.

    Parameters
    ----------
    gamma : float
        scale factor

    skip_input_checks : bool
        Flag to skip input validation to save time.
        Skipping validation is strongly discouraged for normal use,
        unless you know exactly what you are doing (expert users).

    """

    def __init__(self, gamma=1.0, skip_input_checks=False):
        """
        Constructor

        Parameters
        ----------
        gamma : float
            scale factor

        skip_input_checks : bool
            Flag to skip input validation to save time.
            Skipping validation is strongly discouraged for normal use,
            unless you know exactly what you are doing (expert users).

        """

        super().__init__(name='chi2')

        self.gamma = gamma

        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.float64)

        if (x < 0).any() or (y < 0).any():
            raise Chi2NegativeValuesException(
                'Chi^2 kernel requires non-negative values!'
                ' x or y contains non-negative values')

        # Note: NaNs due to Zero division are being ignored via np.nansum!
        value = np.exp(-self.gamma * np.nansum(np.power(x - y, 2) / (x + y)))

        return value

    def __str__(self):
        """human readable repr"""

        return "{}(gamma={})".format(self.name, self.gamma)


class SigmoidKernel(BaseKernelFunction):
    """
    Sigmoid kernel function (also known as hyperbolic tangent kernel)

    NOTE: This kernel is not always PSD, and normalizing its kernel matrix can
    result in numerical issues or errors.

    Parameters
    ----------
    gamma : float
        scale factor

    offset : float
        value of offset/bias

    skip_input_checks : bool
        Flag to skip input validation to save time.
        Skipping validation is strongly discouraged for normal use,
        unless you know exactly what you are doing (expert users).

    """

    def __init__(self, gamma=1.0, offset=1.0, skip_input_checks=False):
        """
        Constructor

        Parameters
        ----------
        gamma : float
            scale factor

        offset : float
            value of offset/bias

        skip_input_checks : bool
            Flag to skip input validation to save time.
            Skipping validation is strongly discouraged for normal use,
            unless you know exactly what you are doing (expert users).

        """

        super().__init__(name='sigmoid')

        self.gamma = gamma
        self.offset = offset

        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.number)

        return np.tanh(self.offset + (self.gamma * np.dot(x, y)))

    def __str__(self):
        """human readable repr"""

        return "{}(gamma={},offset={})".format(self.name, self.gamma, self.offset)


class LinearKernel(BaseKernelFunction):
    """Linear kernel function

    Parameters
    ----------
    skip_input_checks : bool
        Flag to skip input validation to save time.
        Skipping validation is strongly discouraged for normal use,
        unless you know exactly what you are doing (expert users).
    """

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

        super().__init__(name='linear')
        self.skip_input_checks = skip_input_checks

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        if not self.skip_input_checks:
            x, y = check_input_arrays(x, y, ensure_dtype=np.number)

        return x.dot(y.T)

    def __str__(self):
        """human readable repr"""

        return self.name


DEFINED_KERNEL_FUNCS = (PolyKernel(),
                        GaussianKernel(),
                        LaplacianKernel(),
                        LinearKernel(),
                        SigmoidKernel(),
                        )
