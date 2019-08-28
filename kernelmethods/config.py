from operator import add, mul
import numpy as np

class KernelMethodsException(Exception):
    """
    Generic exception to indicate invalid use of the ``kernelmethods`` library.

    Allows to distinguish improper use of KernelMatrix from other code exceptions
    """
    pass


class KMAccessError(KernelMethodsException):
    """Exception to indicate invalid access to the kernel matrix elements!"""
    pass


class KMNormError(KernelMethodsException):
    """Custom exception to indicate error during normalization of kernel matrix"""
    pass


class KMSetAdditionError(KernelMethodsException):
    """Exception to indicate invalid addition of kernel matrix to a KernelSet"""
    pass


class KernelMethodsWarning(Warning):
    """Custom warning to indicate kernelmethods-specific warning!"""
    pass


class Chi2NegativeValuesException(KernelMethodsException):
    """Custom exception to indicate Chi^2 kernel requires non-negative values"""
    pass


VALID_KERNEL_MATRIX_OPS = ('sum', 'product', 'average')

OPER_KM_OPS = {'sum'    : add,
               'product': mul}


# default values and ranges

kernel_bucket_strategies = ('exhaustive', 'light', 'linear_only')
# strategy: exhaustive
default_degree_values_poly_kernel = (2, 3, 4)
default_sigma_values_gaussian_kernel = tuple([2**exp for exp in range(-5, 6, 2)])
default_gamma_values_laplacian_kernel = tuple([2**exp for exp in range(-5, 7, 2)])
default_gamma_values_sigmoid_kernel = tuple([2**exp for exp in range(-5, 7, 2)])
default_offset_values_sigmoid_kernel = tuple([-2.0, 1.0, 2.0])

# light
light_degree_values_poly_kernel = (2, 3, )
light_sigma_values_gaussian_kernel = tuple([2**exp for exp in range(-3, 3, 2)])
light_gamma_values_laplacian_kernel = tuple([2**exp for exp in range(-3, 3, 2)])
light_gamma_values_sigmoid_kernel = tuple([2**exp for exp in range(-3, 7, 2)])
light_offset_values_sigmoid_kernel = tuple([1.0, ])

# ranking

VALID_RANKING_METHODS = ("align/corr", "cv_risk")

# controls the precision for kernel_matrix elements
km_dtype = np.dtype('f8')

# categorical variables
dtype_categorical = np.unicode_

