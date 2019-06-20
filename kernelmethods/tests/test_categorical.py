
import numpy as np
from numbers import Number
from pytest import raises, warns
from hypothesis import given, strategies, unlimited
from hypothesis import settings as hyp_settings
from hypothesis import HealthCheck
from kernelmethods.categorical import MatchCountKernel
from kernelmethods.utils import check_callable
from kernelmethods.base import KernelMatrix
from kernelmethods.operations import is_positive_semidefinite
import string
import random

default_feature_dim = 10
range_feature_dim = [10, 500]
range_num_samples = [50, 500]

np.random.seed(42)

# choosing skip_input_checks=False will speed up test runs
# default values for parameters
SupportedKernels = (MatchCountKernel(), )
num_tests_psd_kernel = 3

def random_string(length=5):

    return ''.join(random.choices(string.ascii_letters, k=length))

def gen_random_categorical_array(dim, length):
    """To better control precision and type of floats"""

    return [random_string(length) for _ in range(dim)]

def gen_random_sample(num_samples, sample_dim, string_length):
    """To better control precision and type of floats"""

    return np.array([gen_random_categorical_array(sample_dim, string_length) for
                    _ in range(num_samples)])


def _test_for_all_kernels(kernel, sample_dim):
    """Common tests that all kernels must pass."""

    x = gen_random_categorical_array(sample_dim)
    y = gen_random_categorical_array(sample_dim)

    try:
        result = kernel(x, y)
    except Exception:
        raise RuntimeError('{} unable to calculate!\n'
                           ' on x {}\n y{}'.format(kernel, x, y))

    if not isinstance(result, Number):
        raise ValueError('result {} of type {} is not a number!\n'
                         'x={}\ny={}\nkernel={}\n'
                         ''.format(result, type(result), x, y, kernel))

    if kernel(y, x) != result:
        raise ValueError('{} is not symmetric!'
                         'x={}\n y={}\n kernel={}\n'.format(kernel.name, x, y, kernel))


def test_kernel_design():
    """
    Every kernel must be
    1. must have a name defined
    2. must be callable with two samples
    3. returns a number

    """

    for kernel in SupportedKernels:

        # must be callable with 2 args
        check_callable(kernel, min_num_args=2)

        if not hasattr(kernel, 'name'):
            raise TypeError('{} does not have name attribute!'.format(kernel))

        # only numeric data is accepted and other dtypes must raise an error
        for non_num in ['string',
                        (True, False, True),
                        [object, object] ]:
            with raises(ValueError):
                _ = kernel(non_num, non_num)
