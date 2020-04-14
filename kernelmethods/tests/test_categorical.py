import random
import string
import traceback
from numbers import Number

import numpy as np
from hypothesis import (HealthCheck, given, settings as hyp_settings, strategies,
                        unlimited)
from kernelmethods.base import KernelMatrix
from kernelmethods.categorical import MatchCountKernel
from kernelmethods.config import dtype_categorical
from kernelmethods.operations import is_positive_semidefinite
from kernelmethods.utils import check_callable
from pytest import raises

default_feature_dim = 10
range_feature_dim = [10, 500]
range_num_samples = [50, 500]
range_string_length = [3, 25]

np.random.seed(42)

# choosing skip_input_checks=False will speed up test runs
# default values for parameters
SupportedKernels = (MatchCountKernel(),)
num_tests_psd_kernel = 3


def random_string(length=5):
    return ''.join(random.choices(string.ascii_letters, k=length))


def gen_random_categorical_array(dim, length):
    """To better control precision and type of floats"""

    return np.array([random_string(length) for _ in range(dim)],
                    dtype=dtype_categorical)


def gen_random_sample(num_samples, sample_dim, string_length):
    """To better control precision and type of floats"""

    return np.array([gen_random_categorical_array(sample_dim, string_length) for
                     _ in range(num_samples)])


def _test_for_all_kernels(kernel, sample_dim, string_length):
    """Common tests that all kernels must pass."""

    x = gen_random_categorical_array(sample_dim, string_length)
    y = gen_random_categorical_array(sample_dim, string_length)

    try:
        result = kernel(x, y)
    except Exception:
        traceback.print_exc()
        raise RuntimeError('{} unable to calculate!\n'
                           ' on x {}\n y{}'.format(kernel, x, y))

    if not isinstance(result, Number):
        raise ValueError('result {} of type {} is not a number!\n'
                         'x={}\ny={}\nkernel={}\n'
                         ''.format(result, type(result), x, y, kernel))

    if kernel(y, x) != result:
        raise ValueError('{} is not symmetric!'
                         'x={}\n y={}\n kernel={}\n'.format(kernel.name, x, y,
                                                            kernel))


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
        for non_catg in [(True, False, True),
                         [1.0, 2.4],
                         [object, object]]:
            with raises(TypeError):
                _ = kernel(non_catg, non_catg)


def _test_func_is_valid_kernel(kernel, sample_dim, num_samples, string_length):
    """A func is a valid kernel if the kernel matrix generated by it is PSD.

    Not including this in tests for all kernels to allow for non-PSD kernels in
    the future

    """

    KM = KernelMatrix(kernel, name='TestKM')
    KM.attach_to(gen_random_sample(num_samples, sample_dim, string_length))
    is_psd = is_positive_semidefinite(KM.full, verbose=True)
    if not is_psd:
        raise ValueError('{} is not PSD'.format(str(KM)))


@hyp_settings(max_examples=num_tests_psd_kernel, deadline=None,
              suppress_health_check=HealthCheck.all())
@given(strategies.integers(range_feature_dim[0], range_feature_dim[1]),
       strategies.integers(range_num_samples[0], range_num_samples[1]),
       strategies.integers(range_string_length[0], range_string_length[1]),
       strategies.booleans())
def test_match_count_kernel(sample_dim, num_samples, string_length, perc_flag):
    """Tests specific for Polynomial kernel."""

    poly = MatchCountKernel(return_perc=perc_flag, skip_input_checks=False)
    _test_for_all_kernels(poly, sample_dim, string_length)
    _test_func_is_valid_kernel(poly, sample_dim, num_samples, string_length)
