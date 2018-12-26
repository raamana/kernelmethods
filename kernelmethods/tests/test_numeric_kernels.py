
import numpy as np
from numbers import Number
from hypothesis import given, strategies
from hypothesis import settings as hyp_settings

from kernelmethods.numeric_kernels import PolyKernel

default_feature_dim = 10
range_feature_dim = [10, 10000]
range_polynomial_degree = [1, 10]

SupportedKernels = (PolyKernel(), )

def gen_random_array(dim):
    """To better control precision and type of floats"""

    return np.random.rand(dim)


@hyp_settings(max_examples=100)
@given(strategies.integers(range_feature_dim[0], range_feature_dim[1]))
def test_kernel_design(sample_dim):
    """
    Every kernel must be
    1. must have a name defined
    2. must be callable with two samples
    3. returns a number

    """

    x = gen_random_array(sample_dim)
    y = gen_random_array(sample_dim)

    for kernel in SupportedKernels:

        if not hasattr(kernel, 'name'):
            raise TypeError('{} does not have name attribute!'.format(kernel))

        try:
            result = kernel(x, y)
        except Exception:
            raise SyntaxError('{} is not callable!'.format(kernel.name))

        if not isinstance(result, Number):
            raise ValueError('result from {} is not a number!'.format(kernel.name))
