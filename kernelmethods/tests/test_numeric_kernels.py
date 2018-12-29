
import numpy as np
from numbers import Number
from pytest import raises, warns
from hypothesis import given, strategies
from hypothesis import settings as hyp_settings

from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel
from kernelmethods.utils import check_callable

default_feature_dim = 10
range_feature_dim = [10, 10000]
range_polynomial_degree = [1, 10]

# choosing skip_input_checks=False will speed up test runs
# default values for parameters
SupportedKernels = (GaussianKernel(), PolyKernel())

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

    for kernel in SupportedKernels:

        # must be callable with 2 args
        check_callable(kernel, min_num_args=2)

        if not hasattr(kernel, 'name'):
            raise TypeError('{} does not have name attribute!'.format(kernel))

        try:
            result = kernel(x, y)
        except Exception:
            raise SyntaxError('{} is not callable!'.format(kernel.name))

        if not isinstance(result, Number):
            raise ValueError('result {} of type {} from {} kernel is not a number!'
                             ''.format(result, type(result), kernel.name))

        if kernel(y, x) != result:
            raise ValueError('{} is not symmetric!'.format(kernel.name))

        # only numeric data is accepted and other dtypes must raise an error
        for non_num in ['string',
                        (True, False, True),
                        [object, object] ]:
            with raises(ValueError):
                _ = kernel(non_num, non_num)


@hyp_settings(max_examples=1000)
@given(strategies.integers(range_feature_dim[0], range_feature_dim[1]),
       strategies.integers(range_polynomial_degree[0], range_polynomial_degree[1]),
       strategies.floats(min_value=0, max_value=np.Inf,
                         allow_nan=False, allow_infinity=False))
def test_polynomial_kernel(sample_dim, poly_degree, poly_intercept):
    """Tests specific for Polynomial kernel."""

    # TODO input sparse arrays for test
    x = gen_random_array(sample_dim)
    y = gen_random_array(sample_dim)
    poly = PolyKernel(degree=poly_degree, b=poly_intercept)

    try:
        result = poly(x, y)
    except RuntimeWarning:
        raise RuntimeWarning('RunTime warning for:\n'
                             ' x={}\n y={}\n kernel={}\n'.format(x, y, poly))
    except Exception:
        raise Exception('unanticipated exception:\n'
                        ' x={}\n y={}\n kernel={}\n'.format(x, y, poly))

    if not isinstance(result, Number):
        raise ValueError('poly kernel result {} is not a number!\n'
                         'x={}\ny={}\nkernel={}\n'
                         ''.format(result, x, y, poly))


@hyp_settings(max_examples=1000)
@given(strategies.integers(range_feature_dim[0], range_feature_dim[1]),
       strategies.floats(min_value=0, max_value=np.Inf,
                         allow_nan=False, allow_infinity=False))
def test_gaussian_kernel(sample_dim, sigma):
    """Tests specific for Polynomial kernel."""

    # TODO input sparse arrays for test
    x = gen_random_array(sample_dim)
    y = gen_random_array(sample_dim)
    gaussian = GaussianKernel(sigma=sigma)

    try:
        result = gaussian(x, y)
    except RuntimeWarning:
        raise RuntimeWarning('RunTime warning for:\n'
                             ' x={}\n y={}\n kernel={}\n'.format(x, y, gaussian))
    except Exception:
        raise Exception('unanticipated exception:\n'
                        ' x={}\n y={}\n kernel={}\n'.format(x, y, gaussian))

    if not isinstance(result, Number):
        raise ValueError('gaussian kernel result {} is not a number!\n'
                         'x={}\ny={}\nkernel={}\n'
                         ''.format(result, x, y, gaussian))

