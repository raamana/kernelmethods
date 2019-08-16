
import numpy as np
from numbers import Number
from pytest import raises, warns
from hypothesis import given, strategies, unlimited
from hypothesis import settings as hyp_settings
from hypothesis import HealthCheck
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel, \
    LaplacianKernel
from kernelmethods.utils import check_callable, check_input_arrays, \
    check_operation_kernel_matrix, ensure_ndarray_1D, ensure_ndarray_2D, \
    get_callable_name, min_max_scale, not_symmetric
from kernelmethods.base import KernelMatrix, KernelFromCallable, \
    BaseKernelFunction, KernelMatrixPrecomputed, ConstantKernelMatrix
from kernelmethods.operations import is_positive_semidefinite
from kernelmethods.config import KMAccessError, KMNormError

default_feature_dim = 10
range_feature_dim = [10, 500]
range_num_samples = [50, 500]
num_samples = np.random.randint(20)
sample_dim = np.random.randint(10)

range_polynomial_degree = [2, 10] # degree=1 is tested in LinearKernel()

np.random.seed(42)

# choosing skip_input_checks=False will speed up test runs
# default values for parameters
SupportedKernels = (GaussianKernel(), PolyKernel(), LinearKernel(),
                    LaplacianKernel())
num_tests_psd_kernel = 3


def test_inpur_arrays():

    with raises(ValueError):
        check_input_arrays(np.random.rand(10, 5), np.random.rand(5, 4))

    # from scipy.sparse import csr_matrix
    # s1 = csr_matrix((3,4))
    # s2 = csr_matrix((3, 4))
    # _, _ = check_input_arrays(s1, s2)



def test_valid_op():

    for invalid_op in ('foo', 'bar', 'adition', 'some'):
        with raises(ValueError):
            check_operation_kernel_matrix(invalid_op)

def test_ensure_array_dim():

    with raises(ValueError):
        ensure_ndarray_2D(np.random.rand(10, 5), ensure_num_cols=3)

    with raises(ValueError):
        ensure_ndarray_2D(np.random.rand(10), ensure_num_cols=3)

    with raises(ValueError):
        ensure_ndarray_1D(np.random.rand(10, 5))

    with raises(ValueError):
        ensure_ndarray_1D(np.random.rand(10, 5, 10))

def test_misc():

    _ = get_callable_name(test_ensure_array_dim, 'test')
    _ = get_callable_name('test_ensure_array_dim', None)

    with raises(TypeError):
        check_callable('kdjkj')

    def func_with_less_than_min_args(): return None

    with raises(TypeError):
        check_callable(func_with_less_than_min_args)

    with raises(TypeError):
        check_callable(func_with_less_than_min_args, 1)

    with raises(TypeError):
        check_callable(func_with_less_than_min_args, 3)
