
import numpy as np
from numbers import Number
from pytest import raises, warns
from hypothesis import given, strategies, unlimited
from hypothesis import settings as hyp_settings
from hypothesis import HealthCheck
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel, \
    LaplacianKernel
from kernelmethods.utils import check_callable
from kernelmethods.base import KernelMatrix, KernelFromCallable, BaseKernelFunction
from kernelmethods.operations import is_positive_semidefinite
from kernelmethods.config import KMAccessError, KMNormError

default_feature_dim = 10
range_feature_dim = [10, 500]
range_num_samples = [50, 500]
num_samples = 10
sample_dim = 4

range_polynomial_degree = [2, 10] # degree=1 is tested in LinearKernel()

np.random.seed(42)

# choosing skip_input_checks=False will speed up test runs
# default values for parameters
SupportedKernels = (GaussianKernel(), PolyKernel(), LinearKernel(),
                    LaplacianKernel())
num_tests_psd_kernel = 3

def gen_random_array(dim):
    """To better control precision and type of floats"""

    # TODO input sparse arrays for test
    return np.random.rand(dim)

def gen_random_sample(num_samples, sample_dim):
    """To better control precision and type of floats"""

    # TODO input sparse arrays for test
    return np.random.rand(num_samples, sample_dim)

km_lin = KernelMatrix(kernel=LinearKernel())
km_lin.attach_to(gen_random_sample(num_samples, sample_dim))

def simple_callable(x, y):
    return np.dot(x, y)

def test_kernel_from_callable():

    kf = KernelFromCallable(simple_callable)
    if not isinstance(kf, BaseKernelFunction):
        raise TypeError('Error in implementation of KernelFromCallable')


def test_KernelMatrix_design():

    with raises(TypeError):
        km = KernelMatrix(kernel=simple_callable)

    with raises(TypeError):
        km = KernelMatrix(kernel=LinearKernel, normalized='True')

    assert len(km_lin) == num_samples**2

    colon_access = km_lin[:,:]
    if colon_access.size != km_lin.size:
        raise ValueError('error in getitem implementation when using [:, :]')

    _ = km_lin[1, :]
    _ = km_lin[:, 1]
    for invalid_index in (-1, np.Inf, np.NaN):
        with raises(KMAccessError):
            _ = km_lin[:, invalid_index]


def test_centering():

    km = KernelMatrix(kernel=LinearKernel())
    km.attach_to(gen_random_sample(num_samples, sample_dim))
    km.center()


def test_normalize():

    km = KernelMatrix(kernel=LinearKernel())
    km.attach_to(gen_random_sample(num_samples, sample_dim))
    km.normalize()


def test_KM_results_in_NaN_Inf():
    """"""
    pass
