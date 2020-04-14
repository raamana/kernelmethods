
import numpy as np
from pytest import raises

from kernelmethods.base import (AverageKernel, BaseKernelFunction, CompositeKernel,
                                KernelFromCallable, KernelMatrix,
                                KernelMatrixPrecomputed, ProductKernel,
                                SumKernel, WeightedAverageKernel)
from kernelmethods.config import KMAccessError
from kernelmethods.numeric_kernels import (GaussianKernel, LaplacianKernel,
                                           LinearKernel, PolyKernel)
from kernelmethods.sampling import make_kernel_bucket
from kernelmethods.tests.test_numeric_kernels import _test_for_all_kernels

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

    _test_for_all_kernels(kf, 5)


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


def test_km_precomputed():

    rand_size = np.random.randint(5, 50)
    rand_matrix = np.random.rand(rand_size, rand_size)
    # making symmetric
    rand_matrix = rand_matrix + rand_matrix.T
    pre = KernelMatrixPrecomputed(rand_matrix, name='rand')

    assert pre.size == rand_size == len(pre)
    assert np.isclose(pre.full, rand_matrix).all()
    assert np.isclose(pre.diag, rand_matrix.diagonal()).all()
    # __getitem__
    for _ in range(min(5, rand_size)):
        indices = np.random.randint(0, rand_size, 2)
        assert pre[indices[0], indices[1]] == rand_matrix[indices[0], indices[1]]

    with raises(ValueError): # not symmtric
        pre = KernelMatrixPrecomputed(np.random.rand(rand_size, rand_size+1))

    with raises(ValueError):
        pre = KernelMatrixPrecomputed([[1, 2], [2, 3, 4, 9]])

    # 3D or 1D
    with raises(ValueError):
        pre = KernelMatrixPrecomputed(np.random.rand(rand_size, rand_size, 2))

    with raises(ValueError):
        pre = KernelMatrixPrecomputed(np.random.rand(rand_size))

    # must have real values
    with raises(ValueError):
        pre = KernelMatrixPrecomputed([[1, 2+4j], [9+2j, 3]])

    with raises(KMAccessError):
        _= pre[np.Inf, 0]


def test_composite_kernels():

    kset = make_kernel_bucket()
    kset.attach_to(gen_random_sample(num_samples, sample_dim))

    for ck in (AverageKernel, SumKernel, WeightedAverageKernel, ProductKernel):

        if issubclass(ck, WeightedAverageKernel):
            result_km = ck(kset, np.random.rand(kset.size))
        else:
            result_km = ck(kset)

        if not isinstance(result_km, CompositeKernel):
            raise TypeError(' Composite kernel {} not defined properly: '
                            'it must be a child of {}'
                            ''.format(result_km, CompositeKernel))

        result_km.fit()

        reqd_attrs = ('composite_KM', 'full')
        for reqd in reqd_attrs:
            if not hasattr(result_km, reqd):
                raise TypeError('{} does not have attr {}'.format(result_km, reqd))
