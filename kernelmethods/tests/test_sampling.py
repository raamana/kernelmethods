
from kernelmethods.operations import center_km, frobenius_product, frobenius_norm, \
    normalize_km, normalize_km_2sample, alignment_centered, linear_combination
from kernelmethods.config import KMNormError
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, SigmoidKernel, \
    LaplacianKernel
from kernelmethods.utils import check_callable
from kernelmethods.base import KernelMatrix, KernelFromCallable, \
    BaseKernelFunction
from kernelmethods.sampling import make_kernel_bucket, KernelBucket, \
    ideal_kernel, pairwise_similarity, correlation_km
from kernelmethods.config import KernelMethodsException, KernelMethodsWarning,\
    kernel_bucket_strategies

import numpy as np
from scipy.sparse import issparse
from scipy.linalg import eigh

from pytest import raises, warns

num_samples = 50 # 9
sample_dim = 3 # 2
target_label_set = [1, 2]

sample_data = np.random.rand(num_samples, sample_dim)
target_labels = np.random.choice(target_label_set, (num_samples, 1))

A = np.random.rand(4, 4)
B = np.random.rand(4, 4)

def gen_random_array(dim):
    """To better control precision and type of floats"""

    # TODO input sparse arrays for test
    return np.random.rand(dim)

def gen_random_sample(num_samples, sample_dim):
    """To better control precision and type of floats"""

    # TODO input sparse arrays for test
    return np.random.rand(num_samples, sample_dim)

kset = make_kernel_bucket('light')
kset.attach_to(sample_data)

def test_make_bucket():

    with warns(UserWarning):
        _ = make_kernel_bucket(kset)

    with raises(ValueError):
        _ = make_kernel_bucket('blah_invalid_strategy')

    # ensure correct values work
    for strategy in kernel_bucket_strategies:
        _ = make_kernel_bucket(strategy=strategy)

def test_KB_class():

    for param in ['normalize_kernels', 'skip_input_checks']:
        for invalid_value in (1, 'str', 34., 2+4j):
            with raises(TypeError):
                _ = KernelBucket(**{param: invalid_value})



def test_add_parametrized_kernels():

    kb = KernelBucket()
    for invalid_kfunc in ('kfunc', gen_random_sample, KernelBucket, ):
        with raises(KernelMethodsException):
            kb.add_parametrized_kernels(invalid_kfunc, 'param', (1, ))

    for invalid_values in ('string', gen_random_sample, [], KernelBucket):
        with raises(ValueError):
            kb.add_parametrized_kernels(PolyKernel, 'param', invalid_values)

    for invalid_param in ('__param__', (), 'blahblah', 5):
        for ker_func in (PolyKernel, LaplacianKernel, GaussianKernel, SigmoidKernel):
            with raises(ValueError):
                kb.add_parametrized_kernels(ker_func, invalid_param, 2)


def test_ideal_kernel():

    ik = ideal_kernel(np.random.randint(1, 5, num_samples))
    if ik.size != num_samples**2:
        raise ValueError('ideal kernel size unexpected')

def test_correlation_km():

    corr_coef = correlation_km(np.random.rand(10,10), np.random.rand(10,10))
    if corr_coef > 1 or corr_coef < -1:
        raise ValueError('correlation out of bounds [-1, 1]')

def test_pairwise_similarity():

    ps = pairwise_similarity(kset)
    if ps.shape != (kset.size, kset.size):
        raise ValueError('invalid shape for pairwise_similarity computation')

