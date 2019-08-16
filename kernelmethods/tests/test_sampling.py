
from kernelmethods.operations import center_km, frobenius_product, frobenius_norm, \
    normalize_km, normalize_km_2sample, alignment_centered, linear_combination
from kernelmethods.config import KMNormError
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel, \
    LaplacianKernel
from kernelmethods.utils import check_callable
from kernelmethods.base import KernelMatrix, KernelFromCallable, \
    BaseKernelFunction
from kernelmethods.sampling import make_kernel_bucket, ideal_kernel, \
    pairwise_similarity, correlation_km

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

