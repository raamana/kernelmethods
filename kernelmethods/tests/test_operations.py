
from kernelmethods.operations import center_km, frobenius_product, frobenius_norm, \
    normalize_km, normalize_km_2sample, alignment_centered, linear_combination, \
    is_positive_semidefinite, is_PSD
from kernelmethods.config import KMNormError
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel, \
    LaplacianKernel
from kernelmethods.utils import check_callable
from kernelmethods.base import KernelMatrix, KernelFromCallable, \
    BaseKernelFunction
from kernelmethods.sampling import make_kernel_bucket

import numpy as np
from scipy.sparse import issparse
from scipy.linalg import eigh

from pytest import raises, warns

num_samples = np.random.randint(20, 50)
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


def test_psd():

    with raises(TypeError):
        is_PSD([2, 34, 23])

    if is_PSD(np.random.rand(2, 4)):
        raise ValueError('Non-square matrix is being deemed PSD!!! Big error!')

    if is_PSD(np.random.rand(5, 5)):
        raise ValueError('Non-symmetric matrix is being deemed PSD!!! Big error!')

    negative_semi_def_matrix = np.array([[-1, 0], [0, -1]])
    if is_PSD(negative_semi_def_matrix):
        raise ValueError('Implementation for PSD check failed. '
                         'negative_semi_def_matrix is approved as PSD.')

    # from scipy.linalg import LinAlgError
    # TODO find a matrix that produces LinAlgError
    # bad_matrix = {}
    # raises(LinAlgError, is_PSD, bad_matrix)


def test_frobenius_product():

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[4, 1], [2, 5]])
    C = np.array([[10, 2, 5], [6, 8, 6]])

    fprod = frobenius_product(A, B)
    if not np.isclose(fprod, 32):
        raise ValueError('Frobenius product implementation is wrong!')

    with raises(ValueError):
        frobenius_product(B, C)

    fnorm = frobenius_norm(A)
    if not np.isclose(fnorm, np.sqrt(frobenius_product(A, A))):
        raise ValueError('Frobenius norm implementation is wrong!')

def test_centering():

    with raises(ValueError):
        center_km(np.full((3,4), 1))

    with raises(ValueError):
        center_km([])

    kmc = center_km(np.random.rand(10,10))


def test_normalize():
    with raises(ValueError):
        normalize_km(np.full((3, 4), 1))

    with raises(KMNormError):
        normalize_km(np.zeros((5,5)))

    kmc = normalize_km(np.random.randn(10, 10))

def test_normalize_two_sample():

    num_samples_one = 3
    num_samples_two = 4
    with raises(ValueError):
        normalize_km_2sample(np.random.randn(num_samples_one, num_samples_two),
                     np.random.randn(num_samples_two+1, 1), [])

    with raises(ValueError):
        normalize_km_2sample(np.random.randn(num_samples_one, num_samples_two),
                     np.random.randn(num_samples_one, 1),
                     np.random.randn(num_samples_two-1, 1), )

    with raises((KMNormError, RuntimeError)):
        normalize_km_2sample(np.zeros((5,5)), np.zeros((5,1)), np.zeros((5,1)))

    kmc = normalize_km(np.random.randn(10, 10))


def test_alignment_centered():

    km1 = KernelMatrix(kernel=LinearKernel())
    km1.attach_to(gen_random_sample(num_samples, sample_dim))

    km2 = KernelMatrix(kernel=LinearKernel())
    km2.attach_to(gen_random_sample(num_samples, sample_dim))

    km3_bad_size = KernelMatrix(kernel=LinearKernel())
    km3_bad_size.attach_to(gen_random_sample(num_samples+2, sample_dim))

    with raises(ValueError):
        alignment_centered(km1.full, km3_bad_size.full)

    # bad type : must be ndarray
    with raises(TypeError):
        alignment_centered(km1, km2.full)

    # bad type : must be ndarray
    with raises(TypeError):
        alignment_centered(km1.full, km2)

    for flag in (True, False):
        _ = alignment_centered(km1.full, km2.full, centered_already=flag)

    with raises(ValueError):
        _ = alignment_centered(np.zeros((10, 10)), np.random.randn(10, 10),
                               value_if_zero_division='raise')

    return_val_requested = 'random_set_value'
    with warns(UserWarning):
        ret_value = alignment_centered(np.random.randn(10, 10),
                               np.zeros((10, 10)),
                               value_if_zero_division=return_val_requested)
    if ret_value != return_val_requested:
        raise ValueError('Not returning the value requested in case of error!')


def test_linear_comb():

    kset = make_kernel_bucket('light')
    weights = np.random.randn(kset.size)
    kset.attach_to(sample_data)
    lc = linear_combination(kset, weights)

    with raises(ValueError):
        lc = linear_combination(kset, np.random.randn(kset.size+1))


test_psd()
