
from kernelmethods.operations import center_km, frobenius_product, frobenius_norm
import numpy as np
from scipy.sparse import issparse
from scipy.linalg import eigh

from pytest import raises

num_samples = 50 # 9
sample_dim = 3 # 2
target_label_set = [1, 2]

sample_data = np.random.rand(num_samples, sample_dim)
target_labels = np.random.choice(target_label_set, (num_samples, 1))

A = np.random.rand(4, 4)
B = np.random.rand(4, 4)



def test_frobenius_product():

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[4, 1], [2, 5]])

    fprod = frobenius_product(A, B)
    if not np.isclose(fprod, 32):
        raise ValueError('Frobenius product implementation is wrong!')

    fnorm = frobenius_norm(A)
    if not np.isclose(fnorm, np.sqrt(frobenius_product(A, A))):
        raise ValueError('Frobenius norm implementation is wrong!')
