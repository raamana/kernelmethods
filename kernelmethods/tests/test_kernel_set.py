
import numpy as np
from scipy.sparse import issparse
from scipy.linalg import eigh
from pytest import raises
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel
from kernelmethods.base import KernelMatrix, KernelSet, \
    SumKernel, ProductKernel, AverageKernel, \
    KMAccessError, KernelMatrixException
from kernelmethods.operations import is_PSD
from kernelmethods.sampling import KernelBucket, pairwise_similarity
from kernelmethods.operations import alignment_centered, center_km
from pytest import raises

num_samples = 50 # 9
sample_dim = 3 # 2
target_label_set = [1, 2]

sample_data = np.random.rand(num_samples, sample_dim)
target_labels = np.random.choice(target_label_set, (num_samples, 1))

IdealKM = target_labels.dot(target_labels.T)

rbf = KernelMatrix(GaussianKernel(sigma=10, skip_input_checks=True))
lin = KernelMatrix(LinearKernel(skip_input_checks=True))
poly = KernelMatrix(PolyKernel(degree=2, skip_input_checks=True))

# lin.attach_to(sample_data)
# rbf.attach_to(sample_data)
# poly.attach_to(sample_data)

kset = KernelSet([lin, poly, rbf])
print(kset)

def test_size():

    assert kset.size == 3
    assert len(kset) == 3

def test_get_item():
    """access by index"""

    for invalid_index in [-1, kset.size]:
        with raises(IndexError):
            print(kset[invalid_index])

    for invalid_index in [-1.0, '1']:
        with raises(ValueError):
            print(kset[invalid_index])


def test_take():
    """access by index"""

    for invalid_index in [-1, kset.size]:
        with raises(IndexError):
            print(kset.take([invalid_index]))

    k2 = kset.take([0, 1])
    assert isinstance(k2, KernelSet)
    assert k2.size == 2



# kb = KernelBucket()
# # this attach is necessary for anything useful! :)
# kb.attach_to(sample_data)
#
# print('Alignment to Ideal Kernel:')
# ag = np.zeros(kb.size)
# for ix, km in enumerate(kb):
#     ag[ix] = alignment_centered(km.full, IdealKM)
#     print('{:4} {:>60} : {:10.5f}'.format(ix, str(km),ag[ix]))

test_take()
