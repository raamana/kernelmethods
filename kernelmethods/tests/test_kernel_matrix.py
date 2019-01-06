

import numpy as np
from scipy.sparse import issparse
from scipy.linalg import eigh
from pytest import raises
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel
from kernelmethods.base import KernelMatrix, KMAccessError, KernelMatrixException
from kernelmethods.operations import is_PSD

num_samples = 50 # 9
sample_dim = 3 # 2
target_label_set = [1, 2]

sample_data = np.random.rand(num_samples, sample_dim)
target_labels = np.random.choice(target_label_set, num_samples)

km = KernelMatrix(PolyKernel(degree=2, skip_input_checks=True))

km.attach_to(sample_data)
km_dense = km.full # this will force computation of full KM

km.center()

max_num_elements = max_num_ker_eval = num_samples * (num_samples + 1) / 2

def test_symmetry():

    if not np.isclose(km_dense, km_dense.T).all():
        print('KM not symmetric')

def test_PSD():

    if not is_PSD(km_dense):
        raise ValueError('this kernel matrix is not PSD!')

def test_get_item():

    for invalid_index in [-1, num_samples+1]:
        # out of range indices must raise an error on any dim
        with raises(KMAccessError):
            print(km[invalid_index,:])
        with raises(KMAccessError):
            print(km[:, invalid_index])

    # max 2 dims allowed for access
    # TODO no restriction on float: float indices will be rounded down towards 0
    # (1.0, 2), (1, 3.5) are valid at the moment
    for invalid_access in [(2  , 4, 5), (5,),
                           ('1', 1), (2, 'efd'),
                           ( ((0, 1), 2), (3, 4)), # no tuple of tuples for a single dim
                           ]:
        with raises((KMAccessError, TypeError)):
            print(km[invalid_access])


def test_random_submatrix_access():

    # for trial in range(10):

    subset_len1 = np.random.choice(np.arange(num_samples - 1) + 1, 2)
    subset_len2 = np.random.choice(np.arange(num_samples - 1) + 1, 2)
    subset_len1.sort()
    subset_len2.sort()

    if subset_len1[0]==subset_len1[1]:
        subset_len1[1] = subset_len1[0] + 1

    if subset_len2[0]==subset_len2[1]:
        subset_len2[1] = subset_len2[0] + 1

    sub_matrix = km[subset_len1[0]:subset_len1[1], subset_len2[0]:subset_len2[1]]
    if not sub_matrix.shape == (subset_len1[1]-subset_len1[0],
                                subset_len2[1]-subset_len2[0]):
        raise ValueError('error in KM access implementation')

def test_diag():

    if len(km.diag) != num_samples:
        raise ValueError('KM diagonal does not have N elements!')

def test_sparsity():

    # reset!
    km.attach_to(sample_data)
    if not issparse(km.full_sparse):
        raise TypeError('error in sparse format access of KM : it is not sparse')

    if issparse(km.full):
        raise TypeError('error in dense format access of KM : it is sparse!')

def test_reset_flags_on_new_attach():

    km.attach_to(sample_data)
    if km._populated_fully:
        raise ValueError('flag _populated_fully not set to False upon reset')
    if km._lower_tri_km_filled:
        raise ValueError('flag _lower_tri_km_filled not set to False upon reset')
    if km._num_ker_eval > 0:
        raise ValueError('counter _num_ker_eval > 0 upon reset!')
    if hasattr(km, '_full_km'):
        raise ValueError('_full_km from previous run is not cleared!')
    if len(km._KM) > 0:
        raise ValueError('internal dict not empty upon reset!')

def test_internal_flags_on_recompute():

    km.attach_to(sample_data) # reset first
    new_dense = km.full # recompute
    if not km._populated_fully:
        raise ValueError('flag _populated_fully not set to True upon recompute')
    if km._num_ker_eval != max_num_ker_eval:
        raise ValueError('unexpected value for counter _num_ker_eval upon recompute!')
    if not hasattr(km, '_full_km'):
        raise ValueError('_full_km is not populated yet!')
    if len(km._KM)!=max_num_elements:
        raise ValueError('internal dict not empty upon recompute!')
    if not km._lower_tri_km_filled:
        raise ValueError('flag _lower_tri_km_filled not set to True '
                         'upon recompute with fill_lower_tri=True')

test_PSD()
