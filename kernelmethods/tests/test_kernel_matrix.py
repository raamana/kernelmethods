

import numpy as np
np.set_printoptions(linewidth=120, precision=4)
from scipy.sparse import issparse
from scipy.linalg import eigh
from pytest import raises
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel, \
    DEFINED_KERNEL_FUNCS
from kernelmethods import KernelMatrix, KMAccessError, KernelMethodsException
from kernelmethods.base import ConstantKernelMatrix
from kernelmethods.operations import is_PSD

num_samples = np.random.randint(30, 100)
sample_dim = np.random.randint(3, 10) # 2
target_label_set = [1, 2]

num_samples_two = np.random.randint(30, 100)
sample_two_dim = sample_dim

sample_data = np.random.rand(num_samples, sample_dim)
target_labels = np.random.choice(target_label_set, num_samples)

poly = PolyKernel(degree=2, skip_input_checks=True)
# suffix 1 to indicate one sample case
km1 = KernelMatrix(poly)
km1.attach_to(sample_data)

max_num_elements = max_num_ker_eval = num_samples * (num_samples + 1) / 2

def test_symmetry():

    if not np.isclose(km1.full, km1.full.T).all():
        print('KM not symmetric')

def test_PSD():

    if not is_PSD(km1.full):
        raise ValueError('this kernel matrix is not PSD!')

def test_normalization():

    km1.normalize(method='cosine')
    if not hasattr(km1, 'normed_km'):
        raise ValueError('Attribute exposing normalized km does not exist!')

    if not np.isclose(km1.normed_km.diagonal(), 1.0).all():
        raise ValueError('One or more diagonal elements of normalized KM != 1.0:\n\t'
                         '{}'.format(km1.normed_km.diagonal()))

    km2 = KernelMatrix(poly)
    km2.attach_to(sample_data)
    normed_km = km2.normed_km
    assert normed_km.shape == km2.shape

    frob = km1.frob_norm
    assert np.isreal(frob)

    # during init
    with raises(TypeError):
        _ = KernelMatrix(poly, normalized='True')

def test_centering():

    km2 = KernelMatrix(poly)
    km2.attach_to(sample_data)
    assert km2.centered.shape == km2.shape

def test_get_item():

    for invalid_index in [-1, num_samples+1]:
        # out of range indices must raise an error on any dim
        with raises(KMAccessError):
            print(km1[invalid_index, :])
        with raises(KMAccessError):
            print(km1[:, invalid_index])

    # max 2 dims allowed for access
    # TODO no restriction on float: float indices will be rounded down towards 0
    # (1.0, 2), (1, 3.5) are valid at the moment
    for invalid_access in [(2  , 4, 5), (5,),
                           ('1', 1), (2, 'efd'),
                           ( ((0, 1), 2), (3, 4)), # no tuple of tuples for a single dim
                           ]:
        with raises((KMAccessError, TypeError)):
            print(km1[invalid_access])

    with raises(KMAccessError):
        km1[1, 2, 3] # no 3-dim access

    with raises(KMAccessError):
        km1[1, 2, 3, 4] # no 4-dim access either

    # selection must result in valid indices
    with raises(KMAccessError):
        km1[0,km1.size+5]

    with raises(KMAccessError):
        km1[km1.size + 5, 0]

    # linear indexing is now allowed
    for valid_index in np.random.randint(0, km1.size, 5):
        _ = km1[valid_index]

    # as well as vectorized/colon
    _ = km1[:,0]
    _ = km1[0, :]


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

    sub_matrix = km1[subset_len1[0]:subset_len1[1], subset_len2[0]:subset_len2[1]]
    if not sub_matrix.shape == (subset_len1[1]-subset_len1[0],
                                subset_len2[1]-subset_len2[0]):
        raise ValueError('error in KM access implementation')

def test_size_properties():

    if len(km1.diagonal()) != num_samples:
        raise ValueError('KM diagonal does not have N elements!')

    if km1.size != num_samples**2:
        raise ValueError('KM size does not match N^2, N=num_samples')

    if km1.size != km1.num_samples**2:
        raise ValueError('KM size does not match N^2, invalid internal representation!')

def test_sparsity():

    km = KernelMatrix(poly, normalized=False)
    km.attach_to(sample_data)
    # when normalized=True, full KM won't be sparse!
    if not km._keep_normed and not issparse(km.full_sparse):
        raise TypeError('error in sparse format access of KM : it is not sparse')

    if issparse(km1.full):
        raise TypeError('error in dense format access of KM : it is sparse!')

def test_reset_flags_on_new_attach():

    km1.attach_to(sample_data)
    if km1._populated_fully:
        raise ValueError('flag _populated_fully not set to False upon reset')
    if km1._lower_tri_km_filled:
        raise ValueError('flag _lower_tri_km_filled not set to False upon reset')
    if km1._num_ker_eval > 0:
        raise ValueError('counter _num_ker_eval > 0 upon reset!')
    if hasattr(km1, '_full_km'):
        raise ValueError('_full_km from previous run is not cleared!')
    if len(km1._KM) > 0:
        raise ValueError('internal dict not empty upon reset!')

def test_internal_flags_on_recompute():

    km1.attach_to(sample_data) # reset first
    new_dense = km1.full # recompute
    if not km1._populated_fully:
        raise ValueError('flag _populated_fully not set to True upon recompute')
    if km1._num_ker_eval != max_num_ker_eval:
        raise ValueError('unexpected value for counter _num_ker_eval upon recompute!')
    if not hasattr(km1, '_full_km'):
        raise ValueError('_full_km is not populated yet!')
    if len(km1._KM)!=max_num_elements:
        raise ValueError('internal dict not empty upon recompute!')
    if not km1._lower_tri_km_filled:
        raise ValueError('flag _lower_tri_km_filled not set to True '
                         'upon recompute with fill_lower_tri=True')

def test_attach_to_two_samples():
    """
    Behaviour of KM when attached to two samples.

    0. it is not necessarily symmetric

    """

    sample_two = np.random.rand(num_samples_two, sample_two_dim)
    targets_two = np.random.choice(target_label_set, num_samples_two)

    for kernel in DEFINED_KERNEL_FUNCS:
        km2 = KernelMatrix(kernel=kernel, normalized=False)
        km2.attach_to(sample_data, name_one='S1', sample_two=sample_two, name_two='S2')
        km2_dense = km2.full  # this will force computation of full KM

        rand_ix_one = np.random.choice(range(num_samples), 5)
        rand_ix_two = np.random.choice(range(num_samples_two), 5)
        for ix_one, ix_two in zip(rand_ix_one, rand_ix_two):
            external_eval = kernel(sample_data[ix_one,:], sample_two[ix_two,:])
            if not np.isclose(km2[ix_one, ix_two], external_eval):
                raise ValueError('Invalid implementation in two sample case:'
                                 '\n\tcomputed values do not match external evaluation!'
                                 '\n\t for {}'.format(kernel))

    if km2.size != num_samples*num_samples_two:
        raise ValueError('KM size does not match N1*N2, N=num_samples for dataset i')

    if km2.size != np.prod(km2.num_samples):
        raise ValueError('KM size does not match N1*N2, invalid internal representation!')

    with raises(NotImplementedError):
        km2.center()

    with raises(KMAccessError):
        km2.centered

    with raises((KMAccessError, NotImplementedError)):
        km2.diagonal()

    with raises(ValueError):
        # dimensionalities can not differ!
        more_dims = np.hstack((sample_data, sample_data[:,:1]))
        km2.attach_to(sample_data, sample_two=more_dims)


def test_attributes():

    km = KernelMatrix(LinearKernel())
    km.set_attr('name', 'linear')
    assert km.get_attr('name') == 'linear'
    assert km.get_attr('noname', '404') == '404'
    km.set_attr('weight', 42)

    kma = km.attributes()
    for attr in ('name', 'weight'):
        assert attr in kma


def test_constant_km():

    rand_val = np.random.random()
    rand_size = np.random.randint(50)

    const = ConstantKernelMatrix(num_samples=rand_size,
                                 value=rand_val)
    # trying name param also
    const = ConstantKernelMatrix(num_samples=rand_size,
                                 value=rand_val, name=None)

    assert const.num_samples == rand_size == const.size
    assert len(const) == rand_size
    assert const.shape == (rand_size, rand_size)

    for _ in range(min(5, rand_size)):
        indices = np.random.randint(0, rand_size, 2)
        assert all(const[indices[0], indices[1]] == rand_val)

    for invalid_index in ('index', ':',
                          [np.Inf, ], [ 1,-rand_size-2],
                          [], [None, 2]):
        with raises(KMAccessError):
            const[invalid_index]

    # there must be a single unique value in the matrix or diagonal
    assert np.isclose(np.unique(const.full), rand_val).all()
    assert np.isclose(np.unique(const.diag), rand_val).all()

    expected = np.full((rand_size, rand_size), fill_value=rand_val)
    assert np.isclose(const.full, expected).all()


# test_attributes()
# test_constant_km()
test_get_item()
