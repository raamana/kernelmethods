
import numpy as np
from pytest import raises

from kernelmethods.base import KMSetAdditionError, KernelMatrix, KernelSet, \
    BaseKernelFunction
from kernelmethods.numeric_kernels import GaussianKernel, LinearKernel, PolyKernel


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

def test_creation():

    try:
        ks = KernelSet()
    except:
        raise SyntaxError('empty set creation failed.')


def test_size_property_mismatch():

    ks = KernelSet(num_samples=sample_data.shape[0]+1)
    lin = KernelMatrix(LinearKernel(skip_input_checks=True))
    lin.attach_to(sample_data)
    with raises(KMSetAdditionError):
        ks.append(lin)


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


def test_get_ker_funcs():

    for index in (0, 1):
        kf_list = kset.get_kernel_funcs([index, ])
        for kf in kf_list:
            if not isinstance(kf, BaseKernelFunction):
                raise TypeError('get_kernel_funcs not returning proper output type')

def test_take():
    """access by index"""

    for invalid_index in [-1, kset.size]:
        with raises(IndexError):
            print(kset.take([invalid_index]))

    k2 = kset.take([0, 1])
    assert isinstance(k2, KernelSet)
    assert k2.size == 2

def test_extend():

    kset1 = KernelSet([poly, rbf, lin])
    kset2 = KernelSet([poly, rbf])
    kset1.extend(kset2)

    if kset1.size != 5:
        raise ValueError('KernelSet.extend() failed')

    with raises(KMSetAdditionError):
        kset1.extend(['blah', ])

    with raises(KMSetAdditionError):
        k4_diff_size = KernelSet(num_samples=kset.size+1)
        kset1.extend(k4_diff_size)


def test_attributes():

    kset.set_attr('name', 'linear')
    for km in kset:
        assert km.get_attr('name') == 'linear'
        assert km.get_attr('noname', '404') == '404'

    values = np.random.rand(kset.size)
    kset.set_attr('weight', values)
    for ii, km in enumerate(kset):
        assert km.get_attr('weight') == values[ii]


# kb = KernelBucket()
# # this attach is necessary for anything useful! :)
# kb.attach_to(sample_data)
#
# print('Alignment to Ideal Kernel:')
# ag = np.zeros(kb.size)
# for ix, km in enumerate(kb):
#     ag[ix] = alignment_centered(km.full, IdealKM)
#     print('{:4} {:>60} : {:10.5f}'.format(ix, str(km),ag[ix]))

test_attributes()
