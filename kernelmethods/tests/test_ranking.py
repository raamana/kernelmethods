
from kernelmethods.sampling import make_kernel_bucket
from kernelmethods.ranking import find_optimal_kernel, rank_kernels, \
    alignment_ranking, min_max_scale, CV_ranking, get_estimator
import numpy as np
from pytest import raises, warns

kb = make_kernel_bucket()

def test_misc():

    raises(TypeError, find_optimal_kernel, 'bucket', None, None)

    with raises(NotImplementedError):
        rank_kernels(kb, None, method='align/corr')

