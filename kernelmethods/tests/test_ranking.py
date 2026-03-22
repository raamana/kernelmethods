
from kernelmethods.sampling import make_kernel_bucket
from kernelmethods.ranking import find_optimal_kernel, rank_kernels, \
    alignment_ranking, min_max_scale, CV_ranking, get_estimator
import numpy as np
from pytest import raises, warns

kb = make_kernel_bucket()
sample = np.random.rand(10, 4)
kb.attach_to(sample)

def test_misc():

    raises(TypeError, find_optimal_kernel, 'bucket', None, None)

    scores = rank_kernels(kb, np.ones(sample.shape[0]), method='align/corr')
    assert scores.ndim == 1
