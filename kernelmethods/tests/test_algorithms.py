import numpy as np
from numbers import Number
from pytest import raises, warns
from hypothesis import given, strategies, unlimited
from hypothesis import settings as hyp_settings
from hypothesis import HealthCheck
from sklearn.datasets import make_classification
from kernelmethods.base import KernelMatrix
from kernelmethods.operations import is_positive_semidefinite
from kernelmethods.sampling import KernelBucket, make_kernel_bucket
from kernelmethods.algorithms import OptimalKernelSVR
import warnings

def gen_random_sample(num_samples, sample_dim):
    """To better control precision and type of floats"""

    # TODO input sparse arrays for test
    return np.random.rand(num_samples, sample_dim)

sample_dim = 10

def test_optimal_kernel_svr():

    train_data, labels = make_classification(n_features=sample_dim)
    test_data = gen_random_sample(30, sample_dim)
    k_bucket = make_kernel_bucket('light')

    try:
        OKSVR = OptimalKernelSVR(k_bucket)
    except:
        raise RuntimeError('Unable to instantiate OptimalKernelSVR!')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            OKSVR.fit(train_data, labels)
    except:
        raise RuntimeError('Unable to fit OptimalKernelSVR to training data!')

    try:
        OKSVR.predict(test_data)
    except:
        raise RuntimeError('Unable to make predictions with OptimalKernelSVR!')


test_optimal_kernel_svr()
