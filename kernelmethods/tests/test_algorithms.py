import numpy as np
from numbers import Number
from pytest import raises, warns
from hypothesis import given, strategies, unlimited
from hypothesis import settings as hyp_settings
from hypothesis import HealthCheck
import warnings
import traceback
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator
from kernelmethods.base import KernelMatrix
from kernelmethods.config import KMNormError, KernelMethodsException
from kernelmethods.operations import is_positive_semidefinite
from kernelmethods.sampling import KernelBucket, make_kernel_bucket

from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LinearKernel, \
    LaplacianKernel
from kernelmethods.algorithms import OptimalKernelSVR, KernelMachine
from traceback import print_exc

rnd = np.random.RandomState(0)

SupportedKernels = (GaussianKernel(),
                    PolyKernel(), LinearKernel(),
                    LaplacianKernel())

def gen_random_sample(num_samples, sample_dim):
    """To better control precision and type of floats"""

    # TODO input sparse arrays for test
    return np.random.rand(num_samples, sample_dim)

sample_dim = 10
n_training = 100
n_testing  = 30

def _test_estimator_can_fit_predict(estimator, est_name):

    train_data, labels = make_classification(n_features=sample_dim,
                                             n_samples=n_training)
    test_data = gen_random_sample(n_testing, sample_dim)

    try:
        check_estimator(estimator)
    except (KMNormError, KernelMethodsException, RuntimeError) as kme:
        print('KernelMethodsException encountered during estimator checks - '
              'ignoring it!')
        # traceback.print_exc()
        pass
    except:
        traceback.print_exc()
        raise TypeError('{} failed sklearn checks to be an estimator'
                        ''.format(est_name))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(train_data, labels)
    except:
        raise RuntimeError('{} is unable to fit to training data!'.format(est_name))

    try:
        estimator.predict(test_data)
    except:
        raise RuntimeError('{} is unable to make predictions'.format(est_name))


def test_optimal_kernel_svr():

    k_bucket = make_kernel_bucket('light')

    try:
        OKSVR = OptimalKernelSVR(k_bucket=k_bucket)
    except:
        raise RuntimeError('Unable to instantiate OptimalKernelSVR!')

    _test_estimator_can_fit_predict(OKSVR, 'OptimalKernelSVR')



def test_kernel_machine():

    for kernel in SupportedKernels:
        # print('\n\nTesting {}'.format(kernel))
        try:
            k_machine = KernelMachine(kernel)
        except:
            raise RuntimeError('Unable to instantiate KernelMachine with this func '
                               '{}!'.format(kernel))

        print('\n{}'.format(k_machine))
        _test_estimator_can_fit_predict(k_machine,
                                        'kernel machine with ' + str(kernel))


test_kernel_machine()
