import traceback
import warnings

import numpy as np
from kernelmethods.algorithms import KernelMachine, OptimalKernelSVR
from kernelmethods.config import KMNormError, KernelMethodsException, \
    KernelMethodsWarning, Chi2NegativeValuesException
from kernelmethods.numeric_kernels import DEFINED_KERNEL_FUNCS
from kernelmethods.sampling import make_kernel_bucket, KernelBucket
from pytest import raises
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

rnd = np.random.RandomState(0)
np.set_printoptions(precision=3, linewidth=120)

def gen_random_sample(num_samples, sample_dim):
    """To better control precision and type of floats"""

    # TODO input sparse arrays for test
    return np.random.rand(num_samples, sample_dim)


sample_dim = 5
n_training = 100
n_testing = 30


def _test_estimator_can_fit_predict(estimator, est_name):

    # fresh data for each call
    train_data, labels = make_classification(n_features=sample_dim,
                                             n_samples=n_training)
    test_data = gen_random_sample(n_testing, sample_dim)

    if hasattr(estimator, 'k_func') and 'chi2' in estimator.k_func.name:
        train_data = np.abs(train_data)
        test_data = np.abs(test_data)

    try:
        check_estimator(estimator)
    except (KMNormError, Chi2NegativeValuesException,
            KernelMethodsException, KernelMethodsWarning,
            RuntimeError) as kme:
        print('KernelMethodsException encountered during estimator checks - '
              'ignoring it!')
        traceback.print_exc()
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

    train_data, labels = make_classification(n_features=sample_dim, n_classes=2,
                                             n_samples=n_training)
    test_data = gen_random_sample(n_testing, sample_dim)

    # creating the smallest bucket, just with linear kernel, to speed up tests
    kb = KernelBucket(poly_degree_values=None,
                      rbf_sigma_values=None,
                      laplace_gamma_values=None)
    try:
        OKSVR = OptimalKernelSVR(k_bucket=kb)
    except:
        raise RuntimeError('Unable to instantiate OptimalKernelSVR!')

    _test_estimator_can_fit_predict(OKSVR, 'OptimalKernelSVR')

    for invalid_value in (np.random.randint(10), 10.1, ('tuple')):
        with raises(ValueError):
            OKSVR = OptimalKernelSVR(k_bucket=invalid_value)
            OKSVR.fit(train_data, labels)

    OKSVR = OptimalKernelSVR(k_bucket=kb)
    OKSVR.set_params(k_bucket=kb)


def test_kernel_machine():
    for kernel in DEFINED_KERNEL_FUNCS:
        # print('\n\nTesting {}'.format(kernel))
        try:
            k_machine = KernelMachine(kernel)
        except:
            raise RuntimeError('Unable to instantiate KernelMachine with this func '
                               '{}!'.format(kernel))

        print('\n{}'.format(k_machine))
        _test_estimator_can_fit_predict(k_machine,
                                        'kernel machine with ' + str(kernel))


test_optimal_kernel_svr()
