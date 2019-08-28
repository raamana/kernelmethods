"""

This is a simple comparison of kernelmethods to JKernelMachines and Weka.

Repeated holdout (80% train, 20% test) with 20 repetitions, on four UCI datasets

"""

from os.path import abspath, dirname, join as pjoin
from time import gmtime, strftime
from warnings import simplefilter

import numpy as np
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC

from kernelmethods.algorithms import KernelMachine
from kernelmethods.numeric_kernels import GaussianKernel
from kernelmethods.utils import _ensure_min_eps

simplefilter('ignore')

ds_dir = dirname(abspath(__file__))
ds_names = (
    "ionosphere_scale",
    "heart_scale",
    "breast-cancer_scale",
    "german.numer_scale",)

ds_paths = [pjoin(ds_dir, 'libsvm', name) for name in ds_names]


def sigma_from_gamma(gamma=0.1):
    return _ensure_min_eps(np.sqrt(1.0 / (2 * gamma)))


def gamma_from_sigma(sigma=0.1):
    return _ensure_min_eps(1.0 / (2 * sigma ** 2))


for name, ds_path in zip(ds_names, ds_paths):
    time_stamp = strftime("%H:%M:%S", gmtime())

    X, y = load_svmlight_file(ds_path)
    X = X.toarray()

    print('\n{:10}  {:20} {}'.format(time_stamp, name, X.shape))

    gamma = 0.1
    skl_svm = SVC(C=1.0, kernel='rbf', gamma=gamma)
    ss_cv1 = ShuffleSplit(n_splits=20, train_size=0.8, test_size=0.2)
    scores_skl = cross_val_score(skl_svm, X, y, cv=ss_cv1)

    ker_func = GaussianKernel(sigma=sigma_from_gamma(gamma))
    km_svm = KernelMachine(k_func=ker_func, learner_id='SVM', normalized=False)
    ss_cv2 = ShuffleSplit(n_splits=20, train_size=0.8, test_size=0.2)
    scores_km = cross_val_score(km_svm, X, y, cv=ss_cv2)

    print('\tSKLearn    Accuracy: {:.4f} +/- {:.4f}'
          ''.format(np.mean(scores_skl), np.std(scores_skl)))

    print('\tKM    SVM  Accuracy: {:.4f} +/- {:.4f}'
          ''.format(np.mean(scores_km), np.std(scores_km)))
