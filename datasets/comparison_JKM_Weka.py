"""

This is a simple comparison of kernelmethods to JKernelMachines and Weka.

Repeated holdout (80% train, 20% test) with 20 repetitions, on four UCI datasets

"""

from os.path import abspath, dirname, join as pjoin

import numpy as np
# from time import gmtime, strftime
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC
from kernelmethods import KernelMachine, OptimalKernelSVR, OptimalKernelSVC
from kernelmethods.numeric_kernels import GaussianKernel
from warnings import simplefilter
simplefilter('ignore')

ds_dir = dirname(abspath(__file__))
ds_names = (
    "ionosphere_scale",
    "heart_scale",
    "breast-cancer_scale",
    "german.numer_scale",)

ds_paths = [pjoin(ds_dir, 'libsvm', name) for name in ds_names]

for name, ds_path in zip(ds_names, ds_paths):
    # time_stamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # print('Processing {} ... \t {}'.format(name, time_stamp))

    X, y = load_svmlight_file(ds_path)
    X = X.toarray()

    # skl_svm = SVC(C=1.0, kernel='rbf', gamma=0.1)
    # ss_cv = ShuffleSplit(n_splits=20, train_size=0.8, test_size=0.2)
    # scores_skl = cross_val_score(skl_svm, X, y, cv=ss_cv)
    #
    # ker_func = GaussianKernel(sigma=0.1)
    # km_svm = KernelMachine(k_func=ker_func, learner_id='SVM')
    # ss_cv = ShuffleSplit(n_splits=20, train_size=0.8, test_size=0.2)
    # scores_km = cross_val_score(skl_svm, X, y, cv=ss_cv)

    # print('\nDataset: {:20}\n\tSKLearn  Accuracy: {:.4f} +/- {:.4f}'
    #       ''.format(name, np.mean(scores_skl), np.std(scores_skl)))
    #
    # print('\tKM   SVM Accuracy: {:.4f} +/- {:.4f}'
    #       ''.format(np.mean(scores_km), np.std(scores_km)))

    ok_svm = OptimalKernelSVC(C=1.0, k_bucket='light')
    ss_cv = ShuffleSplit(n_splits=20, train_size=0.8, test_size=0.2)
    scores_oksvm = cross_val_score(ok_svm, X, y, cv=ss_cv)

    print('\tKM OKSVM Accuracy: {:.4f} +/- {:.4f}'
          ''.format(np.mean(scores_oksvm), np.std(scores_oksvm)))

    # time_stamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # print('Done processing {} ... \t {}'.format(name, time_stamp))
