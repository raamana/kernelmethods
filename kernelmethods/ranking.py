"""

Module gathering techniques and helpers to rank kernels using various methods and
metrics, such as

 - their target alignment,
 - performance in cross-validation

"""

import numpy as np
from kernelmethods import config as cfg
from kernelmethods.sampling import KernelBucket
from kernelmethods.utils import min_max_scale


def find_optimal_kernel(kernel_bucket, sample, targets, method='align/corr',
                        **method_params):
    """
    Finds the optimal kernel for the current sample given their labels.

    Parameters
    ----------
    kernel_bucket : KernelBucket
        The collection of kernels to evaluate and rank

    sample : ndarray
        The dataset given kernel bucket to be evaluated on

    targets : ndarray
        Target labels for each point in the sample dataset

    method : str
        identifier for the metric to choose to rank the kernels

    Returns
    -------
    km : KernelMatrix
        Instance of KernelMatrix with the optimal kernel function

    """

    if not isinstance(kernel_bucket, KernelBucket):
        raise TypeError('Input is not of required type: KernelBucket')

    method = method.lower()
    if method not in cfg.VALID_RANKING_METHODS:
        raise NotImplementedError('Ranking method not recognized. Choose one of {}'
                                  ''.format(cfg.VALID_RANKING_METHODS))

    kernel_bucket.attach_to(sample=sample)
    metric = rank_kernels(kernel_bucket, targets, method=method, **method_params)

    return kernel_bucket[np.argmax(metric)]


def rank_kernels(kernel_bucket, targets, method='align/corr', **method_params):
    """
    Computes a given ranking metric for all the kernel matrices in the bucket.

    Choices for the method include: "align/corr", "cv_risk"

    Parameters
    ----------
    kernel_bucket : KernelBucket

    targets : Iterable
        target values of the sample attached to the bucket

    method : str
        Identifies one of the metrics: ``align/corr``, ``cv_risk``

    method_params : dict
        Additional parameters to be passed on to the method chosen above.

    Returns
    -------
    scores : ndarray
        Values of the ranking metrics computed for the kernel matrices in the bucket

    """

    method = method.lower()
    if method not in cfg.VALID_RANKING_METHODS:
        raise NotImplementedError('Ranking method not recognized. Choose one of {}'
                                  ''.format(cfg.VALID_RANKING_METHODS))

    if method in ("align/corr",):
        return alignment_ranking(kernel_bucket, targets, **method_params)
    elif method in ('cv_risk', 'cv'):
        return CV_ranking(kernel_bucket, targets, **method_params)


def CV_ranking(kernel_bucket, targets, num_folds=3, estimator_name='SVM'):
    """
    Ranks kernels by their performance measured via cross-validation (CV).

    Parameters
    ----------
    kernel_bucket : KernelBucket

    targets : Iterable
        target values of the sample attached to the bucket

    num_folds : int
        Number of folds for the CV to be employed

    estimator_name : str
        Name of a valid Scikit-Learn estimator. Default: ``SVM``

    Returns
    -------
    scores : ndarray
        CV performance computed for the kernel matrices in the bucket

    """

    from sklearn.model_selection import GridSearchCV

    cv_scores = list()
    for km in kernel_bucket:
        estimator, param_grid = get_estimator(estimator_name)
        gs = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=num_folds)
        gs.fit(km.full, targets)
        cv_scores.append(gs.best_score_)

    # scaling helps compare across multiple metrics
    return 100 * min_max_scale(cv_scores)


def alignment_ranking(kernel_bucket, targets, **method_params):
    """Method to rank kernels that depend on target alignment.

    .. note:

        To be implemented.

    """

    raise NotImplementedError()


def get_estimator(learner_id='svm'):
    """
    Returns a valid kernel machine to become the base learner of the MKL methods.

    Base learner must be able to accept a precomputed kernel for fit/predict methods!

    Parameters
    ----------
    learner_id : str
        Identifier for the estimator to be chosen.
        Options: ``SVM`` and ``SVR``.
        Default: ``SVM``

    Returns
    -------
    base_learner : Estimator
        An sklearn estimator

    param_grid : dict
        Parameter grid (sklearn format) for the chosen estimator.

    """

    # TODO hyper-param optimization needs to be incorporated somewhere!!
    #   Perhaps by returning a GridSearchCV(base_learner) object or similar?

    learner_id = learner_id.lower()
    if learner_id in ('svm', 'svc'):
        from sklearn.svm import SVC
        range_C = np.power(10.0, range(-6, 6))
        param_grid = dict(C=range_C)
        base_learner = SVC(kernel='precomputed', probability=True, C=10)
    elif learner_id in ('svr',):
        from sklearn.svm import SVR
        range_C = np.power(10.0, range(-6, 6))
        param_grid = dict(C=range_C)
        base_learner = SVR(kernel='precomputed', C=10)
    else:
        raise NotImplementedError('Requested base learner {} is not implemented yet!'
                                  ''.format(learner_id))

    return base_learner, param_grid
