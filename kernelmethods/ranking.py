"""

Module gathering techniques and helpers to rank kernels using various methods and
metrics, including based on their target alignment, performance in
cross-validation etc.

"""

import numpy as np
from kernelmethods.sampling import make_kernel_bucket
from kernelmethods.utils import min_max_scale


def find_optimal_kernel(kernel_bucket, sample, targets, method='align/corr'):
    """Finds the optimal kernel for the current sample"""

    KB = make_kernel_bucket(kernel_bucket)
    KB.attach_to(sample=sample)
    metric = rank_kernels(kernel_bucket, targets, method=method)

    return KB[np.argmax(metric)]


def rank_kernels(kernel_bucket, targets, method='align/corr', **method_params):
    """
    Main interface.

    Choices for the method include: "align/corr", "cv_risk"

    """

    ranking_methods = ("align/corr", "cv_risk")
    method = method.lower()

    if method in ("align/corr",):
        return alignment_ranking(kernel_bucket, targets, **method_params)
    elif method in ('cv_risk', 'cv'):
        return CV_ranking(kernel_bucket, targets, **method_params)
    else:
        raise NotImplementedError('Choose one of {}'.format(ranking_methods))


def CV_ranking(kernel_bucket, targets, num_folds=3, estimator_name='SVM'):
    """Ranks kernels by their performance in cross-validation."""

    from sklearn.model_selection import GridSearchCV

    cv_scores = list()
    for km in kernel_bucket:
        estimator, param_grid = get_estimator(estimator_name)
        gs = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=num_folds)
        gs.fit(km.full, targets)
        cv_scores.append(gs.best_score_)

    return 100 * min_max_scale(cv_scores) #scaling helps compare across multiple metrics


def alignment_ranking(kernel_bucket, targets, **method_params):
    """Method to rank kernels that depend on target alignment."""

    raise NotImplementedError()


def get_estimator(learner_id='svm'):
    """Returns a valid kernel machine to become the base learner of the MKL methods.

    This base learner must be able to accept a precomputed kernel for fit/predict methods!

    TODO hyper-param optimization needs to be incorporated somewhere!!
        Perhaps by returning a GridSearchCV(base_learner) object or similar?
    """

    learner_id = learner_id.lower()
    if learner_id in ('svm', 'svc'):
        from sklearn.svm import SVC
        range_C = np.power(10.0, range(-6, 6))
        param_grid = dict(C=range_C)
        base_learner = SVC(kernel='precomputed', probability=True, C=10)

    else:
        raise NotImplementedError('Requested base learner {} is not implemented yet!'
                                  ''.format(learner_id))

    return base_learner, param_grid
