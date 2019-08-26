"""

Module to gather various high-level algorithms based on the kernel methods,
    such as kernel-based predictive models for classification and regression.

"""

from copy import deepcopy

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.svm import SVR, SVC
from sklearn.utils.validation import check_X_y, check_array

from kernelmethods import config as cfg
from kernelmethods.base import KernelMatrix
from kernelmethods.ranking import find_optimal_kernel, get_estimator
from kernelmethods.sampling import KernelBucket, make_kernel_bucket


class KernelMachine(BaseEstimator):
    """Generic class to return a drop-in sklearn estimator.

    Parameters
    ----------
    k_func : KernelFunction
        The kernel function the kernel machine bases itself on

    learner_id : str
        Identifier for the estimator to be built based on the kernel function.
        Options: ``SVM`` and ``SVR``.
        Default: ``SVR``

    """


    def __init__(self,
                 k_func,
                 learner_id='SVR'):
        """
        Constructor for the KernelMachine class.

        Parameters
        ----------
        k_func : KernelFunction
            The kernel function the kernel machine bases itself on

        learner_id : str
            Identifier for the estimator to be built based on the kernel function.
            Options: ``SVM`` and ``SVR``.
            Default: ``SVR``
        """

        self.k_func = k_func
        self.learner_id = learner_id
        self._estimator, self.param_grid = get_estimator(self.learner_id)


    def fit(self, X, y, sample_weight=None):
        """Fit the chosen Estimator based on the user-defined kernel.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.

        """

        self._train_X, self._train_y = check_X_y(X, y, y_numeric=True)

        self._km = KernelMatrix(self.k_func, name='train_km')
        self._km.attach_to(self._train_X)

        self._estimator.fit(X=self._km.full, y=self._train_y,
                            sample_weight=sample_weight)

        return self


    def predict(self, X):
        """
        Make predictions on the new samplets in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """

        X = check_array(X)

        # sample_one must be test data to get the right shape for sklearn X
        self._km.attach_to(sample_one=X, sample_two=self._train_X)
        test_train_KM = self._km.full
        predicted_y = self._estimator.predict(test_train_KM)

        return predicted_y
        # TODO we don't need data type conversion, as things can be
        #  different in classifiers and regressors?
        # return np.asarray(predicted_y, dtype=np.intp)


    def get_params(self, deep=True):
        """returns all the relevant parameters for this estimator!"""

        # est_param_dict = self._estimator.get_params(deep=deep)
        # est_param_dict['k_func'] = self.k_func
        # est_param_dict['learner_id'] = self.learner_id
        # est_param_dict['learner_params'] = self.learner_params
        # return est_param_dict

        return {'k_func'    : self.k_func,
                'learner_id': self.learner_id}


    def set_params(self, **parameters):
        """Param setter"""

        for parameter, value in parameters.items():
            if parameter in ('k_func', 'learner_id'):  # 'learner_params'
                setattr(self, parameter, value)
            # else:
            #     setattr(self._estimator, parameter, value)

        return self


class OptimalKernelSVR(SVR, RegressorMixin):
    """
    An estimator to learn the optimal kernel for a given sample and
    build a support vector regressor based on this custom kernel.

    This class is wrapped around the sklearn SVR estimator to function as its
    drop-in replacement, whose implementation is in turn based on LIBSVM.

    Parameters
    ----------

    k_bucket : KernelBucket or str
        An instance of KernelBucket that contains all the kernels to be compared,
        or a string identifying the sampling_strategy which populates a KernelBucket.

    method : str
        Scoring method to rank different kernels

    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    epsilon : float, optional (default=0.1)
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.


    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.

    support_vectors_ : array-like, shape = [nSV, n_features]
        Support vectors.

    dual_coef_ : array, shape = [1, n_SV]
        Coefficients of the support vector in the decision function.

    coef_ : array, shape = [1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : array, shape = [1]
        Constants in decision function.

    """


    def __init__(self, k_bucket='exhaustive',
                 method='cv_risk',
                 C=1.0,
                 epsilon=0.1,
                 shrinking=True,
                 tol=1e-3):
        """

        Parameters
        ----------
        k_bucket : KernelBucket or str
            An instance of KernelBucket that contains all the kernels to be compared,
            or a string identifying sampling strategy to populate a KernelBucket.

        method : str
            Scoring method to rank different kernels

        C : float, optional (default=1.0)
            Penalty parameter C of the error term.

        epsilon : float, optional (default=0.1)
             Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
             within which no penalty is associated in the training loss function
             with points predicted within a distance epsilon from the actual
             value.

        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.

        tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.

        """

        super().__init__(kernel='precomputed', C=C, epsilon=epsilon,
                         shrinking=shrinking, tol=tol)

        self.k_bucket = k_bucket
        self.method = method
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.tol = tol


    def fit(self, X, y, sample_weight=None):
        """Estimate the optimal kernel, and fit a SVM based on the custom kernel.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.

        """

        if isinstance(self.k_bucket, str):
            try:
                # using a new internal variable to retain user supplied param
                self._k_bucket = make_kernel_bucket(self.k_bucket)
            except:
                raise ValueError('Input for k_func can only an instance of '
                                 'KernelBucket or a sampling strategy to generate '
                                 'one with make_kernel_bucket.'
                                 'sampling strategy must be one of {}'
                                 ''.format(cfg.kernel_bucket_strategies))
        elif isinstance(self.k_bucket, KernelBucket):
            self._k_bucket = deepcopy(self.k_bucket)
        else:
            raise ValueError('Input for k_func can only an instance of '
                             'KernelBucket or a sampling strategy to generate '
                             'one with make_kernel_bucket')

        self._train_X, self._train_y = check_X_y(X, y, y_numeric=True)

        self.opt_kernel_ = find_optimal_kernel(self._k_bucket,
                                               self._train_X, self._train_y,
                                               method=self.method,
                                               estimator_name='SVR')

        super().fit(X=self.opt_kernel_.full, y=self._train_y,
                    sample_weight=sample_weight)

        # temporary hack to pass sklearn estimator checks till a bug is fixed
        # for more see: https://github.com/scikit-learn/scikit-learn/issues/14712
        self.n_iter_ = 1

        return self


    def predict(self, X):
        """
        Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """

        if not hasattr(self, 'opt_kernel_'):
            raise ValueError("Can't predict - not fitted yet! Run .fit() first.")

        X = check_array(X)

        # sample_one must be test data to get the right shape for sklearn X
        self.opt_kernel_.attach_to(sample_one=X, sample_two=self._train_X)
        test_train_KM = self.opt_kernel_.full
        predicted_y = super().predict(test_train_KM)

        return predicted_y
        # TODO we don't need data type coversion, as its not classification?
        # return np.asarray(predicted_y, dtype=np.intp)


    def get_params(self, deep=True):
        """returns all the relevant parameters for this estimator!"""

        return {'k_bucket' : self.k_bucket,
                'method'   : self.method,
                'C'        : self.C,
                'epsilon'  : self.epsilon,
                'shrinking': self.shrinking,
                'tol'      : self.tol}


    def set_params(self, **parameters):
        """Param setter"""

        for parameter, value in parameters.items():
            if parameter in ('k_bucket', 'method',
                             'C', 'epsilon', 'shrinking', 'tol'):
                setattr(self, parameter, value)

        return self


class OptimalKernelSVC(SVC, ClassifierMixin):
    """
    An estimator to learn the optimal kernel for a given sample and
    build a support vector classifier based on this custom kernel.

    This class is wrapped around the sklearn SVC estimator to function as its
    drop-in replacement, whose implementation is in turn based on LIBSVM.

    Parameters
    ----------

    k_bucket : KernelBucket or str
        An instance of KernelBucket that contains all the kernels to be compared,
        or a string identifying the sampling_strategy which populates a KernelBucket.

    method : str
        Scoring method to rank different kernels

    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.


    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.

    support_vectors_ : array-like, shape = [nSV, n_features]
        Support vectors.

    dual_coef_ : array, shape = [1, n_SV]
        Coefficients of the support vector in the decision function.

    coef_ : array, shape = [1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : array, shape = [1]
        Constants in decision function.

    """


    def __init__(self, k_bucket='exhaustive',
                 method='cv_risk',
                 C=1.0,
                 shrinking=True,
                 tol=1e-3):
        """
        SVC classifier trained with the sample-wise optimal kernel

        Parameters
        ----------
        k_bucket : KernelBucket or str
            An instance of KernelBucket that contains all the kernels to be compared,
            or a string identifying sampling strategy to populate a KernelBucket.

        method : str
            Scoring method to rank different kernels

        C : float, optional (default=1.0)
            Penalty parameter C of the error term.

        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.

        tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.

        """

        super().__init__(kernel='precomputed',
                         C=C,
                         shrinking=shrinking,
                         tol=tol)

        self.k_bucket = k_bucket
        self.method = method
        self.C = C
        self.shrinking = shrinking
        self.tol = tol

    @property
    def _pairwise(self):
        "temp hack to pass cross_val_score"
        return False


    def fit(self, X, y, sample_weight=None):
        """Estimate the optimal kernel, and fit a SVM based on the custom kernel.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.

        """

        if isinstance(self.k_bucket, str):
            try:
                # using a new internal variable to retain user supplied param
                self._k_bucket = make_kernel_bucket(self.k_bucket)
            except:
                raise ValueError('Input for k_func can only an instance of '
                                 'KernelBucket or a sampling strategy to generate '
                                 'one with make_kernel_bucket.'
                                 'sampling strategy must be one of {}'
                                 ''.format(cfg.kernel_bucket_strategies))
        elif isinstance(self.k_bucket, KernelBucket):
            self._k_bucket = deepcopy(self.k_bucket)
        else:
            raise ValueError('Input for k_func can only an instance of '
                             'KernelBucket or a sampling strategy to generate '
                             'one with make_kernel_bucket')

        self._train_X, self._train_y = check_X_y(X, y, y_numeric=True)

        self.opt_kernel_ = find_optimal_kernel(self._k_bucket,
                                               self._train_X, self._train_y,
                                               method=self.method,
                                               estimator_name='SVR')

        super().fit(X=self.opt_kernel_.full, y=self._train_y,
                    sample_weight=sample_weight)

        # temporary hack to pass sklearn estimator checks till a bug is fixed
        # for more see: https://github.com/scikit-learn/scikit-learn/issues/14712
        self.n_iter_ = 1

        return self


    def predict(self, X):
        """
        Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """

        if not hasattr(self, 'opt_kernel_'):
            raise ValueError("Can't predict - not fitted yet! Run .fit() first.")

        X = check_array(X)

        # sample_one must be test data to get the right shape for sklearn X
        self.opt_kernel_.attach_to(sample_one=X, sample_two=self._train_X)
        test_train_KM = self.opt_kernel_.full
        predicted_y = super().predict(test_train_KM)

        return predicted_y
        # TODO we don't need data type coversion, as its not classification?
        # return np.asarray(predicted_y, dtype=np.intp)


    def get_params(self, deep=True):
        """returns all the relevant parameters for this estimator!"""

        return {'k_bucket' : self.k_bucket,
                'method'   : self.method,
                'C'        : self.C,
                'shrinking': self.shrinking,
                'tol'      : self.tol}


    def set_params(self, **parameters):
        """Param setter"""

        for parameter, value in parameters.items():
            if parameter in ('k_bucket', 'method',
                             'C', 'shrinking', 'tol'):
                setattr(self, parameter, value)

        return self
