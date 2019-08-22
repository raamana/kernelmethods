# -*- coding: utf-8 -*-

"""
This module implements the common kernel operations such as

 - normalization of a kernel matrix (KM),
 - centering (one- and two-sample cases),
 - evaluating similarity, computing alignment,
 - frobenius norms,
 - linear combinations and
 - checking whether a KM is PSD.

API
----

"""

import traceback
import warnings

import numpy as np
from kernelmethods.config import KMNormError, KernelMethodsException
from kernelmethods.utils import contains_nan_inf, ensure_ndarray_1D
from numpy import multiply as elem_wise_multiply
from scipy.linalg import LinAlgError, eigh


def is_positive_semidefinite(sym_matrix,
                             tolerance=1e-6,
                             verbose=False):
    """
    Tests whether a given matrix is positive-semidefinite (PSD).

    A symmetric matrix is PSD if ALL its eigen values >= 0 (non-negative).
    If any of its eigen values are negative, it is not PSD.

    This functions accounts for numerical instabilities with a tolerance parameter.

    This function can also be called with a shorthand ``is_PSD()``

    Parameters
    ----------
    sym_matrix : ndarray
        Matrix to be evaluted for PSDness

    tolerance : float
        Tolerance parameter to account for numerical instabilities in the eigen
        value computations (which can result in negative eigen values very slightly
        below 0)

    verbose : bool
        Flag to indicate whether to print traceback in case of errors
        during the computation of the eigen values

    Returns
    -------
    psd : bool
        Flag indicating whether the matrix is PSD.

    """

    if not isinstance(sym_matrix, np.ndarray):
        raise TypeError('Input matrix must be in numpy array format!')

    if sym_matrix.shape[0] != sym_matrix.shape[1]:
        warnings.warn('Input matrix is not square, and hence not PSD')
        return False

    if not np.isclose(sym_matrix, sym_matrix.T).all():
        warnings.warn('Input matrix is not symmetric, and hence not PSD')
        return False

    try:
        eig_values = eigh(sym_matrix, eigvals_only=True)
    except LinAlgError:
        if verbose:
            traceback.print_exc()
        # we are not actually raising LinAlgError, just using it to categorize as
        # not PSD. So, can't use test cases to try raise LinAlgError, so not
        # testable!
        print('LinAlgError raised - eigen value computation failed --> not PSD')
        psd = False
    except:
        if verbose:
            traceback.print_exc()
        warnings.warn('Unknown exception during eigen value computation --> not PSD')
        psd = False
    else:
        if verbose:
            print('Smallest eigen values are:\n'
                  '{}'.format(eig_values[:min(10, len(eig_values))]))
        if any(eig_values < -tolerance):  # notice the negative sign before tolerance
            psd = False
        else:
            psd = True

    return psd


# shorter alias
is_PSD = is_positive_semidefinite


def center_km(KM):
    """
    Centers a given kernel matrix.

    Implements the definition according to Lemma 1 in Section 2.2 in
    Cortes, Corinna, Mehryar Mohri, and Afshin Rostamizadeh, 2012, "Algorithms for
    Learning Kernels Based on Centered Alignment", Journal of Machine Learning
    Research 13(Mar): 795–828.

    Parameters
    ----------
    KM : ndarray
        Symmetric matrix to be centered.

    Returns
    -------
    centered_km : ndarray
        Centered kernel matrix

    """

    if isinstance(KM, np.ndarray):
        if KM.shape[0] == KM.shape[1]:
            n_rows = KM.shape[0]
        else:
            raise ValueError('Input matrix is not square!')
    else:
        raise ValueError('Unknown format for input matrix -'
                         'must be a square numpy ndarray')

    # directly initializing one_oneT without going through unnecessary matrix
    # products
    #   vec_1s = np.ones((n_rows, 1)) # row vector of 1s
    #   one_oneT = vec_1s.dot(vec_1s.T) # 1 dot 1T
    one_oneT = np.ones((n_rows, n_rows))
    Ic = np.eye(n_rows) - (one_oneT / n_rows)

    return Ic.dot(KM).dot(Ic)


def normalize_km(KM, method='cosine'):
    """
    Normalize a kernel matrix to have unit diagonal.

    Cosine normalization normalizes the kernel matrix to have unit diagonal.
    Implements definition according to Section 5.1 in book (Page 113)
    Shawe-Taylor and Cristianini, "Kernels Methods for Pattern Analysis", 2004

    Matrix must be square (and coming from a single sample: K(X,X), not K(X,Y)

    Parameters
    ----------
    KM : ndarray
        Symmetric matrix to be normalized

    method : str
        Method of normalization. Options: ``cosine`` only.

    Returns
    -------
    normed_km : ndarray
        Normalized kernel matrix

    """

    if KM.shape[0] != KM.shape[1]:
        raise ValueError('Input kernel matrix must be square! '
                         'i.e. K(X,X) must be generated from '
                         'inner products on a single sample X, '
                         'not an inner-product on two separate samples X and Y')

    try:
        method = method.lower()
        if method == 'cosine':
            km_diag = KM.diagonal()
            if np.isclose(km_diag, 0.0).any():
                raise KMNormError(
                    'Some diagnoal entries in KM are [close to] zero - '
                    ' this results in infinite or Nan values '
                    'during Cosine normalization of KM!')
            # D = diag(1./sqrt(diag(K)))
            # normed_K = D * K * D;
            _1bySqrtDiag = np.diagflat(1 / np.sqrt(km_diag))
            # notice @ is matrix multiplication operator
            normed_km = _1bySqrtDiag @ KM @ _1bySqrtDiag
            # in case of two samples K(X, Y), the left- and right-most factors
            #  must come from K(X,X) & K(Y,Y) respectively: see normalize_km_2sample
        else:
            raise NotImplementedError('normalization method {} is not implemented'
                                      'yet!'.format(method))
    except (KMNormError, KernelMethodsException):
        raise
    except:
        warn('Unable to normalize kernel matrix using method {}'.format(method))
        raise
    else:
        if contains_nan_inf(normed_km):
            warnings.warn('normalization of kernel matrix resulted in Inf / NaN '
                          'values - check your parameters and data!')

    return normed_km


def normalize_km_2sample(cross_K_XY, diag_K_XX, diag_K_YY, method='cosine'):
    """
    Normalize a kernel matrix K(X,Y) to have unit diagonal.

    Cosine normalization normalizes the kernel matrix to have unit diagonal.
    Implements definition _similar_ to Section 5.1 in book (Page 113)
    Shawe-Taylor and Cristianini, "Kernels Methods for Pattern Analysis", 2004


    Parameters
    ----------
    cross_K_XY : ndarray, 2D
        Matrix of inner-products for samples from X onto Y i.e. K(X,Y)

    diag_K_XX : array
        Diagonal from matrix of inner-products for samples from X onto itself i.e.
        K(X,X)
        K(X,X) must NOT be normalized (otherwise they will all be 1s)

    diag_K_YY : array
        Diagonal from matrix of inner-products for samples from Y onto itself i.e.
        K(Y,Y)

    Returns
    -------
    normed_km : ndarray
        Normalized version of K(X,Y)

        NOTE: K_XY may NOT have unit diagonal, as k(x,y) != sqrt(k(x,x))*sqrt(k(y,y))
    """

    if diag_K_XX.size != cross_K_XY.shape[0] or \
        cross_K_XY.shape[1] != diag_K_YY.size:
        raise ValueError('Shape mismatch for multiplication across the 3 kernel '
                         'matrices! Length of diag_K_XX must match '
                         'number of rows in K_XY, and number of columns in K_XY '
                         'must match length of diag_K_XX.')

    method = method.lower()
    if method == 'cosine':
        if np.isclose(diag_K_XX, 0.0).any() or \
            np.isclose(diag_K_YY, 0.0).any():
            raise KMNormError(
                'Some diagnoal entries in one of the KMs are [close to] zero - '
                ' this results in infinite or Nan values '
                'during Cosine normalization of KM!')

        # using diagflat to explicitly construct a matrix from diag values
        diag_factor_xx = np.diagflat(1 / np.sqrt(diag_K_XX))
        diag_factor_yy = np.diagflat(1 / np.sqrt(diag_K_YY))
        # notice @ is matrix multiplication operator
        normed_km = diag_factor_xx @ cross_K_XY @ diag_factor_yy
    else:
        raise NotImplementedError('Two-sample normalization method {} is not'
                                  'implemented yet!'.format(method))

    return normed_km


def frobenius_product(A, B):
    """
    Computes the Frobenious product between two matrices of equal dimensions.

    <A, B>_F is equal to the sum of element-wise products between A and B.

    .. math::
        <\mathbf{A}, \mathbf{B}>_F = \sum_{i, j} \mathbf{A}_{ij} \mathbf{B}_{ij}

    Parameters
    ----------
    A, B : ndarray
        Two matrices of equal dimensions to compute the product.

    Returns
    -------
    product : float
        Frobenious product

    """

    if A.shape != B.shape:
        raise ValueError('Dimensions of the two matrices must be the same '
                         'to compute Frobenious product! They differ: {}, {}'
                         ''.format(A.shape, B.shape))

    return np.sum(elem_wise_multiply(A, B), axis=None)


def frobenius_norm(A):
    """Computes the Frobenius norm of a matrix A, which  is the square root of the
    Frobenius product with itself.

    Parameters
    ----------
    A : ndarray
        Matrix to compute the norm of

    Returns
    -------
    norm : float
        Frobenious norm

    """

    return np.sqrt(frobenius_product(A, A))


def alignment_centered(km_one, km_two,
                       value_if_zero_division='raise',
                       centered_already=False):
    """
    Computes the centered alignment between two kernel matrices

    (Alignment is computed on centered kernel matrices)

    Implements Definition 4 (Kernel matrix alignment) from Section 2.3 in Cortes,
    Corinna, Mehryar Mohri, and Afshin Rostamizadeh, 2012, "Algorithms for
    Learning Kernels Based on Centered Alignment", Journal of Machine Learning
    Research 13(Mar): 795–828.

    Parameters
    ----------

    km_one, km_two : KernelMatrix

    value_if_zero_division : str or float
        determines the value of alignment, in case the norm of one of the two
        kernel matrices is close to zero and we are unable to compute it.

        Default is 'raise', requesting to raise an exception.

        One could also choose 0.0, which assigns lowest alignment,  effectively
        discarding it for ranking purposes.

    centered_already : bool
        Flag to indicate whether the input kernel matrices are centered already
        or not. If False, input KMs will be centered.

    Returns
    -------
    centered_alignment : float
        Value of centered_alignment between the two kernel matrices

    """

    if km_one.shape != km_two.shape:
        raise ValueError('Dimensions of the two matrices must be the same '
                         'to compute their alignment! They differ: {}, {}'
                         ''.format(km_one.shape, km_two.shape))

    if not isinstance(km_one, np.ndarray) or not isinstance(km_two, np.ndarray):
        raise TypeError('Input KMs must be numpy arrays')

    if not centered_already:
        kC_one = center_km(km_one)
        kC_two = center_km(km_two)
    else:
        kC_one = km_one
        kC_two = km_two

    fnorm_one = frobenius_norm(kC_one)
    fnorm_two = frobenius_norm(kC_two)

    if np.isclose(fnorm_one, 0.0) or np.isclose(fnorm_two, 0.0):
        if value_if_zero_division in ('raise', Exception):
            raise ValueError('The Frobenius norm of KM1 or KM2 is 0. '
                             'Can not compute alignment!')
        else:
            warnings.warn('The Frobenius norm of KM1 or KM2 is 0. Setting value of '
                          'alignment as {} as requested'.format(
                value_if_zero_division))
            return value_if_zero_division

    return frobenius_product(kC_one, kC_two) / (fnorm_one * fnorm_two)


def eval_similarity(km_one, km_two):
    """Evaluate similarity between two kernel matrices"""

    raise NotImplementedError()


def linear_combination(km_set, weights):
    """
    Weighted linear combinations of a set of given kernel matrices

    Parameters
    ----------
    km_set : KernelSet
        Collection of compatible kernel matrices

    weights : Iterable
        Set of weights for the kernel matrices in km_set

    Returns
    -------
    lin_comb_KM : ndarray
        Final result of weighted linear combination of the kernel matrix set

    """

    if km_set.size == len(weights):
        weights = ensure_ndarray_1D(weights)
    else:
        raise ValueError('Number of weights ({}) supplied differ '
                         'from the kernel set size ({})'
                         ''.format(km_set.size, len(weights)))

    # TODO should we not ensure weights sum to 1.0?

    # Computes the weighted average kernel
    # km_set.num_samples is a tuple (N, M) when operating on two samples
    #   e.g. train x test
    KM = np.zeros(km_set.num_samples)
    for weight, km in zip(weights, km_set):
        KM = KM + weight * km.full

    return KM
