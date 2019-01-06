# -*- coding: utf-8 -*-

"""Kernel methods module."""

import numpy as np
from numpy import multiply as elem_wise_multiply
from scipy.linalg import eigh, LinAlgError
import warnings
import traceback

def is_positive_semidefinite(sym_matrix,
                             tolerance=1e-6,
                             verbose=False):
    """
    Tests whether a given matrix is PSD.

    A symmetric matrix is PSD if ALL its eigen values >= 0 (non-negative).
        If any of its eigen values are negative, it is not PSD.

    Accouting for numerical instabilities with tolerance

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
                  '{}'.format(eig_values[:min(10,len(eig_values))]))
        if any(eig_values < -tolerance): # notice the negative sign before tolerance
            psd = False
        else:
            psd = True

    return psd


# shorter alias
is_PSD = is_positive_semidefinite


def center_km(KM):
    """
    Center a given kernel matrix.

    Implements the definition according to Lemma 1 in Section 2.2 in
    Cortes, Corinna, Mehryar Mohri, and Afshin Rostamizadeh, 2012,
        "Algorithms for Learning Kernels Based on Centered Alignment",
        Journal of Machine Learning Research 13(Mar): 795–828.
    """

    if isinstance(KM, np.ndarray):
        if KM.shape[0] == KM.shape[1]:
            n_rows = KM.shape[0]
        else:
            raise ValueError('Input matrix is not square!')
    else:
        raise ValueError('Unknown format for input matrix -'
                         'must be a square numpy ndarray')

    # directly initializing one_oneT without going through unnecessary matrix products
    #   vec_1s = np.ones((n_rows, 1)) # row vector of 1s
    #   one_oneT = vec_1s.dot(vec_1s.T) # 1 dot 1T
    one_oneT = np.ones((n_rows, n_rows))
    Ic = np.eye(n_rows) - (one_oneT/n_rows)

    return Ic.dot(KM).dot(Ic)


def frobenius_product(A, B):
    """Computes the Frobenious product between two matrices of equal dimension.

    <A, B>_F is equal to the sum of element-wise products between A and B.

    .. math::
        <\mathbf{A}, \mathbf{B}>_F = \sum_{i, j} \mathbf{A}_{ij} \mathbf{B}_{ij}

    """

    if A.shape != B.shape:
        raise ValueError('Dimensions of the two matrices must be the same '
                         'to compute Frobenious product! They differ: {}, {}'
                         ''.format(A.shape, B.shape))

    return np.sum(elem_wise_multiply(A, B), axis=None)


def frobenius_norm(A):
    """Computes the Frobenius norm of a matrix A.

    It is the square root of the Frobenius product with itself.

    """

    return np.sqrt(frobenius_product(A, A))


def alignment_centered(km_one, km_two, centered_already=False):
    """
    Computes the centered alignment between two kernel matrices

    (Alignment is computed on centered kernel matrices)

    Implements Definition 4 (Kernel matrix alignment) from Section 2.3 in
    Cortes, Corinna, Mehryar Mohri, and Afshin Rostamizadeh, 2012,
        "Algorithms for Learning Kernels Based on Centered Alignment",
        Journal of Machine Learning Research 13(Mar): 795–828.
    """

    if km_one.shape != km_two.shape:
        raise ValueError('Dimensions of the two matrices must be the same '
                         'to compute their alignment! They differ: {}, {}'
                         ''.format(km_one.shape, km_two.shape))

    if not centered_already:
        kC_one = center_km(km_one)
        kC_two = center_km(km_two)
    else:
        kC_one = km_one
        kC_two = km_two

    fnorm_one = frobenius_norm(kC_one)
    fnorm_two = frobenius_norm(kC_two)

    if np.isclose(fnorm_one, 0.0):
        raise ValueError('The Frobenius norm of KM1 is 0. Can not compute alignment!')

    if np.isclose(fnorm_two, 0.0):
        raise ValueError('The Frobenius norm of KM1 is 0. Can not compute alignment!')

    return frobenius_product(kC_one, kC_two) / (fnorm_one*fnorm_two)


def eval_similarity(km_one, km_two):
    """Evaluate similarity between two kernel matrices"""

    pass

def linear_comb(km_list, param_list):
    """Linear combinations of a list of kernels"""
