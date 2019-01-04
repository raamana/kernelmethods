# -*- coding: utf-8 -*-

"""Kernel methods module."""

from kernelmethods.base import KernelMatrix, KernelSet
from kernelmethods.numeric_kernels import GaussianKernel, LaplacianKernel, LinearKernel, \
    PolyKernel
from kernelmethods import config as cfg
import numpy as np
from scipy.linalg import eigh, LinAlgError
import warnings
import traceback



class KernelBucket(KernelSet):
    """
    Class to generate a "bucket" of candidate kernels.

    Applications:
    1. to rank/filter/select kernels based on a given sample via many metrics


    """

    def __init__(self,
                 poly_degree_values=cfg.default_degree_values_poly_kernel,
                 rbf_sigma_values=cfg.default_sigma_values_gaussian_kernel,
                 laplacian_gamma_values=cfg.default_gamma_values_laplacian_kernel,
                 name='KernelBucket',
                 ):
        """constructor"""

        # start with the addition of linear kernel
        super().__init__(km_set=[LinearKernel(), ],
                         name=name)
        # not attached to a sample yet
        self._num_samples = None

        self._add_parametrized_kernels(poly_degree_values, PolyKernel, 'degree')
        self._add_parametrized_kernels(rbf_sigma_values, GaussianKernel, 'sigma')
        self._add_parametrized_kernels(laplacian_gamma_values, LaplacianKernel, 'gamma')

    def _add_parametrized_kernels(self, values, kernel_func, param_name):
        """Adds a list of kernels corr. to various values for a given param"""

        if values is not None:
            for val in values:
                self.append(KernelMatrix(kernel_func(**{param_name:val})))


class PredictiveModelFromKernelMatrix(KernelMatrix):
    """Class to turn a given kernel into a predictive model"""

    raise NotImplementedError()


def is_positive_semidefinite(input_matrix,
                             tolerance=1e-6,
                             verbose=False):
    """
    Tests whether a given matrix is PSD.

    A symmetric matrix is PSD if ALL its eigen values >= 0 (non-negative).
        If any of its eigen values are negative, it is not PSD.

    Accouting for numerical instabilities with tolerance

    """

    if isinstance(input_matrix, KernelMatrix):
        sym_matrix = input_matrix.full
    else:
        sym_matrix = input_matrix

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


def eval_similarity(km_one, km_two):
    """Evaluate similarity between two kernel matrices"""

    pass

def linear_comb(km_list, param_list):
    """Linear combinations of a list of kernels"""
