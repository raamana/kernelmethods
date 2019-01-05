from kernelmethods import config as cfg
from kernelmethods.base import KernelSet, KernelMatrix
from kernelmethods.numeric_kernels import PolyKernel, LinearKernel, GaussianKernel, \
    LaplacianKernel
from kernelmethods.operations import alignment_centered
from scipy.stats.stats import pearsonr
import numpy as np

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

        # start with the addition of kernel matrix for linear kernel
        super().__init__(km_set=[KernelMatrix(LinearKernel()), ],
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


def correlation_km(k1, k2):
    """Computes correlation coefficient between two kernel matrices"""

    corr_coef, p_val = pearsonr(k1.ravel(), k2.ravel())

    return corr_coef


def pairwise_similarity(k_bucket, metric='corr'):
    """Computes the similarity between all pairs of kernel matrices in a given bucket."""


    metric_func = {'corr': correlation_km,
                   'align': alignment_centered}

    num_kernels = k_bucket.size
    estimator = metric_func[metric]
    piarwise_metric = np.full((k_bucket.size, k_bucket.size), fill_value=np.nan)
    for idx_one in range(num_kernels):
        # kernel matrix is symmetric
        for idx_two in range(idx_one+1, num_kernels):
            piarwise_metric[idx_one, idx_two] = estimator(k_bucket[idx_one].full,
                                                          k_bucket[idx_two].full)

        # not computing diagonal entries (can also be set to 1 for some metrics)

    # making it symmetric
    idx_lower_tri = np.tril_indices(num_kernels)
    piarwise_metric[idx_lower_tri] = piarwise_metric.T[idx_lower_tri]

    return piarwise_metric