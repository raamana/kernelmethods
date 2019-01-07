import numpy as np
from scipy.stats.stats import pearsonr

from kernelmethods import config as cfg
from kernelmethods.base import KernelMatrix, KernelSet
from kernelmethods.numeric_kernels import GaussianKernel, LaplacianKernel, LinearKernel, \
    PolyKernel
from kernelmethods.operations import alignment_centered


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
                self.append(KernelMatrix(kernel_func(**{param_name: val})))


def make_kernel_bucket(strategy='exhaustive'):
    """Generates a candidate kernels based on user preferences."""

    strategy = strategy.lower()
    if strategy == 'exhaustive':
        return KernelBucket(name='KBucketExhaustive',
                            poly_degree_values=cfg.default_degree_values_poly_kernel,
                            rbf_sigma_values=cfg.default_sigma_values_gaussian_kernel,
                            laplacian_gamma_values=cfg.default_gamma_values_laplacian_kernel)
    elif strategy == 'light':
        return KernelBucket(name='KBucketLight',
                            poly_degree_values=cfg.light_degree_values_poly_kernel,
                            rbf_sigma_values=cfg.light_sigma_values_gaussian_kernel,
                            laplacian_gamma_values=cfg.light_gamma_values_laplacian_kernel)
    else:
        raise ValueError('Invalid choice of strategy '
                         '- must be one of {}'.format(cfg.kernel_bucket_strategies))


def ideal_kernel(targets):
    """Computes the kernel matrix from the given target labels"""

    targets = np.array(targets).reshape((-1, 1))  # row vector

    return targets.dot(targets.T)


def correlation_km(k1, k2):
    """Computes correlation coefficient between two kernel matrices"""

    corr_coef, p_val = pearsonr(k1.ravel(), k2.ravel())

    return corr_coef


def pairwise_similarity(k_bucket, metric='corr'):
    """Computes the similarity between all pairs of kernel matrices in a given bucket."""

    # mutual info?
    metric_func = {'corr' : correlation_km,
                   'align': alignment_centered}

    num_kernels = k_bucket.size
    estimator = metric_func[metric]
    pairwise_metric = np.full((k_bucket.size, k_bucket.size), fill_value=np.nan)
    for idx_one in range(num_kernels):
        # kernel matrix is symmetric
        for idx_two in range(idx_one, num_kernels): # computing i,i as well to be consistent
            pairwise_metric[idx_one, idx_two] = estimator(k_bucket[idx_one].full,
                                                          k_bucket[idx_two].full)

        # not computing diagonal entries (can also be set to 1 for some metrics)

    # making it symmetric
    idx_lower_tri = np.tril_indices(num_kernels)
    pairwise_metric[idx_lower_tri] = pairwise_metric.T[idx_lower_tri]

    return pairwise_metric
