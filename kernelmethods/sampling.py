from kernelmethods import config as cfg
from kernelmethods.base import KernelSet, KernelMatrix
from kernelmethods.numeric_kernels import PolyKernel, LinearKernel, GaussianKernel, \
    LaplacianKernel


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
