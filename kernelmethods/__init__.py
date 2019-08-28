# -*- coding: utf-8 -*-

"""Top-level package for kernelmethods."""

__all__ = ['KernelMatrix',
           'BaseKernelFunction',
           'KernelMethodsException', 'KMAccessError', 'KMNormError',
           'KMSetAdditionError',
           'PolyKernel', 'GaussianKernel', 'LaplacianKernel', 'LinearKernel',
           'Chi2Kernel', 'SigmoidKernel',
           'KernelBucket', 'KernelSet',
           'KernelMachine', 'OptimalKernelSVC', 'OptimalKernelSVR', ]

from kernelmethods.algorithms import (KernelMachine, KernelMachineRegressor,
                                      OptimalKernelSVC, OptimalKernelSVR)
from kernelmethods.base import BaseKernelFunction, KernelMatrix, KernelSet
from kernelmethods.config import (KMAccessError, KMNormError, KMSetAdditionError,
                                  KernelMethodsException)
from kernelmethods.numeric_kernels import (Chi2Kernel, GaussianKernel,
                                           LaplacianKernel, LinearKernel, PolyKernel,
                                           SigmoidKernel)
from kernelmethods.sampling import KernelBucket
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

__author__ = """Pradeep Reddy Raamana"""
__email__ = 'raamana@gmail.com'
