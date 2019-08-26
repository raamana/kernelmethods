# -*- coding: utf-8 -*-

"""Top-level package for kernelmethods."""

__all__ = ['KernelMatrix',
           'BaseKernelFunction',
           'KernelMethodsException', 'KMAccessError', 'KMNormError',
           'KMSetAdditionError',
           'PolyKernel', 'GaussianKernel', 'LaplacianKernel', 'LinearKernel',
           'Chi2Kernel', 'SigmoidKernel',
           'KernelBucket', 'KernelSet',
           'OptimalKernelSVR', 'KernelMachine']

from kernelmethods.config import KernelMethodsException, \
    KMAccessError, KMNormError, KMSetAdditionError
from kernelmethods.base import BaseKernelFunction, KernelMatrix, KernelSet
from kernelmethods.numeric_kernels import PolyKernel, GaussianKernel, LaplacianKernel, \
    LinearKernel, Chi2Kernel, SigmoidKernel
from kernelmethods.sampling import KernelBucket
from kernelmethods.algorithms import KernelMachine, OptimalKernelSVR

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = """Pradeep Reddy Raamana"""
__email__ = 'raamana@gmail.com'
