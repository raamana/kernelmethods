# -*- coding: utf-8 -*-

"""Top-level package for kernelmethods."""

__all__ = ['KernelMatrix',
           'BaseKernelFunction',
           'KernelMethodsException', 'KMAccessError', 'KMNormError',
           'KMSetAdditionError',
           'PolyKernel', 'GaussianKernel', 'LaplacianKernel', 'LinearKernel',
           'Chi2Kernel', 'SigmoidKernel', 'HadamardKernel',
           'KernelBucket', 'KernelSet',
           'KernelMachine', 'OptimalKernelSVC', 'OptimalKernelSVR', ]

from kernelmethods.base import BaseKernelFunction, KernelMatrix, KernelSet
from kernelmethods.config import (KMAccessError, KMNormError, KMSetAdditionError,
                                  KernelMethodsException)
from kernelmethods.numeric_kernels import (Chi2Kernel, GaussianKernel,
                                           LaplacianKernel, LinearKernel, PolyKernel,
                                           SigmoidKernel, HadamardKernel)
from kernelmethods.sampling import KernelBucket
from ._version import __version__

__author__ = """Pradeep Reddy Raamana"""
__email__ = 'raamana@gmail.com'


def __getattr__(name):
    """Lazily expose estimator classes to keep import-time dependencies light."""

    if name in {
        'KernelMachine',
        'KernelMachineRegressor',
        'OptimalKernelSVC',
        'OptimalKernelSVR',
    }:
        from kernelmethods.algorithms import (
            KernelMachine,
            KernelMachineRegressor,
            OptimalKernelSVC,
            OptimalKernelSVR,
        )

        exports = {
            'KernelMachine': KernelMachine,
            'KernelMachineRegressor': KernelMachineRegressor,
            'OptimalKernelSVC': OptimalKernelSVC,
            'OptimalKernelSVR': OptimalKernelSVR,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
