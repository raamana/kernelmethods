
import numpy as np
from scipy.sparse import issparse

from kernelmethods.base import BaseKernelFunction
from kernelmethods.utils import check_input_arrays

class PolyKernel(BaseKernelFunction):
    """Polynomial kernel function"""

    def __init__(self, degree=2, b=0):
        """
        Constructor

        Parameters
        ----------
        degree : int
            degree to raise the inner product

        b : float
            intercept

        """

        super().__init__(name='polynomial')

        # TODO implement param check
        self.degree = degree
        self.b = b

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        # TODO should we check x and y are of same dimension?

        # TODO special handling for sparse arrays
        #   (e.g. custom dot product) might be more efficient

        return (self.b + x.dot(y.T)) ** self.degree

    def __str__(self):
        """human readable repr"""

        return "{}(degree={},b={})".format(self.name, self.degree, self.b)

    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__
