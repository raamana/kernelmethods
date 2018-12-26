
from kernelmethods.base import BaseKernelFunction

import numpy as np
from scipy.sparse import issparse

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

        super().__init__('polynomial')

        # TODO implement param check
        self.degree = degree
        self.b = b

    def __call__(self, x, y):
        """Actual implementation of kernel func"""

        # TODO should we check x and y are of same dimension?

        if issparse(x):
            return np.array(x.dot(y.T).todense()) ** self.degree

        # ** is faster than math.pow() or np.power()
        if not hasattr(x, "shape"):
            return (self.b + x * y) ** self.degree
        else:
            return (self.b + x.dot(y.T)) ** self.degree

    def __str__(self):
        """human readable repr"""

        return "{}(degree={},b={})".format(self.name, self.degree, self.b)

    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__
