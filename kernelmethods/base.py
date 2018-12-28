
from abc import ABC, abstractmethod

class BaseKernelFunction(ABC):
    """Abstract base class for kernel functions.

    Enforces each derived kernel:
    1. to be callable, with two inputs
    2. to have a name and a str representation

    """

    def __init__(self, name):
        """
        Constructor.

        Parameters
        ----------
        name : str
            short name to describe the nature of the kernel function

        """

        self.name = name

    @abstractmethod
    def __call__(self, x, y):
        """Actual computation!"""

    @abstractmethod
    def __str__(self):
        """Representation"""


class KernelFromCallable(BaseKernelFunction):
    """Class to create a custom kernel from a given callable."""


    def __init__(self, input_func, name=None, **func_params):
        """
        Constructor.

        Parameters
        ----------
        input_func : callable
            A callable that can accept atleast 2 args
            Must not be builtin or C function.
            If func is a C or builtin func, wrap it in a python def

        name : str
            A name to identify this kernel in a human readable way

        func_params : dict
            Parameters to func

        """

        self.func = check_callable(input_func, min_num_args=2)
        self.params = func_params

        super().__init__(name=get_callable_name(input_func, name))


    def __call__(self, x, y):
        """Actual computation!"""

        return self.func(x, y, **self.params)


    def __str__(self):
        """human readable repr"""

        arg_repr = '({})'.format(self.params) if len(self.params) > 0 else ''
        return "{}{}".format(self.name, arg_repr)


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


