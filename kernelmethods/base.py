
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
