Kernel functions
----------------

Kernel functions are the key to producing kernel matrices and hence are the backbone of kernel methods and machines. These are represented by a fundamental [abstract base] class called ``BaseKernelFunction``, which defines several desirable properties, such as making it callable, easy way to check if it induces a positive semi-definite as well as a readable representation of the underlying function.

We also provide a ``KernelFromCallable`` class which makes it even easier to define a kernel function just by specifying the underlying function, without having to define a fully separate class.

In addition, the following classes are provided to enable compositional represenation of multiple kernel functions for advanced applications: ``CompositeKernel``, ``ProductKernel``, ``SumKernel``, ``AverageKernel``, and ``WeightedAverageKernel``.


``kernelmethods`` offers kernel functions that can operate on the following data types:

 - :doc:`numeric_kernels`
 - :doc:`categorical_kernels`
 - :doc:`string_kernels`
 - :doc:`graph_kernels`
 - and others such as trees and sequences (TBA).

.. automodule:: kernelmethods
   :members: BaseKernelFunction, KernelFromCallable
   :undoc-members:
   :inherited-members:
   :show-inheritance:


Composite kernel functions
---------------------------


.. automodule:: kernelmethods
   :members: CompositeKernel, ProductKernel, SumKernel, AverageKernel, WeightedAverageKernel
   :undoc-members:
   :inherited-members:
   :show-inheritance:

