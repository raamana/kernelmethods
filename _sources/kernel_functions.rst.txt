Kernel functions
===================

Kernel functions are the key to producing kernel matrices and hence are the backbone of kernel methods and machines. These are represented by a fundamental [abstract base] class called ``BaseKernelFunction``, which defines several desirable properties, such as

 - easy way to check if it induces a positive semi-definite
 - making it callable, accepting at least two inputs (data points)
 - a readable representation of the underlying function with a name.

We also provide a ``KernelFromCallable`` class which makes it even easier to define a kernel function just by specifying the underlying computation, without having to define a fully separate class.

In addition, the following classes are provided to enable compositional representation of multiple kernel functions for advanced applications: ``CompositeKernel``, ``ProductKernel``, ``SumKernel``, ``AverageKernel``, and ``WeightedAverageKernel``.


``kernelmethods`` offers kernel functions that can operate on the following data types:

 - :doc:`numeric_kernels`
 - :doc:`categorical_kernels`
 - :doc:`string_kernels`
 - :doc:`graph_kernels`
 - and others such as trees and sequences (TBA).


BaseKernelFunction
---------------------------

.. autoclass:: kernelmethods.BaseKernelFunction
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:


KernelFromCallable
---------------------------

.. autoclass:: kernelmethods.base.KernelFromCallable
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:


Composite kernel functions
---------------------------


.. autoclass:: kernelmethods.base.CompositeKernel
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:


.. autoclass:: kernelmethods.base.ProductKernel
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: kernelmethods.base.SumKernel
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: kernelmethods.base.AverageKernel
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: kernelmethods.base.WeightedAverageKernel
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

