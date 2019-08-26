Kernel functions
===================

Kernel functions are the key to producing kernel matrices and hence are the backbone of kernel methods and machines. These are represented by a fundamental [abstract base] class called **``BaseKernelFunction``**, which defines several desirable properties, such as

 - ensures it induces a positive semi-definite kernel matrix
 - making it callable, accepting at least two inputs (data points)
 - a readable representation of the underlying function with a name.

This modularization of all the kernel functions separate from the ``KernelMatrix`` class enables us to support diverse and mixed data types. This also enables to support various formats and data structures beyond numpy arrays such as `pyradigm <https://github.com/raamana/pyradigm>`_.

We also provide a ``KernelFromCallable`` class which makes it even easier to define a kernel function just by specifying the underlying computation, without having to define a fully separate class.

In addition, the following classes are provided to enable compositional representation of multiple kernel functions for advanced applications: ``CompositeKernel``, ``ProductKernel``, ``SumKernel``, ``AverageKernel``, and ``WeightedAverageKernel``.


``kernelmethods`` aims to offer kernel functions that can operate on the following data types:

 - :doc:`numeric_kernels`
 - :doc:`categorical_kernels`
 - :doc:`string_kernels`
 - :doc:`graph_kernels`
 - and others such as trees and sequences (TBA).


**Below**, we document the API for the important classes related to kernel functions, such as :

 - ``BaseKernelFunction``
 - ``KernelFromCallable``
 - and composites as noted above.

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

