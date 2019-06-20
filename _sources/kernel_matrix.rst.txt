KernelMatrix class
------------------

``KernelMatrix`` is a self-contained class for the Gram matrix induced by a kernel function on a given sample. This class defines the central data structure for all kernel methods, as it acts a key bridge between input data space and the learning algorithms.

The class is designed in such a way that

 - it only computes elements of the kernel matrix (KM) as neeeded, and nothing more, which can save a lot computation and storage
 - it supports both callable as well as attribute access, allowing easy access to partial or random portions of the KM. Indexing is aimed to be compliant with numpy as much as possible.
 - allows parallel computation of different part of the KM to speed up computation when ``N`` is large
 - allows setting of user-defined attributes to allow easy identification and differentiation among a collection of KMs when working in applications such as Multiple Kernel Learning (MKL)
 - implements basic operations such as centering and normalization (whose implementation differs from that of manipulating regular matrices)
 - exposes several convenience attributes to make advanced development a breeze

This library also provides convenience wrappers:

 - ``KernelMatrixPrecomputed`` turns a precomputed kernel matrix into a ``KernelMatrix`` class with all its attractive properties
 - ``ConstantKernelMatrix`` that defines a ``KernelMatrix`` with a constant everywhere


.. autoclass:: kernelmethods.KernelMatrix
   :members:
   :undoc-members:


Exceptions
==========

.. autoclass:: kernelmethods.KMAccessError
   :undoc-members:
   :inherited-members:
   :show-inheritance:

