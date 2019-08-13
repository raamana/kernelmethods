Collection of kernel matrices
==========================================

When dealing multiple kernel matrices e.g. as part of multiple kernel learning (MKL), a number of validation and sanity checks need to be performed. Some of these checks include ensuring compatible size of the kernel matrices (KMs), as well as knowing these matrices are generated from the same sample. We refer to such collection of KMs a ``KernelSet``. Moreover, accessing a subset of these KMs e.g. filtered by some metric is often necessary while trying to optimize algorithms like MKL. To serve as candidates for optimization, it is often necessary to *sample* and generate a large number of KMs, here referred to as a *bucket*. The ``KernelSet`` and ``KernelBucket`` make these tasks easy and extensible while keeping their rich annotations and structure (meta-data etc).


Kernel Set
-----------------------------

.. autoclass:: kernelmethods.KernelSet
   :undoc-members:
   :inherited-members:
   :show-inheritance:


Kernel Bucket
-----------------------------

.. autoclass:: kernelmethods.KernelBucket
   :undoc-members:
   :inherited-members:
   :show-inheritance:


Exceptions
-----------------------------

.. autoclass:: kernelmethods.KMSetAdditionError
   :undoc-members:
   :inherited-members:
   :show-inheritance:

