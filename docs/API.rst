API Reference
=============

A tutorial-like demo is available at :doc:`usage`.

Roughly, this library consists of following set of classes and methods:

 - diverse library of :doc:`kernel_functions`,
 - ``KernelMatrix`` documented in :doc:`kernel_matrix`,
 - container classes ``KernelSet`` and ``KernelBucket`` described in :doc:`km_collections`,
 - a library of :doc:`operations`,
 - a set of related :doc:`utilities` and
 - custom exceptions (listed below) to improve user experience and testability.



Exceptions
--------------

.. autoclass:: kernelmethods.KernelMethodsException
   :undoc-members:
   :show-inheritance:


.. autoclass:: kernelmethods.KMAccessError
   :undoc-members:
   :show-inheritance:

.. autoclass:: kernelmethods.KMNormError
   :undoc-members:
   :show-inheritance:


.. autoclass:: kernelmethods.KMSetAdditionError
   :undoc-members:
   :show-inheritance:

