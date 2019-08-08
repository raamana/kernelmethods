Drop-in Estimator classes
--------------------------

Besides being able to use the aforementioned ``KernelMatrix`` in SVM or another kernel machine, this library makes life even easier by providing drop-in Estimator classes directly. It's called ``KernelMachine`` and they can be dropped in place of ``sklearn.svm.SVC`` anywhere. For example:


.. code-block:: python

    from kernelmethods import KernelMachine
    km = KernelMachine(k_func=rbf)
    km.fit(X=sample_data, y=labels)
    predicted_y = km.predict(sample_data)



And if you're not sure which kernel function is optimal for your dataset, you can employ ``OptimalKernelSVR``


.. code-block:: python

    from kernelmethods import OptimalKernelSVR
    opt_km = OptimalKernelSVR(k_bucket='exhaustive')
    opt_km.fit(X=sample_data, y=labels)
    predicted_y = opt_km.predict(sample_data)


See below for the API.

**Stay tuned** for more tutorials, examples and comprehensive docs.


Kernel Machine
==============

.. autoclass:: kernelmethods.KernelMachine
   :undoc-members:
   :inherited-members:
   :show-inheritance:


OptimalKernelSVR
=================

.. autoclass:: kernelmethods.OptimalKernelSVR
   :undoc-members:
   :inherited-members:
   :show-inheritance:

