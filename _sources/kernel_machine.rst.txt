Drop-in Estimator classes
--------------------------

Besides being able to use the aforementioned ``KernelMatrix`` in SVM or another kernel machine, this library makes life even easier by providing drop-in Estimator classes directly for use in scikit-learn. This interface is called ``KernelMachine`` and it can be dropped in place of ``sklearn.svm.SVC`` or another kernel machine of user choice anywhere an sklearn Estimator can be used. For example:


.. code-block:: python

    from kernelmethods import KernelMachine
    km = KernelMachine(k_func=rbf)
    km.fit(X=sample_data, y=labels)
    predicted_y = km.predict(sample_data)



And if you're not sure which kernel function is optimal for your dataset, you can simply employ the ``OptimalKernelSVR`` which evaluates a large ``KernelBucket`` and trains the ``SVR`` estimator with the most optimal kernel for your sample. Using it is as easy as:


.. code-block:: python

    from kernelmethods import OptimalKernelSVR
    opt_km = OptimalKernelSVR(k_bucket='exhaustive')
    opt_km.fit(X=sample_data, y=labels)
    predicted_y = opt_km.predict(sample_data)

|

See below for their API. **Stay tuned** for more tutorials, examples and comprehensive docs.


Kernel Machine (API)
=====================

.. autoclass:: kernelmethods.KernelMachine
   :undoc-members:
   :inherited-members:
   :show-inheritance:


OptimalKernelSVR (API)
=======================

.. autoclass:: kernelmethods.OptimalKernelSVR
   :undoc-members:
   :inherited-members:
   :show-inheritance:

