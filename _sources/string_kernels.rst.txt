String kernels (coming soon)
-----------------------------

The design of ``kernelmethods`` allows for easy implementation to accept any input data types, as that only matters in the definition of the kernel function which are modularized out of the remaining parts of library.

For example, to implement mismatch kernels for protein sequence data, it is simply a matter of implementing `k-mer <https://en.wikipedia.org/wiki/K-mer>`_ feature extraction tool e.g. ``kmer_spectrum()``. Then a simple mismatch kernel can easily be defined in few lines via:


.. code-block:: python

    from kernelmethods.base import KernelFromCallable
    from kernelmethods.strings import kmer_spectrum # Note: NotImplemented
    import numpy as np

    def kmer_similarity(string_one, string_two):

        return np.dot(kmer_spectrum(string_one), kmer_spectrum(string_two))

    mismatch_kernel = KernelFromCallable(kmer_similarity)


Or one could also turn this into a full-fledged ``StringKernel`` or ``MismatchStringKernel`` classes, while reusing a lot of boilerplate from ``kernelmethods.base`` mimicking numerical or categorical kernels already defined.

Once a ``MismatchStringKernel`` is available, we can leverage the full functionality of the ``kernelmethods`` to the user domain/application.
