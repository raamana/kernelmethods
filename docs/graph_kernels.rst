Graph kernels (coming soon)
----------------------------

The design of ``kernelmethods`` allows for easy implementation to accept any input data types, as that only matters in the definition of the kernel function which are modularized out of the remaining parts of library.

For example, a simple graph kernel can be implemented by measuring the similarity in the `degree distributions <https://en.wikipedia.org/wiki/Degree_distribution>`_ of two graphs. An example implementation is available at `NetworkX <https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_degree_histogram.html>`_.

Let's say that method is called ``degree_distr()``. Then a simple graph kernel can easily be defined in few lines via:


.. code-block:: python

    from kernelmethods.base import KernelFromCallable
    from kernelmethods.graphs import degree_distr # Note: NotImplemented
    import numpy as np

    def degree_distr_similarity(graph_one, graph_two):

        return np.dot(degree_distr(graph_one), degree_distr(graph_two))

    graph_kernel = KernelFromCallable(degree_distr_similarity)


Or one could also turn this into a full-fledged ``DegreeDistrKernel`` class, while reusing a lot of boilerplate from ``kernelmethods.base`` mimicking numerical or categorical kernels already defined.

Once a ``DegreeDistrKernel`` is available, we can leverage the full functionality of the ``kernelmethods`` to the user domain/application.
