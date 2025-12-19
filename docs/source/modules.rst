fastfusion
==========

The API has the following parts:

- ``frontend`` parses and modifies input specifications. The input specification is
  handled by a top-level :py:class:`~fastfusion.frontend.specification.Specification`
  class. Each attribute of the specification is another class that represents a
  different part of the input, and having its own module in the ``frontend`` package.
  For example, `Specification.arch` is an instance of
  :py:class:`fastfusion.frontend.arch.Arch`.

- ``mapper`` maps workloads onto architectures. If you have a workload and architecture
  and would like to evaluate energy, latency, or other metrics, you can use the
  ``mapper`` package to do so. Currently, the only supported mapper is FFM.

- ``model`` MICHAEL HELP

.. toctree::
   :maxdepth: 10

   fastfusion