Migrating from Timeloop
=======================

The following steps can be taken to migrate from Timeloop [1]_ to AccelForge:

1. Review the following set of features to ensure that AccelForge supports the features
   you need.
2. Install AccelForge using the instructions in the :doc:`./guide` page.
3. Convert the Timeloop specification to an AccelForge specifications. AccelForge
   specifications are generally simpler than Timeloop specifications.
4. For any questions on the migration process, please open an issue on the
   `AccelForge GitHub repository <https://github.com/Accelergy-Project/accelforge>`_.

Features
--------

The following are features **suppported by AccelForge, not Timeloop**:

- **Uneven Mapping**: AccelForge supports uneven mapping [2]_, an optimization
  that allows for lower resource usage for accelerators' memories.
- **Fusion**: AccelForge supports fusion, which allows multiple Einsums to be fused and
  share data without communication with off-chip memories.
- **Component Latency**: AccelForge, in addition to user-defined latency for each
  component, can use `hwcomponents <https://github.com/Accelergy-Project/hwcomponents>`_
  to model the latency of components. Timeloop requires a user-defined bandwidths for
  each component.
- **Fast unconstrained mapping**: The AccelForge mapper can find optimal mappings
  quickly, while Timeloop requires orders-of-magnitude more time and requires
  user-written constraints to reduce the search space.
- **Heterogenous Architectures**: AccelForge supports heterogeneous architectures, where
  different Einsums may be executed using different compute units. The mapper
  automatically finds the best compute unit for each Einsum.
- **Python API**: AccelForge inputs and outputs can be fully defined in Python.
- **Ease of modification and debugging**: AccelForge is written fully in Python, making
  it easy to modify and debug. Its :py:mod:`~accelforge.frontend` objects are carried
  throughout the modeling and mapping process, so modifications to the frontend can be
  accessed by the model and mapper.
- **Strong Input Validation**: AccelForge's frontend is written in Pydantic and performs
  type checking. Evaluation of expressions is performed during modeling and mapping,
  giving error reports for any invalid fields.
- **Easy-To-Modify Component Models**: AccelForge uses `hwcomponents
  <https://github.com/Accelergy-Project/hwcomponents>`_ to model the area, energy,
  latency, and leak power of components. Timeloop uses `Accelergy
  <https://github.com/Accelergy-Project/accelergy>`_. Hwcomponents is a successor to
  Accelergy, and makes it significantly easier to write custom component models.
- **Easy Installation**: AccelForge and dependencies are pip installable.

The following are features **supported by Timeloop, work-in-progress in AccelForge**:

- **Sparsity**: Support for sparse tensors.
- **Peer-to-Peer Communication**: Spatial instances at the same level of the memory
  hierarchy can share data without communicaiton with a higher-level memory.
- **Layout Support**: Support for the costs of how data is laid out in memory.
- **Skew in Model**: User-defined spatial skews that can split data between multiple
  multiple spatial instances.
- **Mapper for $\sim 1-2$ imperfectly-factorized loop levels**: The mapper can explore
  imperfect mappings for a handful of loop levels, but becomes intractible for more loop
  levels. This, as well as full imperfect mapping, is present in AccelForge, but the API
  is not yet implemented.
- **ISL-Based Model**: Timeloop supports high-fidelity modeling with ISL to generate the
  accesses for every spatial instance, while AccelForge uses a simpler and faster
  analytical model.

The following are features **work-in-progress in AccelForge, not supported by
Timeloop**:

- **Skew in Mapping**: Automatically deriving the best skew for a given workload and
  architecture.
- **Mapper for all imperfectly-factorized loop levels**: The mapper can explore
  imperfect mappings for all loop levels.
- **Einsum-Level Spatial Parallelism**: The mapper can parallelize Einsums across
  spatial instances, rather than executing them sequentially.

References
----------

.. [1] A. Parashar et al., "Timeloop: A Systematic Approach to DNN Accelerator
   Evaluation," 2019 IEEE International Symposium on Performance Analysis of Systems and
   Software (ISPASS), Madison, WI, USA, 2019, pp. 304-315, doi:
   10.1109/ISPASS.2019.00042. `Code Here <https://github.com/NVlabs/timeloop>`_
.. [2] L. Mei, P. Houshmand, V. Jain, S. Giraldo and M. Verhelst, "ZigZag:
   Enlarging Joint Architecture-Mapping Design Space Exploration for DNN Accelerators,"
   in IEEE Transactions on Computers, vol. 70, no. 8, pp. 1160-1174, 1 Aug. 2021, doi:
   10.1109/TC.2021.3059962.
