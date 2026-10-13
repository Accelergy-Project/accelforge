from accelforge.frontend.mapper.metrics import Metrics
from accelforge.util._basetypes import EvalableModel


class Model(EvalableModel):
    """Configuration for the model."""

    metrics: Metrics = Metrics.all_metrics()
    """
    Metrics to evaluate.

    If using spec to call mapper, leave this configuration as is. The mapper
    will make necessary configurations.
    """

    _share_resource_latency_above_splits: bool = False
    """
    SEE COMMENT IN _include_wind_up_latency
    """

    _include_wind_up_latency: bool = False
    """
    Latency Model
    -------------

    We assume that, for sequentialized branches, one branch fully completes all data
    movement before handing all resources to the next branch. For parallized branches,
    we assume the same (pessimistically; only needed if they have data dependencies
    between), but if the branches are repeated without switching to a sequential split,
    then the latency is only paid once across repetitions.

    Pessimisms:

    - For parallelized Einsums, we sum wind-up and wind-down latencies. However, we may
      be able to max if there exist no data dependencies
    - We do not allow pre-fetching between subsets of branches (e.g., 4 parallelized
      branches, prefetch for 4 during 3). Reservations must be placed above all or above
      only one branch.

    Optimisms:

    - We assume that wind-up and wind-down latency is only paid when switching to a
      branch. Within a branch, we assume that we can hide all latencies without any
      double buffering cost. 
    - If two branches are parallelized and have a data dependency, we do not make them
      wait for first tile to fully finish being created

    Types of latency:

    - Each resource has a steady state (#actions / throughput) latency. The latencies
      are summed into the resource's total latency. For memories with separate
      read/write ports, each port counts as a separate resource for latency purposes.

    - Wind-up latency is the additional time it takes, from the time the accelerator
      context switches to an Einsum, to the steady-state latency.

    - Wind-down latency is the additional time it takes, from the time an Einsum's
      computations finish, to the time all data movement is completed (the last output
      value is written to backing storage).
      
    Above_loop_index and below_n_loops
    ----------------------------------
    
    These are synonyms. If we have a tree with the loops:
    
    - above loop index 0 / below n loops 0
    - loop 0
    - above loop index 1 / below n loops 1
    - loop 1
    ...
    
    Within any given above loop index / below n loops slot, we can place a split and
    reservation nodes. The splits go highest, then the reservations. So if there's a
    sequential above loop index 1, then the sequential is below 1 loop in the tree, and
    it's above loop index 1 and it's also above any reservations with the same above
    loop index 1.
    
    NOTE: It is possible to put the reservation above the split but below any loops,
    which would expand its lifetime so that it can be pre-fetched. This is NOT SUPPORTED
    YET.

    Ascent and descent latencies
    ----------------------------
    
    Descent latencies are how much time does it take to propagate data down through
    reservations at this split. Ascent latencies are how much time does it take to
    propagate up.
    
    Descent and ascent latency at a given level is how much time it takes to pass
    through all components at the level of the split. For example, if data passes
    through one buffer, it'll be the latency of one read + one write of the buffer.

    Aggregating latency
    -------------------

    - Within a branch, the steady-state latency is the max steady state of all resources
      used in that branch. Data movement occurs only when the ALL involved resources are
      ready, so if we're moving data from a reservation above the branch to a
      reservation below the branch, it only occurs when the branch begins. Equivalently,
      that means that data movement between A<->B occurs when the lower-in-the-tree
      reservation is ready.

    - Within a branch, the wind-up and wind-down latency:

        - If the branch is entirely private to a single Einsum, wind-up and wind-down
          latency is the sum of the following: the time it takes for the slowest tensor
          to propagate from its highest branch-private reservation to compute, one
          compute, and the time it takes for the slowest output tensor to propagate from
          compute to its highest branch-private reservation.

        - If the branch is shared across multiple Einsums (i.e., if it has
          sub-branches), wind-up and wind-down latency is split into descent latencies
          (how much time does it take to propagate data downward from this split to the
          next split) and ascent latencies (how much time does it take to propagate data
          upward from this split to the next split). Ascent and descent latencies are
          SUMMED across all Einsums in the split, even if parallel. This is pessimistic;
          it assumes that a data dependency blocks us from moving to the next Einsum
          until the previous has propagated a value, even if no such dependency exists.
          Ascent and descent latencies are multiplied each time we switch to a
          sequential split. So, if all Einsums in the splits are parallel, we can repeat
          the split multiple times for a loop above without repaying ascent/descent
          latency. If any Einsum is sequential, then we must pay the ascent/descent
          latency for each iteration of the loop above. Ultimately, ascent/descent
          latency is multiplied by (total # iterations of all loops above iff there is a
          sequential split above this component in the tree and below the loop). then
          added to the steady-state latency of the branch it's in.

    - The total latency of a branch is the sum of the following:

        - Any ascent/descent latencies, if the branch has sub-branches
        - The max of the following:
        
            - The total latency of each sub-branch, if the branch has sub-branches
            - The max, of all resources used in the branch and sub-branches, the sum
              steady-state latency of the resource. Note that this and the above
              automatically lets slack propagate upwards because a resource having low
              latency in one branch means that it can be used to perform actions above
              the branch's split.
              
    WEIRDNESS WITH THE MODEL: Below a parallel split, any wind-up latency is assumed to
    be incurred ABOVE the split. So, we serialize the wind-up latency of below-the-split
    components with the steady-state latency of components above the split. Below a
    sequential split, wind-up latency is assumed to be incurred BELOW the split. So, we
    can parallelize wind-up latency of below-the-split components with the steady-state
    latency of above-the-split components. This may cause, in some cases, parallel to be
    slower than sequential.

    Flags
    -----

    - _share_resource_latency_above_splits: If False, then resource slack cannot
      propagate up above a split. Steady-state latency of all resources used in an
      Einsum are all paid at the Einsum's compute node.

    - _include_wind_up_latency: If False, then wind-up and wind-down latencies are not
      included in the total latency.

    Implementation notes:
    
    - We allow a single _allow_multi_einsum_parallelism spatial fanout that Einsums may
      share. The usage of this fanout is summed across parallelized Einsums, maxed
      across sequentialized Einsums.
    - Components below this fanout may NEVER be shared across Einsums. This is just to
      simplify implementation. Below the fanout, components are never shared:
      sequentially (different timesteps) or parallelized (different spatial instances).
    """
