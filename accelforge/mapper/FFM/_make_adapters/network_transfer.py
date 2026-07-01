"""Join-time network-transfer adapter.

When two compatibilities are joined and a tensor they *share* is bound to
physically distributed storage with a *different* spatial data binding on each
side (``TensorReservation.physical_spatial_loops``), the data has to be
redistributed across the network before the consumer can read it in the layout
it expects. This module prices that redistribution.

It does so by reusing the per-spatial-loop hop-cost models in
:mod:`accelforge.model._looptree.reuse.symbolic._network` -- the same models the
symbolic looptree analysis uses -- summing the cost over only the architecture
dimensions whose binding differs, and converting the resulting hops to
energy/latency via the network component that delivers the tensor at that
storage level.

The cost computation here is intentionally side-effect free and duck-typed so it
can be unit-tested in isolation (see ``tests/network/test_network_transfer.py``)
the same way the topology models are. The plumbing that resolves architecture
components into :class:`StorageNetworkRecord`s and charges the resulting cost
into a join dataframe lives at the call sites in the FFM join path.
"""

from dataclasses import dataclass
from typing import Any

from accelforge.frontend.renames import RANK_DONT_CARE
from accelforge.frontend._workload_isl._symbolic import Irrelevant, Relevant
from accelforge.model._looptree.reuse.symbolic._network import get_topology_model


def _binding_key(loop) -> tuple:
    """Identity of a physical-spatial loop for the purpose of comparing two data
    bindings.

    Two loops describe the same binding iff they fan the same rank over the same
    architecture dimension. Tile *sizes* don't change which physical instance
    holds which datum, so they're excluded from the comparison.
    """
    return (loop.spatial_dim, loop.rank_name)


def _relevancy(loop):
    """Relevancy of a physical-spatial loop's rank to the tensor being moved.

    A loop that binds a real rank distributes distinct data to each instance
    along the dimension (unicast -> :class:`Relevant`). A ``RANK_DONT_CARE``
    binding replicates the same data (multicast -> :class:`Irrelevant`).
    """
    if loop is None or loop.rank_name == RANK_DONT_CARE:
        return Irrelevant()
    return Relevant(loop.rank_name)


def differing_physical_dims(left, right) -> list:
    """Return, per architecture dimension, the loop describing the binding that
    must be *delivered* when redistributing a tensor from the ``left`` data
    binding to the ``right`` one.

    A dimension is charged iff the two sides bind it differently. The consumer
    (``right``) loop is preferred to describe the post-transfer binding; if a
    dimension exists only on the producer side it is still charged (the data has
    to be gathered off that distribution).
    """
    left_by_dim = {l.spatial_dim: l for l in left.physical_spatial_loops}
    right_by_dim = {l.spatial_dim: l for l in right.physical_spatial_loops}
    out = []
    for dim in sorted(set(left_by_dim) | set(right_by_dim), key=lambda d: (d is None, d)):
        lk = _binding_key(left_by_dim[dim]) if dim in left_by_dim else None
        rk = _binding_key(right_by_dim[dim]) if dim in right_by_dim else None
        if lk != rk:
            out.append(right_by_dim.get(dim) or left_by_dim[dim])
    return out


@dataclass
class StorageNetworkRecord:
    """Everything needed to price a redistribution at one storage level.

    Precomputed once (from the flattened architecture) per storage
    ``resource_name`` so the join path never has to touch the arch tree.
    """

    topology: Any
    """The delivering network's :class:`TopologySpec`."""
    joules_per_hop: float
    """Dynamic energy of one ``hop`` action on the delivering network."""
    seconds_per_hop: float = 0.0
    """Latency of one ``hop`` action on the delivering network (0 if unknown)."""
    bits_per_action: float | None = None
    """Network ``bits_per_action`` (``None`` -> one action carries one value)."""
    src_component: Any = None
    """The storage component sourcing the data, queried for physical
    fanout/stride. May be ``None`` for a non-distributed stand-in."""
    memory_size_bits: float = 1.0
    """Backing-memory capacity in bits, used to convert the fractional tensor
    size column back to raw bits."""


@dataclass
class TransferCost:
    """Result of pricing one shared tensor's redistribution. ``energy`` and
    ``latency`` may be scalars or per-row (numpy/pandas) values, matching the
    ``volume`` that was passed in."""

    energy: Any = 0
    latency: Any = 0
    total_hops: Any = 0
    max_hops: Any = 0
    max_traffic: Any = 0


def transfer_cost(
    record: StorageNetworkRecord,
    left_res,
    right_res,
    volume,
) -> TransferCost | None:
    """Price moving the shared tensor from the ``left_res`` binding to the
    ``right_res`` binding over ``record``'s network.

    ``volume`` is the data moved per destination, *in network actions* (the same
    unit ``_network.py`` expects). It may be a scalar or a vectorized
    numpy/pandas value; the returned cost has the same shape.

    Returns ``None`` when nothing needs to move (the bindings already match).
    """
    dims = differing_physical_dims(left_res, right_res)
    if not dims:
        return None

    model = get_topology_model(record.topology)
    total_hops = 0
    max_hops = 0
    max_traffic = 0
    for loop in dims:
        dim = loop.spatial_dim
        if record.src_component is not None:
            shape_repeats = record.src_component._get_physical_fanout_along(dim)
        else:
            shape_repeats = 1
        cost = model.per_loop_transfer_cost(
            _relevancy(loop),
            # The redistribution delivers to each physical instance along the
            # dimension; the intra-instance stride is not modeled here, so the
            # nodes below contribute a unit fanout.
            shape_repeats=shape_repeats,
            last_fanout=1,
            volume=volume,
            src_component=record.src_component,
            dim_name=dim,
        )
        # Hops accumulate across the (serial) per-dimension transfers, mirroring
        # NetworkAnalyzer.accumulate_max_hops; the busiest single link is the max.
        total_hops += cost.total_cost
        max_hops += cost.max_hops
        max_traffic = _elementwise_max(max_traffic, cost.max_traffic)

    return TransferCost(
        energy=total_hops * record.joules_per_hop,
        latency=max_hops * record.seconds_per_hop,
        total_hops=total_hops,
        max_hops=max_hops,
        max_traffic=max_traffic,
    )


class NetworkTransferAdapter:
    """Holds the per-storage cost records and charges layout-change transfers
    into a join dataframe.

    Built once per join (see :meth:`build`) from the flattened architectures, so
    the join path never touches the arch tree. Small and picklable, so it is safe
    to pass through the ``joblib`` fan-out in the join loop.
    """

    def __init__(self, storage_records: dict, tensor_full_bits: dict | None = None):
        # resource_name -> StorageNetworkRecord
        self.storage_records = storage_records
        # tensor_name -> full (untiled) size in bits; a finite fallback used when a
        # backing memory has unbounded (inf) size and the fractional size column is
        # degenerate.
        self.tensor_full_bits = tensor_full_bits or {}

    @classmethod
    def build(cls, pmappings) -> "NetworkTransferAdapter":
        """Construct the adapter from a :class:`MultiEinsumPmappings`' flattened
        architectures and workload. Side-effect free; resolves every physically
        distributed storage to its nearest network and that network's per-hop
        energy/latency/topology."""
        from accelforge.frontend import arch

        tensor_full_bits = {}
        for einsum in pmappings.spec.workload.einsums:
            for ta in einsum.tensor_accesses:
                if ta.name in tensor_full_bits:
                    continue
                try:
                    n_values = 1
                    for r in ta.ranks:
                        n_values *= einsum.rank_sizes[r]
                    tensor_full_bits[ta.name] = n_values * ta.bits_per_value
                except Exception:
                    pass

        records: dict = {}
        for leaves in pmappings.flattened_arches.values():
            indexed = list(enumerate(leaves))
            networks = [(i, n) for i, n in indexed if isinstance(n, arch.Network)]
            if not networks:
                continue
            for mi, mem in indexed:
                if not isinstance(mem, arch.Memory) or mem.name in records:
                    continue
                if not getattr(mem, "_is_distributed", lambda: False)():
                    continue
                # Charge the network physically nearest this storage.
                _, net = min(networks, key=lambda x: abs(x[0] - mi))
                try:
                    hop = net.actions["hop"]
                    joules_per_hop = float(hop.energy)
                    seconds_per_hop = float(getattr(hop, "latency", 0) or 0)
                except (TypeError, ValueError, KeyError):
                    # Unevaluated/symbolic action cost -> can't charge numerically.
                    continue
                records[mem.name] = StorageNetworkRecord(
                    topology=net.topology,
                    joules_per_hop=joules_per_hop,
                    seconds_per_hop=seconds_per_hop,
                    bits_per_action=net.bits_per_action,
                    src_component=mem,
                    memory_size_bits=mem.size,
                )
        return cls(records, tensor_full_bits)

    def _record_for(self, left_res, right_res):
        return self.storage_records.get(
            right_res.resource_name
        ) or self.storage_records.get(left_res.resource_name)

    def charge_dataframe(self, df, tensor_name, left_res, right_res) -> None:
        """Add the redistribution cost of ``tensor_name`` (moving from the
        ``left_res`` binding to the ``right_res`` binding) into ``df``'s objective
        columns, in place. A no-op when the bindings match, no network is known,
        or the tensor's size column is absent."""
        if len(df) == 0:
            return
        record = self._record_for(left_res, right_res)
        if record is None:
            return

        # Local imports keep the pure cost core (transfer_cost) importable without
        # pulling in the dataframe-convention module.
        from accelforge.mapper.FFM._pareto_df.df_convention import (
            add_to_col,
            tensor2col,
        )

        import math

        # The tensor size column holds a *fraction* of the backing memory; recover
        # raw bits, then convert to network actions exactly as
        # NetworkAnalyzer._get_data_volume does. When the backing memory is
        # unbounded (inf size) the fraction is degenerate, so fall back to the
        # tensor's full (untiled) size in bits.
        size_col = tensor2col(tensor_name)
        if math.isfinite(record.memory_size_bits) and size_col in df.columns:
            bits = df[size_col] * record.memory_size_bits
        else:
            bits = self.tensor_full_bits.get(tensor_name)
            if bits is None:
                return

        volume = bits / record.bits_per_action if record.bits_per_action else bits

        cost = transfer_cost(record, left_res, right_res, volume)
        if cost is None:
            return

        # Charge dynamic movement energy and (additively, as a serialized step)
        # latency, onto whichever objective columns are present.
        for col in ("Total<SEP>dynamic_energy", "Total<SEP>energy"):
            if col in df.columns:
                add_to_col(df, col, cost.energy)
                break
        if "Total<SEP>latency" in df.columns:
            add_to_col(df, "Total<SEP>latency", cost.latency)


def _elementwise_max(a, b):
    """max() that also works elementwise over numpy/pandas values."""
    try:
        return a.where(a >= b, b)  # pandas Series
    except AttributeError:
        pass
    try:
        import numpy as np

        if hasattr(a, "shape") or hasattr(b, "shape"):
            return np.maximum(a, b)
    except ImportError:
        pass
    return a if a >= b else b
