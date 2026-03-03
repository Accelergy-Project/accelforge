import logging
from copy import copy, deepcopy
from uuid import uuid4

import pandas as pd

from accelforge.frontend import arch
from accelforge.frontend.arch import Memory
from accelforge.frontend.mapping.mapping import MappingNodeWithChildren
from accelforge.frontend.renames import EinsumName, TensorName
from accelforge.frontend.spec import Mapping, Spec
from accelforge.frontend.mapping import (
    Compute,
    Loop,
    Reservation,
    Spatial,
    Split,
    Nested,
    NodeList,
    Temporal,
    TensorHolder,
)
from accelforge.frontend.workload import Workload
from accelforge.frontend._workload_isl._symbolic import (
    get_stride_and_halo,
    get_rank_variable_relevancy,
)
from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.symbol_relations import (
    get_initial_delta_choices,
)
from accelforge.mapper.FFM._pareto_df.df_convention import col_used_in_joining, col2nameloop

logger = logging.getLogger(__name__)


class InvalidMappingError(Exception):
    """Raised when a mapping violates architecture constraints."""
    pass


def evaluate_mapping(
    spec: Spec,
    flattened_arches: dict[(EinsumName, str), list[arch.Leaf]] | None = None,
    evaluated_specs: dict[EinsumName, Spec] | None = None,
    validate: bool = True,
):
    """
    Evaluate a mapping.

    Parameters
    ----------
    spec:
        The specification of architecture, workload, and mapping.
    flattened_arches:
        A dictionary of (EinsumName, Compute Name) to lists of architecture nodes. These
        contain the evaluated and flattened architecture node for that particular Einsum
        and compute combination. If provided, then these will be used instead of
        re-parsing the architecture.
    evaluated_specs:
        A dictionary of Einsum names to evaluated specifications. These contain the evaluated
        specification for that particular Einsum. If provided, then these will be used
        instead of re-parsing the specification.
    """
    from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
    from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
        PmappingDataframe,
    )
    from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
    from accelforge.mapper.FFM._join_pmappings.join_pmappings import (
        clean_compress_and_join_pmappings,
    )
    from accelforge.mapper.FFM.pmappings import MultiEinsumPmappings
    from accelforge.mapper.FFM._make_pmappings.make_pmappings import (
        get_rank_variable_bounds_for_all_einsums,
    )
    from accelforge.model.run_model import (
        run_model,
    )
    from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.make_tile_shapes import (
        _calculate_iterations_and_rank_columns,
        _clean_energy_columns,
    )
    from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job

    assert (evaluated_specs is not None) == (
        flattened_arches is not None
    ), f"Provide either flattened_arches or evaluated_specs, not both."

    original_job = Job(
        metrics=spec.model.metrics,
        rank_variable_bounds=get_rank_variable_bounds_for_all_einsums(spec),
        spec_one_einsum=spec,
    )

    einsum2pmappings = {}
    pmapping_objects = {}
    einsum2jobs = {}
    s = (
        "Spec must not be evaluated before evaluating a mapping. Was "
        "this spec returned by spec.calculate_component_area_energy_latency_leak()?"
    )

    needs_reservations = not bool(spec.mapping.get_nodes_of_type(Reservation))

    fusable_tensors = spec.workload.tensor_names_used_in_multiple_einsums
    stride_and_halo = get_stride_and_halo(spec.workload)

    assert not getattr(spec, "_evaluated", False), s
    for pmapping in _split_mapping_to_pmappings(spec.mapping, spec.workload):
        einsum_name = pmapping.nodes[-1].einsum
        compute_name = pmapping.nodes[-1].component
        pmapping_id = uuid4()
        job = copy(original_job)

        if flattened_arches is not None:
            flattened_arch = flattened_arches[(einsum_name, compute_name)]
            cur_spec = evaluated_specs[einsum_name]

        else:
            cur_spec = spec.calculate_component_area_energy_latency_leak(
                einsum_name=einsum_name,
                area=False,
            )
            flattened_arch = cur_spec._get_flattened_architecture(
                compute_node=pmapping.nodes[-1].component
            )

        job.spec_one_einsum = cur_spec
        job.einsum_name = pmapping.nodes[-1].einsum
        job.stride_and_halo = stride_and_halo
        # spec, not cur_spec, becuase cur_spec only has one einsum and the delta choices
        # depend on >1 Einsums
        job.initial_delta_choices = get_initial_delta_choices(
            job.einsum_name, spec.workload
        )
        pmapping.split_reservations()
        pmapping.split_loop_with_multiple_rank_variables(job.einsum_name)
        pmapping.split_tensor_holders_with_multiple_tensors()
        _add_backing_to_tensor_holders(pmapping)

        job.mapping = pmapping
        job.tensor_to_relevancy = {
            tensor: get_rank_variable_relevancy(
                job.spec_one_einsum.workload.einsums[job.einsum_name], tensor
            )
            for tensor in job.spec_one_einsum.workload.einsums[
                job.einsum_name
            ].tensor_names
        }
        pmapping.clear_irrelevant_reservations(set(job.tensor_to_relevancy))

        einsum2jobs[job.einsum_name] = job

        job.flattened_arch = flattened_arch
        job.memories_track_all = [
            m.name for m in flattened_arch if isinstance(m, Memory)
        ]

        job.fusable_tensors = fusable_tensors & set(job.tensor_to_relevancy)
        einsum = cur_spec.workload.einsums[job.einsum_name]
        rank_variable_to_ranks = {
            t.name: t.rank_variable2ranks for t in einsum.tensor_accesses
        }

        _, df, per_memory_usage_df, spatial_usage_df, tensor2mapping, _ = run_model(
            job, add_reservations=needs_reservations
        )

        if validate:
            _validate_mapping(
                df, per_memory_usage_df, spatial_usage_df,
                job, flattened_arch,
            )

        # Calculate iteration counts and rank columns
        _clean_energy_columns(df, job.metrics)
        _calculate_iterations_and_rank_columns(
            job.mapping.nodes, job, df, job.rank_variable_bounds
        )
        compatibility = Compatibility.from_mapping(
            job.mapping,
            job.fusable_tensors,
            rank_variable_to_ranks,
        )
        symbol_renames, compatibility = compatibility.make_fused_loop_symbols(
            einsum_name
        )
        for k, v in symbol_renames.items():
            df[v] = df.pop(k)

        new_df = {}
        for key, value in df.items():
            if not col_used_in_joining(key):
                key = f"{job.einsum_name}<SEP>{key}"
            # Want usage both for joining & for per-einsum info
            if key.startswith("usage<SEP>"):
                new_df[f"{job.einsum_name}<SEP>{key}"] = value
            new_df[key] = value
        df = new_df
        df[f"{job.einsum_name}<SEP>mapping"] = pmapping_id

        einsum2pmappings[job.einsum_name] = [
            PmappingGroup(
                compatibility,
                PmappingDataframe(
                    data=pd.DataFrame(df, columns=df.keys(), index=[0]),
                    n_total_pmappings=1,
                    n_valid_pmappings=1,
                    ignored_resources=set(),
                ),
            )
        ]
        pmapping_objects[job.einsum_name] = {pmapping_id: job.mapping}

    # Restore the original order
    einsum2pmappings = {
        einsum_name: einsum2pmappings[einsum_name]
        for einsum_name in spec.workload.einsum_names
        if einsum_name in einsum2pmappings
    }

    return clean_compress_and_join_pmappings(
        pmappings=MultiEinsumPmappings(
            spec=spec,
            einsum2pmappings=einsum2pmappings,
            pmapping_objects=pmapping_objects,
            einsum2jobs=einsum2jobs,
            can_combine_multiple_runs=False,
            einsums_with_pmappings_generated=set(spec.workload.einsum_names),
            flattened_arches=flattened_arches,
            evaluated_specs=evaluated_specs,
        ),
        metrics=spec.model.metrics,
        print_progress=False,
    )


def _validate_mapping(
    df: dict,
    per_memory_usage_df: dict,
    spatial_usage_df: dict,
    job,
    flattened_arch: list,
):
    """Validate a mapping against architecture constraints.

    Checks memory capacity (including metadata), spatial fanout, and
    architecture constraints. Collects all violations into a single
    ``InvalidMappingError``.
    """
    from accelforge.mapper.FFM._make_pmappings.contraints.constraints import (
        get_constraints,
    )
    from accelforge.util._setexpressions import InvertibleSet

    errors = []
    TOL = 1e-6

    # --- Check 1: Memory capacity (reservation columns > 1.0) ---
    # Reservation columns encode per-level, per-nloop occupancy ratios.
    seen_memories = set()
    for key, value in df.items():
        parsed = col2nameloop(key)
        if parsed is not None:
            name, _ = parsed
            if value > 1.0 + TOL and name not in seen_memories:
                seen_memories.add(name)
                errors.append(
                    f"Memory '{name}' exceeds capacity: "
                    f"usage={value:.4f} "
                    f"({100 * value:.1f}% of capacity, includes metadata)"
                )

    # Also check total per-memory usage (includes all tensors summed).
    for key, value in per_memory_usage_df.items():
        if value > 1.0 + TOL:
            # key format: "usage<SEP>memory<SEP>{memory_name}"
            parts = key.split("<SEP>")
            memory_name = parts[2] if len(parts) >= 3 else key
            if memory_name not in seen_memories:
                seen_memories.add(memory_name)
                errors.append(
                    f"Memory '{memory_name}' total usage exceeds capacity: "
                    f"usage={value:.4f} "
                    f"({100 * value:.1f}% of capacity, includes metadata)"
                )

    # --- Check 2: Spatial fanout ---
    for key, value in spatial_usage_df.items():
        if value > 1.0 + TOL:
            # key format: "usage<SEP>spatial<SEP>{component}<SEP>{dim}"
            parts = key.split("<SEP>")
            component = parts[2] if len(parts) >= 4 else "unknown"
            dim = parts[3] if len(parts) >= 4 else "unknown"
            errors.append(
                f"Spatial fanout exceeded for '{component}' dimension '{dim}': "
                f"usage={value:.4f} "
                f"({100 * value:.1f}% of available instances)"
            )

    # --- Check 3: Architecture constraints (best-effort) ---
    try:
        _check_arch_constraints(
            errors, job, flattened_arch, get_constraints, InvertibleSet,
        )
    except Exception:
        logger.debug(
            "Skipping architecture constraint check (could not evaluate)",
            exc_info=True,
        )

    if errors:
        raise InvalidMappingError(
            "Invalid mapping:\n  - " + "\n  - ".join(errors)
        )


def _check_arch_constraints(errors, job, flattened_arch, get_constraints, InvertibleSet):
    """Evaluate tile_shape and loop_bounds constraints from the architecture."""
    import numpy as np

    mapping_nodes = list(job.mapping.nodes)
    einsum_name = job.einsum_name

    # Build symbol_table: component_name -> InvertibleSet of stored tensors.
    all_tensors = frozenset(job.tensor_to_relevancy.keys())
    symbol_table = {}
    for node in mapping_nodes:
        if isinstance(node, TensorHolder):
            symbol_table[node.component] = InvertibleSet(
                instance=frozenset(node.tensors),
                full_space=all_tensors,
                space_type=str,
            )

    _, constraints = get_constraints(
        flattened_arch, list(mapping_nodes), symbol_table,
        einsum_name, job.tensor_to_relevancy,
    )

    loops = [n for n in mapping_nodes if isinstance(n, Loop)]
    if not loops:
        return

    constraints.set_loop_indices(mapping_nodes)

    # Extract concrete tile sizes from each loop.
    tile_sizes = []
    all_concrete = True
    for loop in loops:
        ts = loop.tile_pattern.tile_shape
        if isinstance(ts, (int, float)):
            tile_sizes.append(int(ts))
        else:
            all_concrete = False
            break

    if not all_concrete:
        return

    tile_array = np.array([tile_sizes], dtype=np.float64)
    complete_indices = list(range(len(loops)))

    # Tile shape constraints.
    for c in constraints.tile_shape_constraints:
        if not c.target_mapping_nodes:
            continue
        indices = c._target_loop_indices
        result = c(set(range(len(loops))), tile_array[:, indices])
        if hasattr(result, '__len__'):
            violated = not result[0]
        else:
            violated = not result
        if violated:
            errors.append(f"Tile shape constraint violated: {c.pretty_str()}")

    # Loop bounds constraints.
    for c in constraints.loop_bounds_constraints:
        if not c.target_mapping_nodes:
            continue
        indices = c._target_loop_indices
        result = c(set(range(len(loops))), tile_array[:, indices])
        if hasattr(result, '__len__'):
            violated = not result[0]
        else:
            violated = not result
        if violated:
            errors.append(f"Loop bounds constraint violated: {c.pretty_str()}")


def _add_backing_to_tensor_holders(pmapping: Mapping):
    seen_tensors = set()
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            new_tensors = set(node.tensors) - seen_tensors
            node._backing = new_tensors
            seen_tensors.update(new_tensors)


def _split_mapping_worker(node: MappingNodeWithChildren):
    if isinstance(node, Split):
        for subnodes in node.nodes:
            yield from _split_mapping_worker(subnodes)
        return

    assert isinstance(node, Nested), "BUG"

    for n in node.nodes[:-1]:
        assert not isinstance(n, MappingNodeWithChildren), "BUG"

    if not isinstance(node.nodes[-1], MappingNodeWithChildren):
        yield node.nodes
        return

    for subnodes in _split_mapping_worker(node.nodes[-1]):
        yield node.nodes[:-1] + subnodes


def _split_mapping_to_pmappings(mapping: Mapping, workload: Workload):
    """
    A DFS-like algorithm to split a mapping into pmappings at Split nodes.

    DFS has to be modified because the tree has list of nodes for nested nodes
    instead of links to children.
    """
    for nodes in _split_mapping_worker(mapping):
        mapping = Mapping(nodes=deepcopy(nodes))
        _remove_storage_of_unrelevant_tensors(mapping, workload)
        yield mapping


def _remove_storage_of_unrelevant_tensors(pmapping: Mapping, workload: Workload):
    """
    Remove tensors from Storage nodes that are not relevant to the Einsum being
    mapped.
    """
    einsum_name = pmapping.nodes[-1].einsum
    einsum = workload.einsums[einsum_name]
    relevant_tensors = set(t.name for t in einsum.tensor_accesses)

    new_nodes = []
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            node.tensors = [t for t in node.tensors if t in relevant_tensors]
            if node.tensors:
                new_nodes.append(node)
        else:
            new_nodes.append(node)

    pmapping.nodes = new_nodes
