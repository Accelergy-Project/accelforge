import copy
from collections import defaultdict
import itertools
from typing import Any, Iterator, List
import uuid

from tqdm import tqdm

import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import (
    Compute,
    Iteration,
    Mapping,
    MappingNode,
    TensorHolder,
    Temporal,
)
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload._isl import get_rank_variable_bounds
from fastfusion.frontend.workload._symbolic import get_rank_variable_relevancy, get_stride_and_halo_of_einsum
from fastfusion.frontend.workload.workload import (
    Einsum,
    EinsumName,
    RankVariableName,
    Workload,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.dataflow_generator import (
    get_tensor_choices,
)
from fastfusion.mapper.FFM._make_pmappings.tile_shape_exploration import (
    get_initial_delta_choices,
)
from fastfusion.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
    get_constraints,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.loop_generator import (
    insert_temporal_loops,
    insert_spatial_loops,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import (
    Job,
    SameEinsumJobs
)
from fastfusion.util.setexpressions import eval_set_expression


def unpack_loops_to_rank_variables(mapping: List[MappingNode]):
    mapping_new = []
    for node in mapping:
        if not isinstance(node, Iteration) or not isinstance(node.rank_variable, set):
            mapping_new.append(node)
            continue

        for r in sorted(node.rank_variable):
            mapping_new.append(
                type(node)(
                    rank_variable=r,
                    **node.model_dump(exclude={"rank_variable"}),
                )
            )
    return mapping_new


def label_fused_loops(mapping: List[MappingNode]):
    assert_proper_fusion_labeling(mapping, check_loops=False)
    last_backer = None
    for i, node in enumerate(mapping):
        if isinstance(node, TensorHolder) and node._backing:
            last_backer = i
    if last_backer is None:
        raise ValueError(
            f"No backing TensorHolder found in mapping {", ".join(m.compact_str() for m in mapping)}"
        )

    for i, node in enumerate(mapping):
        if isinstance(node, Iteration):
            node._fused = i < last_backer
    return mapping


# =================================================================================================
# Iterate over mappings
# =================================================================================================
def place_missing_temporal_loops(mapping: List[MappingNode], einsum: Einsum):
    # If any rank variables are missing, add them as high as possible.
    rank_variables = einsum.rank_variables
    for m in mapping:
        if isinstance(m, Temporal) and not m._fused:
            rank_variables.discard(m.rank_variable)

    # insert_point = 0
    # while insert_point < len(mapping) and not isinstance(mapping[insert_point], Temporal):
    #     insert_point += 1
    # Insert point: Right under the last backing
    for i in range(len(mapping) - 1, -1, -1):
        if isinstance(mapping[i], TensorHolder) and mapping[i]._backing:
            insert_point = i + 1
            break

    temporals = [Temporal(rank_variable=r) for r in sorted(rank_variables)]

    if insert_point == len(mapping):
        mapping.extend(temporals)
    else:
        for t in temporals:
            mapping.insert(insert_point, t)


def pad_with_bottom_loops(mapping: list[MappingNode], einsum: Einsum):
    rank_variables = einsum.rank_variables
    rank_var_to_count = defaultdict(lambda: 0)
    for node in mapping:
        if isinstance(node, Temporal):
            rank_var_to_count[node.rank_variable] += 1

    for rank_var in rank_variables:
        if rank_var_to_count[rank_var] < 2:
            mapping.append(Temporal(rank_variable=rank_var))


def timeloop_style_even(mapping: list[MappingNode]):
    # Iterate through the mapping. If there are >2 TensorHolder nodes for the same
    # memory, move all below the 2nd to the same level as the 2nd.
    mapping = copy.deepcopy(mapping)
    memory2indices = defaultdict(list)
    i = 0
    while i < len(mapping):
        node = mapping[i]
        if not isinstance(mapping[i], TensorHolder):
            i += 1
            continue
        node: TensorHolder
        seen = memory2indices[node.component]
        mapping[i]._lower = False # Lowering might re-uneven the reservationsxs

        if len(seen) <= 1:
            seen.append(i)
        else:
            mapping.insert(seen[-1] + 1, mapping.pop(i))
        i += 1
    return mapping


def get_ranks_with_tile_pattern(producer_name: EinsumName, workload: Workload):
    initial_choices = get_initial_delta_choices(producer_name, workload)
    return {
        rank_var
        for rank_var in workload.einsums[producer_name].rank_variables
        if len(initial_choices[rank_var]) > 1
    }


def max_fused_loops(mapping: Mapping, max_fused_loops: int):
    fused_loops = [
        i
        for i, node in enumerate(mapping)
        if isinstance(node, Iteration)
        and node._fused
    ]

    if len(fused_loops) <= max_fused_loops:
        yield mapping
        return

    for choice in itertools.combinations(fused_loops, max_fused_loops):
        to_remove = set(fused_loops) - set(choice)
        mapping_new = list(mapping)
        for f in sorted(to_remove, reverse=True):
            mapping_new.pop(f)
        yield mapping_new

def assert_proper_fusion_labeling(mapping: list[MappingNode], check_loops: bool = True):
    tensors = set()
    for i, t in enumerate(mapping):
        if not isinstance(t, TensorHolder):
            continue
        
        new = set(t.tensors) - tensors
        
        if new and check_loops:
            for j in range(i):
                if isinstance(mapping[j], Iteration):
                    assert mapping[j]._fused, f"Node {j} is not fused in {' '.join(m.compact_str() for m in mapping)}"
        assert t._backing == new, f"Node {i} backing missing {new - t._backing} in {' '.join(m.compact_str() for m in mapping)}"
        tensors.update(new)
        tensors.update(t.tensors)

def iterate_mappings_no_constraints(
    spec: Specification,
    einsum_name: str,
    arch_flattened: list[arch.Leaf],
    rank_variable_bounds: dict[RankVariableName, int],
    except_from_imperfect: set,
):
    first_memory = None
    for node in arch_flattened:
        if isinstance(node, arch.Memory):
            first_memory = node
            break
    if first_memory is None:
        raise ValueError("No memory found in architecture")

    ranks_with_tile_pattern = get_ranks_with_tile_pattern(einsum_name, spec.workload)

    symbol_table = spec.workload.get_constraint_symbol_table(einsum_name, spec.renames)
    einsum = spec.workload.einsums[einsum_name]
    for mapping, symbol_table in get_tensor_choices(
        einsum_name, arch_flattened, symbol_table, spec
    ):
        mapping = copy.deepcopy(mapping)
        for mapping in insert_temporal_loops(
            mapping,
            einsum,
            first_memory,
            rank_variable_bounds,
            ranks_with_tile_pattern,
            spec.workload,
            except_from_imperfect,
        ):
            mapping = copy.deepcopy(mapping)
            insert_spatial_loops(mapping, einsum, arch_flattened)
            mapping = unpack_loops_to_rank_variables(mapping)
            if spec.mapper.ffm.timeloop_style_even:
                mapping = timeloop_style_even(mapping)
                
            place_missing_temporal_loops(mapping, einsum)
            label_fused_loops(mapping)
            for mapping2 in max_fused_loops(
                mapping,
                spec.mapper.ffm.max_fused_loops,
            ):
                assert_proper_fusion_labeling(mapping2)
                yield mapping2, symbol_table

def iterate_mappings_constraints(
    spec: Specification,
    einsum_names: list[str] | str | None = None,
    arch_flattened: list[arch.Leaf] | None = None,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
    except_from_imperfect: set = set(),
) -> Iterator[tuple[Mapping, MappingConstraints, dict[str, str]]]:
    if arch_flattened is None:
        arch_flattened = spec.get_flattened_architecture()
    compute_name = arch_flattened[-1].name

    if isinstance(einsum_names, str):
        einsum_names = [einsum_names]
    if einsum_names is None:
        einsum_names = [e.name for e in spec.workload.einsums]

    if rank_variable_bounds is None:
        rank_variable_bounds = get_rank_variable_bounds(spec, einsum_names)

    for einsum_name in einsum_names:
        for mapping, symbol_table in iterate_mappings_no_constraints(
            spec,
            einsum_name,
            arch_flattened,
            rank_variable_bounds,
            except_from_imperfect,
        ):
            # MAPPING MUST NOT BE MODIFIED AFTER THIS POINT
            mapping, constraints = get_constraints(
                arch_flattened, mapping, symbol_table, einsum_name
            )
            mapping.append(Compute(einsum=einsum_name, compute=compute_name, component_object=arch_flattened[-1]))
            mapping = Mapping(nodes=[copy.copy(n) for n in mapping])
            yield mapping, constraints, symbol_table


# =================================================================================================
# Make sims
# =================================================================================================



def parse_flattened_arch(
    job: Job,
    symbol_table: dict[str, str],
) -> list[arch.Leaf]:
    flattened_arch = [n for n in job.flattened_arch]

    tensor_names = job.spec.workload.einsums[job.einsum_name].tensor_names

    def parse_tensor2bits(to_parse: dict[str, Any], location: str, symbol_table: dict[str, str]) -> dict[str, Any]:
        result = {}
        if not isinstance(to_parse, dict):
            raise ValueError(
                f"Expected a dict, got {type(to_parse)}: {to_parse}"
            )
        for key, value in to_parse.items():
            key_parsed = eval_set_expression(
                expression=key,
                symbol_table=symbol_table,
                expected_space_name="tensors",
                location=f"{location} {key}"
            )
            for k2 in key_parsed:
                if k2 in result and result[k2] != value:
                    raise ValueError(
                        f"Multiple entries for {k2} in {location}: "
                        f"{result[k2]} and {value}"
                    )
                result[k2] = value
                
        for tensor_name in tensor_names:
            if tensor_name not in result:
                raise ValueError(
                    f"Tensor {tensor_name} not found in {location}. "
                    f"Available tensors: {', '.join(result.keys())}. Original "
                    f"expressions: {', '.join(to_parse.keys())}. Symbol table:\n\t"
                    + "\n\t".join(f"{k}: {v}" for k, v in symbol_table.items())
                )

        return result

    for i, node in enumerate(flattened_arch):
        if not isinstance(node, arch.TensorHolder):
            continue
        node = copy.copy(node)
        flattened_arch[i] = node
        node.attributes = copy.copy(node.attributes)

        node.attributes.datawidth = parse_tensor2bits(
            node.attributes.datawidth,
            location=f"datawidth of {node.name} for Einsum {job.einsum_name}",
            symbol_table=symbol_table
        )
            
    return flattened_arch

# =================================================================================================
# Top level
# =================================================================================================
def get_single_einsum_jobs(job: Job) -> SameEinsumJobs:
    compute_name = job.flattened_arch[-1].name
    
    job.tensor_to_relevancy = {
        tensor: get_rank_variable_relevancy(job.spec.workload.einsums[job.einsum_name], tensor)
        for tensor in job.spec.workload.einsums[job.einsum_name].tensor_names
    }

    mappings_constraints = tqdm(
        iterate_mappings_constraints(
            job.spec,
            job.einsum_name,
            job.flattened_arch,
            job.rank_variable_bounds,
            job.except_from_imperfect,
        ),
        desc=f"Generating pmapping templates for compute {compute_name} Einsum {job.einsum_name}",
    )
    rank_variable_bounds = get_rank_variable_bounds(
                job.spec.workload, job.einsum_name
            )
    
    stride_and_halo = get_stride_and_halo_of_einsum(job.einsum_name,
                                                                job.spec.workload,
                                                                rank_variable_bounds)

    jobs = SameEinsumJobs()
    for i, (mapping, constraints, symbol_table) in enumerate(mappings_constraints):
        new_job = copy.copy(job)
        new_job.mapping = mapping
        new_job.constraints = constraints
        new_job.job_id = uuid.uuid4()
        new_job.flattened_arch = parse_flattened_arch(new_job, symbol_table)
        new_job.rank_variable_bounds = rank_variable_bounds
        new_job.stride_and_halo = stride_and_halo
        new_job.compatibility
        jobs.append(new_job)

    return jobs

