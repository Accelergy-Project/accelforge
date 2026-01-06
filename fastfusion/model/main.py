from fastfusion.frontend.spec import Mapping, Spec
from fastfusion.frontend.mapping import Compute, Split, Nested, NodeList
from fastfusion.mapper import Metrics
from fastfusion.mapper.FFM._make_pmappings.make_pmappings import get_rank_variable_bounds_for_all_einsums
from fastfusion.mapper.FFM._make_pmappings.make_pmappings_from_templates.run_model import run_model
from fastfusion.mapper.FFM._make_pmappings.pmapper_job import Job


def evaluate_mapping(spec: Spec):
    spec = spec.calculate_component_energy_area(area=False)
    flattened_arch = spec.get_flattened_architecture()
    job = Job(
        spec=spec,
        metrics=Metrics.all_metrics(),
        rank_variable_bounds=get_rank_variable_bounds_for_all_einsums(spec),
        flattened_arch=flattened_arch,
    )

    for pmapping in _split_mapping_to_pmappings(spec.mapping):
        print(pmapping)
        job.mapping = pmapping
        job.einsum_name = pmapping.nodes[-1].einsum
        result = run_model(job)
        print(result)


def _split_mapping_to_pmappings(mapping: Mapping):
    """
    A DFS-like algorithm to split a mapping into pmappings at Split nodes.

    DFS has to be modified because the tree has list of nodes for nested nodes
    instead of links to children.
    """
    dfs_stack: list[NodeList] = [mapping.nodes]
    cur_pmapping = []

    while dfs_stack:
        # nodes_segment is a list of nested nodes with a Split or Compute at the end.
        nodes_segment = dfs_stack.pop()
        assert isinstance(nodes_segment[-1], (Split, Compute))

        cur_pmapping.append(nodes_segment[:-1])

        last_node = nodes_segment[-1]
        if isinstance(last_node, Split):
            for segment in last_node.nodes:
                assert isinstance(segment, Nested)
                dfs_stack.append(segment.nodes)
        else:
            assert isinstance(last_node, Compute)

            mapping = Mapping()
            mapping.nodes = [n for ns in cur_pmapping for n in ns] + [last_node]
            yield mapping

            cur_pmapping.pop() # Remove the last segment
