from pathlib import Path
import unittest

from typing import Dict, List
from islpy import DEFAULT_CONTEXT, Map
from ruamel.yaml import YAML

from accelforge.frontend._binding import Binding, BindingNode

TESTS_DIR = Path(__file__).parent / "spec" / "binding"
yaml = YAML(typ="safe")


class TestBindingMapper(unittest.TestCase):
    def test_valid_bindings(self):
        """
        Tests that the valid bindings translate into the appropriate
        ISL strings.
        """
        specs_file: str = TESTS_DIR / "valid_bindings.yaml"
        with open(specs_file, mode="r", encoding="utf-8") as f:
            specs: List = yaml.load(f)

        spec: Dict
        for spec in specs:
            binding: Binding = Binding.model_validate(spec["binding"])

            soln: Dict = spec["solution"]

            soln_node: Dict[str, str]
            binding_node: BindingNode
            for soln_node, binding_node in zip(soln["nodes"], binding.nodes):
                isl_relations: Dict[str, Map] = binding_node.isl_relations
                assert soln_node.keys() == isl_relations.keys(), (
                    "Not all isl_relations read in properly. Missing \n"
                    f"{set(soln_node.keys()).difference(isl_relations.keys())} "
                    "\nfrom isl_relations and \n"
                    f"{set(isl_relations.keys()).difference(soln_node.keys())} "
                    "\nfrom solutions.\n"
                )

                tensor: str
                for tensor in soln_node:
                    soln_relation: Map = Map.read_from_str(
                        DEFAULT_CONTEXT, soln_node[tensor]
                    )
                    assert soln_relation.is_equal(
                        isl_relations[tensor]
                    ), f"\n{soln_relation} != \n{isl_relations[tensor]}"
