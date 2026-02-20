"""
Tests that YAML inputs are parsed into exactly the structures we expect.

These are "golden value" tests: we load known YAML files and assert that every
field, property, and derived value matches the expected result.
"""

import math
import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import (
    Workload,
    Einsum,
    TensorAccess,
    ImpliedProjection,
)
from accelforge.frontend.arch import (
    Arch,
    Hierarchical,
    Fork,
    Memory,
    Compute,
    Container,
    Toll,
    Leaf,
    Spatial as ArchSpatial,
    Action,
    TensorHolderAction,
    Tensors,
)
from accelforge.frontend.mapping.mapping import (
    Mapping,
    Temporal,
    Spatial as MappingSpatial,
    Storage,
    Compute as MappingCompute,
    Nested,
    Sequential,
    Pipeline,
)
from accelforge.frontend.renames import (
    Rename,
    RenameList,
    Renames,
    EinsumRename,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


# ============================================================================
# three_matmuls_annotated.yaml -- full workload golden values
# ============================================================================


class TestThreeMatmulsAnnotatedWorkload(unittest.TestCase):
    """Golden-value tests for examples/workloads/three_matmuls_annotated.yaml"""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest(f"YAML not found: {yaml_path}")
        cls.spec = Spec.from_yaml(yaml_path)
        cls.wl = cls.spec.workload

    # ----- rank_sizes -----

    def test_rank_sizes_keys(self):
        self.assertEqual(set(self.wl.rank_sizes.keys()), {"M", "N0", "N1", "N2", "N3"})

    def test_rank_sizes_values(self):
        for key in ("M", "N0", "N1", "N2", "N3"):
            self.assertEqual(self.wl.rank_sizes[key], 128)

    # ----- iteration_space_shape -----

    def test_iteration_space_shape_keys(self):
        self.assertEqual(
            set(self.wl.iteration_space_shape.keys()), {"m", "n0", "n1", "n2", "n3"}
        )

    def test_iteration_space_shape_values(self):
        for key, var in [
            ("m", "m"),
            ("n0", "n0"),
            ("n1", "n1"),
            ("n2", "n2"),
            ("n3", "n3"),
        ]:
            self.assertIn(f"0 <= {var}", self.wl.iteration_space_shape[key])
            self.assertIn("128", self.wl.iteration_space_shape[key])

    # ----- bits_per_value -----

    def test_bits_per_value(self):
        self.assertEqual(self.wl.bits_per_value, {"All": 8})

    # ----- einsum count and names -----

    def test_einsum_count(self):
        self.assertEqual(len(self.wl.einsums), 3)

    def test_einsum_names(self):
        self.assertEqual(
            [e.name for e in self.wl.einsums], ["Matmul1", "Matmul2", "Matmul3"]
        )

    # ----- Matmul1 -----

    def test_matmul1_tensor_count(self):
        m1 = self.wl.einsums["Matmul1"]
        self.assertEqual(len(m1.tensor_accesses), 3)

    def test_matmul1_tensor_names(self):
        m1 = self.wl.einsums["Matmul1"]
        self.assertEqual(m1.tensor_names, {"T0", "W0", "T1"})

    def test_matmul1_input_tensor_names(self):
        m1 = self.wl.einsums["Matmul1"]
        self.assertEqual(m1.input_tensor_names, {"T0", "W0"})

    def test_matmul1_output_tensor_names(self):
        m1 = self.wl.einsums["Matmul1"]
        self.assertEqual(m1.output_tensor_names, {"T1"})

    def test_matmul1_T0_projection(self):
        m1 = self.wl.einsums["Matmul1"]
        t0 = m1.tensor_accesses["T0"]
        self.assertIsInstance(t0.projection, ImpliedProjection)
        self.assertEqual(dict(t0.projection), {"M": "m", "N0": "n0"})

    def test_matmul1_W0_projection(self):
        m1 = self.wl.einsums["Matmul1"]
        w0 = m1.tensor_accesses["W0"]
        self.assertIsInstance(w0.projection, ImpliedProjection)
        self.assertEqual(dict(w0.projection), {"N0": "n0", "N1": "n1"})

    def test_matmul1_T1_projection(self):
        m1 = self.wl.einsums["Matmul1"]
        t1 = m1.tensor_accesses["T1"]
        self.assertIsInstance(t1.projection, ImpliedProjection)
        self.assertEqual(dict(t1.projection), {"M": "m", "N1": "n1"})

    def test_matmul1_T1_is_output(self):
        m1 = self.wl.einsums["Matmul1"]
        t1 = m1.tensor_accesses["T1"]
        self.assertTrue(t1.output)

    def test_matmul1_T0_not_output(self):
        m1 = self.wl.einsums["Matmul1"]
        t0 = m1.tensor_accesses["T0"]
        self.assertFalse(t0.output)

    def test_matmul1_rank_variables(self):
        m1 = self.wl.einsums["Matmul1"]
        self.assertEqual(m1.rank_variables, {"m", "n0", "n1"})

    def test_matmul1_ranks(self):
        m1 = self.wl.einsums["Matmul1"]
        self.assertEqual(m1.ranks, {"M", "N0", "N1"})

    def test_matmul1_inline_renames(self):
        """Matmul1 has inline renames: {input: T0}."""
        m1 = self.wl.einsums["Matmul1"]
        renames_by_name = {r.name: r.source for r in m1.renames}
        self.assertEqual(renames_by_name["input"], "T0")

    # ----- Matmul2 -----

    def test_matmul2_tensor_names(self):
        m2 = self.wl.einsums["Matmul2"]
        self.assertEqual(m2.tensor_names, {"T1", "W1", "T2"})

    def test_matmul2_rank_variables(self):
        m2 = self.wl.einsums["Matmul2"]
        self.assertEqual(m2.rank_variables, {"m", "n1", "n2"})

    def test_matmul2_output(self):
        m2 = self.wl.einsums["Matmul2"]
        self.assertEqual(m2.output_tensor_names, {"T2"})

    def test_matmul2_no_inline_renames(self):
        m2 = self.wl.einsums["Matmul2"]
        self.assertEqual(len(m2.renames), 0)

    # ----- Matmul3 -----

    def test_matmul3_tensor_names(self):
        m3 = self.wl.einsums["Matmul3"]
        self.assertEqual(m3.tensor_names, {"T2", "W2", "T3"})

    def test_matmul3_rank_variables(self):
        m3 = self.wl.einsums["Matmul3"]
        self.assertEqual(m3.rank_variables, {"m", "n2", "n3"})

    # ----- Workload-level properties -----

    def test_workload_tensor_names(self):
        self.assertEqual(
            self.wl.tensor_names,
            {"T0", "W0", "T1", "W1", "T2", "W2", "T3"},
        )

    def test_workload_rank_variables(self):
        self.assertEqual(self.wl.rank_variables, {"m", "n0", "n1", "n2", "n3"})

    # ----- Renames spec -----

    def test_renames_has_default(self):
        self.assertEqual(len(self.spec.renames.einsums), 1)
        self.assertEqual(self.spec.renames.einsums[0].name, "default")

    def test_renames_default_tensor_accesses(self):
        default = self.spec.renames.einsums[0]
        names = [r.name for r in default.tensor_accesses]
        self.assertEqual(names, ["input", "output", "weight"])

    def test_renames_default_input_source(self):
        default = self.spec.renames.einsums[0]
        input_rename = default.tensor_accesses["input"]
        self.assertEqual(input_rename.source, "Inputs & Intermediates")
        self.assertEqual(input_rename.expected_count, 1)

    def test_renames_default_output_source(self):
        default = self.spec.renames.einsums[0]
        output_rename = default.tensor_accesses["output"]
        self.assertEqual(output_rename.source, "Outputs")
        self.assertEqual(output_rename.expected_count, 1)

    def test_renames_default_weight_source(self):
        default = self.spec.renames.einsums[0]
        weight_rename = default.tensor_accesses["weight"]
        self.assertEqual(weight_rename.source, "~(input | output)")
        self.assertEqual(weight_rename.expected_count, 1)


# ============================================================================
# three_matmuls_annotated.yaml -- evaluated golden values
# ============================================================================


class TestThreeMatmulsEvaluated(unittest.TestCase):
    """Golden-value tests after evaluating three_matmuls_annotated.yaml"""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest(f"YAML not found: {yaml_path}")
        cls.spec = Spec.from_yaml(yaml_path)._spec_eval_expressions()
        cls.wl = cls.spec.workload

    def test_bits_per_value_resolved_to_8(self):
        for einsum in self.wl.einsums:
            for ta in einsum.tensor_accesses:
                self.assertEqual(
                    ta.bits_per_value,
                    8,
                    f"{ta.name} in {einsum.name} should be 8 bits",
                )

    def test_matmul1_renames_include_builtins(self):
        m1 = self.wl.einsums["Matmul1"]
        rename_names = {r.name for r in m1.renames}
        for builtin in (
            "All",
            "Tensors",
            "Nothing",
            "Inputs",
            "Outputs",
            "Intermediates",
        ):
            self.assertIn(builtin, rename_names)

    def test_matmul1_renames_include_default(self):
        m1 = self.wl.einsums["Matmul1"]
        rename_names = {r.name for r in m1.renames}
        self.assertIn("input", rename_names)
        self.assertIn("output", rename_names)
        self.assertIn("weight", rename_names)

    def test_matmul1_inline_rename_overrides_default(self):
        """Matmul1 has inline renames: {input: T0}. After evaluation the source
        is an InvertibleSet containing just 'T0'."""
        m1 = self.wl.einsums["Matmul1"]
        input_renames = [r for r in m1.renames if r.name == "input"]
        self.assertEqual(len(input_renames), 1)
        # After evaluation, the source is an InvertibleSet, not a raw string
        self.assertIn("T0", input_renames[0].source)

    def test_einsums_with_tensor_T1(self):
        """T1 appears in Matmul1 (output) and Matmul2 (input)."""
        einsums_with_t1 = self.wl.einsums_with_tensor("T1")
        self.assertEqual({e.name for e in einsums_with_t1}, {"Matmul1", "Matmul2"})

    def test_einsums_with_tensor_T0_input_only(self):
        einsums_with_t0 = self.wl.einsums_with_tensor("T0")
        self.assertEqual({e.name for e in einsums_with_t0}, {"Matmul1"})

    def test_einsums_with_tensor_as_output(self):
        einsums = self.wl.einsums_with_tensor_as_output("T2")
        self.assertEqual({e.name for e in einsums}, {"Matmul2"})

    def test_einsums_with_tensor_as_input(self):
        einsums = self.wl.einsums_with_tensor_as_input("T2")
        self.assertEqual({e.name for e in einsums}, {"Matmul3"})


# ============================================================================
# matmuls.yaml (Jinja parametric) -- golden values
# ============================================================================


class TestMatmulsJinjaParsed(unittest.TestCase):
    """Golden-value tests for examples/workloads/matmuls.yaml with Jinja vars."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest(f"YAML not found: {yaml_path}")
        cls.spec_1 = Spec.from_yaml(yaml_path, jinja_parse_data={"N_EINSUMS": 1})
        cls.spec_2 = Spec.from_yaml(
            yaml_path, jinja_parse_data={"N_EINSUMS": 2, "M": 256, "KN": 64}
        )

    # -- 1 einsum --

    def test_1_einsum_names(self):
        self.assertEqual([e.name for e in self.spec_1.workload.einsums], ["Matmul0"])

    def test_1_einsum_tensor_names(self):
        self.assertEqual(
            self.spec_1.workload.einsums[0].tensor_names, {"T0", "W0", "T1"}
        )

    def test_1_einsum_rank_vars(self):
        self.assertEqual(
            self.spec_1.workload.einsums[0].rank_variables, {"m", "n0", "n1"}
        )

    def test_1_einsum_iteration_space_shape(self):
        wl = self.spec_1.workload
        # Default M=128, KN=128
        self.assertIn("m", wl.iteration_space_shape)
        self.assertIn("128", wl.iteration_space_shape["m"])

    def test_1_einsum_inline_renames(self):
        m0 = self.spec_1.workload.einsums["Matmul0"]
        renames_by_name = {r.name: r.source for r in m0.renames}
        self.assertEqual(renames_by_name["weight"], "W0")
        self.assertEqual(renames_by_name["input"], "T0")
        self.assertEqual(renames_by_name["output"], "T1")

    # -- 2 einsums with custom sizes --

    def test_2_einsum_names(self):
        self.assertEqual(
            [e.name for e in self.spec_2.workload.einsums], ["Matmul0", "Matmul1"]
        )

    def test_2_einsum_iteration_space_shape_m(self):
        wl = self.spec_2.workload
        self.assertIn("256", wl.iteration_space_shape["m"])

    def test_2_einsum_iteration_space_shape_n0(self):
        wl = self.spec_2.workload
        self.assertIn("64", wl.iteration_space_shape["n0"])

    def test_2_einsum_Matmul0_tensors(self):
        m0 = self.spec_2.workload.einsums["Matmul0"]
        self.assertEqual(m0.tensor_names, {"T0", "W0", "T1"})

    def test_2_einsum_Matmul1_tensors(self):
        m1 = self.spec_2.workload.einsums["Matmul1"]
        self.assertEqual(m1.tensor_names, {"T1", "W1", "T2"})


# ============================================================================
# gpt3_6.7B_concise.yaml -- concise einsum string golden values
# ============================================================================


class TestGPTConciseParsed(unittest.TestCase):
    """Golden-value tests for the GPT concise workload."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B_concise.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest(f"YAML not found: {yaml_path}")
        cls.spec = Spec.from_yaml(yaml_path)
        cls.wl = cls.spec.workload

    def test_rank_sizes(self):
        rs = self.wl.rank_sizes
        self.assertEqual(rs["B"], 1)
        self.assertEqual(rs["M"], 8192)
        self.assertEqual(rs["P"], 8192)
        self.assertEqual(rs["H"], 32)
        self.assertEqual(rs["E"], 128)
        self.assertEqual(rs["F"], 128)
        self.assertEqual(rs["D"], 4096)
        self.assertEqual(rs["C"], 16384)
        self.assertEqual(rs["J"], 4096)
        self.assertEqual(rs["G"], 4096)

    def test_einsum_count(self):
        self.assertEqual(len(self.wl.einsums), 10)

    def test_einsum_names_in_order(self):
        self.assertEqual(
            [e.name for e in self.wl.einsums],
            ["I", "V", "K", "Q", "QK", "QK_softmax", "AV", "Z", "FFA", "FFB"],
        )

    def test_I_is_copy_operation_not_forwarded(self):
        """is_copy_operation: True in the YAML is NOT forwarded by _parse_einsum_entry
        when the 'einsum' key is present. This is current behavior -- the concise
        einsum string parser drops extra keys from the entry dict."""
        i_einsum = self.wl.einsums["I"]
        self.assertFalse(i_einsum.is_copy_operation)

    def test_I_output_name(self):
        i_einsum = self.wl.einsums["I"]
        self.assertEqual(i_einsum.output_tensor_names, {"I"})

    def test_I_input_name(self):
        i_einsum = self.wl.einsums["I"]
        self.assertEqual(i_einsum.input_tensor_names, {"I_in"})


    def test_I_projection_parsed_as_dict(self):
        """Concise einsum string projections are parsed by _parse_projection, which
        returns a plain dict (not ImpliedProjection). ImpliedProjection is only
        created when a list is passed to _projection_factory."""
        i_einsum = self.wl.einsums["I"]
        i_out = i_einsum.tensor_accesses["I"]
        # _parse_projection always returns a dict, not ImpliedProjection
        self.assertNotIsInstance(i_out.projection, ImpliedProjection)
        self.assertEqual(dict(i_out.projection), {"B": "b", "M": "m", "D": "d"})

    def test_V_tensor_names(self):
        v = self.wl.einsums["V"]
        self.assertEqual(v.tensor_names, {"V", "I", "WV"})

    def test_V_rank_variables(self):
        v = self.wl.einsums["V"]
        self.assertEqual(v.rank_variables, {"b", "m", "h", "e", "d"})

    def test_V_output(self):
        v = self.wl.einsums["V"]
        self.assertEqual(v.output_tensor_names, {"V"})

    def test_QK_uses_dict_projection(self):
        """QK: K[b, M:p, h, e] uses a dict projection for K."""
        qk = self.wl.einsums["QK"]
        k_ta = qk.tensor_accesses["K"]
        # M:p is explicit, so it's not an ImpliedProjection
        self.assertNotIsInstance(k_ta.projection, ImpliedProjection)
        self.assertEqual(k_ta.projection["M"], "p")

    def test_QK_Q_projection_is_dict(self):
        """Q[b, m, h, e] -- from a concise string, parsed as a plain dict."""
        qk = self.wl.einsums["QK"]
        q_ta = qk.tensor_accesses["Q"]
        self.assertNotIsInstance(q_ta.projection, ImpliedProjection)
        self.assertEqual(
            dict(q_ta.projection), {"B": "b", "M": "m", "H": "h", "E": "e"}
        )

    def test_AV_dict_projection_V(self):
        """AV: V[b, M:p, H:h, E:f]"""
        av = self.wl.einsums["AV"]
        v_ta = av.tensor_accesses["V"]
        self.assertNotIsInstance(v_ta.projection, ImpliedProjection)
        self.assertEqual(v_ta.projection["M"], "p")
        self.assertEqual(v_ta.projection["H"], "h")
        self.assertEqual(v_ta.projection["E"], "f")

    def test_persistent_tensors_field(self):
        self.assertEqual(self.wl.persistent_tensors, "weight - Intermediates")

    def test_bits_per_value_all_8(self):
        self.assertEqual(self.wl.bits_per_value, {"All": 8})


# ============================================================================
# simple.yaml arch -- golden values
# ============================================================================


class TestSimpleArchParsed(unittest.TestCase):
    """Golden-value tests for examples/arches/simple.yaml"""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not yaml_path.exists() or not wl_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(yaml_path, wl_path, jinja_parse_data={"N_EINSUMS": 1})
        cls.arch = cls.spec.arch

    def test_node_count(self):
        self.assertEqual(len(self.arch.nodes), 3)

    def test_node_types(self):
        self.assertIsInstance(self.arch.nodes[0], Memory)
        self.assertIsInstance(self.arch.nodes[1], Memory)
        self.assertIsInstance(self.arch.nodes[2], Compute)

    def test_node_names(self):
        names = [n.name for n in self.arch.nodes]
        self.assertEqual(names, ["MainMemory", "GlobalBuffer", "MAC"])

    def test_main_memory_size_inf(self):
        mm = self.arch.find("MainMemory")
        self.assertEqual(mm.size, "inf")

    def test_main_memory_leak_power(self):
        mm = self.arch.find("MainMemory")
        self.assertEqual(mm.leak_power, 0)

    def test_main_memory_area(self):
        mm = self.arch.find("MainMemory")
        self.assertEqual(mm.area, 0)

    def test_main_memory_actions(self):
        mm = self.arch.find("MainMemory")
        self.assertEqual(len(mm.actions), 2)
        action_names = [a.name for a in mm.actions]
        self.assertEqual(action_names, ["read", "write"])

    def test_main_memory_read_energy(self):
        """read energy is {{MainMemoryEnergy}} which defaults to 1."""
        mm = self.arch.find("MainMemory")
        read = [a for a in mm.actions if a.name == "read"][0]
        self.assertEqual(read.energy, 1)

    def test_main_memory_tensors_keep(self):
        mm = self.arch.find("MainMemory")
        self.assertEqual(mm.tensors.keep, "~Intermediates")

    def test_main_memory_tensors_may_keep(self):
        mm = self.arch.find("MainMemory")
        self.assertEqual(mm.tensors.may_keep, "All")

    def test_global_buffer_tensors_keep(self):
        gb = self.arch.find("GlobalBuffer")
        self.assertEqual(gb.tensors.keep, "~MainMemory")

    def test_mac_actions(self):
        mac = self.arch.find("MAC")
        self.assertEqual(len(mac.actions), 1)
        self.assertEqual(mac.actions[0].name, "compute")
        self.assertEqual(mac.actions[0].energy, 1)
        self.assertEqual(mac.actions[0].latency, 1)

    def test_mac_no_spatial(self):
        mac = self.arch.find("MAC")
        self.assertEqual(len(mac.spatial), 0)


class TestSimpleArchEvaluated(unittest.TestCase):
    """Golden-value tests for simple arch after evaluation."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not yaml_path.exists() or not wl_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(
            yaml_path, wl_path, jinja_parse_data={"N_EINSUMS": 1}
        )._spec_eval_expressions()
        cls.arch = cls.spec.arch

    def test_main_memory_size_is_inf(self):
        mm = self.arch.find("MainMemory")
        self.assertEqual(mm.size, math.inf)

    def test_global_buffer_size_is_inf(self):
        gb = self.arch.find("GlobalBuffer")
        self.assertEqual(gb.size, math.inf)


# ============================================================================
# tpu_v4i.yaml arch -- golden values
# ============================================================================


class TestTPUArchParsed(unittest.TestCase):
    """Golden-value tests for examples/arches/tpu_v4i.yaml (pre-evaluation)."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists() or not wl_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(yaml_path, wl_path)
        cls.arch = cls.spec.arch

    def test_node_count(self):
        # MainMemory, GlobalBuffer, LocalBuffer, ScalarUnit, ProcessingElement, Register, MAC
        self.assertEqual(len(self.arch.nodes), 7)

    def test_node_names_in_order(self):
        names = [n.name for n in self.arch.nodes]
        self.assertEqual(
            names,
            [
                "MainMemory",
                "GlobalBuffer",
                "LocalBuffer",
                "ScalarUnit",
                "ProcessingElement",
                "Register",
                "MAC",
            ],
        )

    def test_main_memory_is_memory(self):
        self.assertIsInstance(self.arch.find("MainMemory"), Memory)

    def test_scalar_unit_is_compute(self):
        self.assertIsInstance(self.arch.find("ScalarUnit"), Compute)

    def test_array_fanout_is_fanout(self):
        self.assertIsInstance(self.arch.find("ProcessingElement"), Container)

    def test_mac_is_compute(self):
        self.assertIsInstance(self.arch.find("MAC"), Compute)

    def test_global_buffer_size_expression(self):
        gb = self.arch.find("GlobalBuffer")
        # Before evaluation, size is the expression string
        self.assertEqual(gb.size, "1024*1024*128*8")

    def test_global_buffer_area(self):
        gb = self.arch.find("GlobalBuffer")
        self.assertEqual(gb.area, 112e-6)

    def test_main_memory_read_energy(self):
        mm = self.arch.find("MainMemory")
        read = [a for a in mm.actions if a.name == "read"][0]
        self.assertEqual(read.energy, 7.03e-12)

    def test_main_memory_read_latency_expression(self):
        mm = self.arch.find("MainMemory")
        read = [a for a in mm.actions if a.name == "read"][0]
        # Before evaluation, latency is a string expression
        self.assertEqual(read.latency, "1 / (8 * 614e9)")

    def test_local_buffer_spatial(self):
        lb = self.arch.find("LocalBuffer")
        self.assertEqual(len(lb.spatial), 1)
        self.assertEqual(lb.spatial[0].name, "Z")
        self.assertEqual(lb.spatial[0].fanout, 4)

    def test_local_buffer_spatial_may_reuse(self):
        lb = self.arch.find("LocalBuffer")
        self.assertEqual(lb.spatial[0].may_reuse, "Nothing")

    def test_local_buffer_spatial_min_usage(self):
        lb = self.arch.find("LocalBuffer")
        self.assertEqual(lb.spatial[0].min_usage, 1)

    def test_local_buffer_tensors_keep(self):
        lb = self.arch.find("LocalBuffer")
        self.assertEqual(lb.tensors.keep, "input | output")

    def test_scalar_unit_enabled_expression(self):
        su = self.arch.find("ScalarUnit")
        self.assertEqual(su.enabled, "len(All) == 2")

    def test_mac_enabled_expression(self):
        mac = self.arch.find("MAC")
        self.assertEqual(mac.enabled, "len(All) == 3")

    def test_array_fanout_spatial_count(self):
        af = self.arch.find("ProcessingElement")
        self.assertEqual(len(af.spatial), 2)

    def test_array_fanout_reuse_input(self):
        af = self.arch.find("ProcessingElement")
        ri = af.spatial[0]
        self.assertEqual(ri.name, "reuse_input")
        self.assertEqual(ri.fanout, 128)
        self.assertEqual(ri.may_reuse, "input")

    def test_array_fanout_reuse_output(self):
        af = self.arch.find("ProcessingElement")
        ro = af.spatial[1]
        self.assertEqual(ro.name, "reuse_output")
        self.assertEqual(ro.fanout, 128)
        self.assertEqual(ro.may_reuse, "output")

    def test_register_size_expression(self):
        reg = self.arch.find("Register")
        self.assertEqual(reg.size, "weight.bits_per_value if weight else 0")

    def test_register_tensors_keep(self):
        reg = self.arch.find("Register")
        self.assertEqual(reg.tensors.keep, "weight")

    def test_mac_latency_expression(self):
        mac = self.arch.find("MAC")
        compute_action = [a for a in mac.actions if a.name == "compute"][0]
        self.assertEqual(compute_action.latency, "1 / 1.05e9")

    def test_mac_energy(self):
        mac = self.arch.find("MAC")
        compute_action = [a for a in mac.actions if a.name == "compute"][0]
        self.assertEqual(compute_action.energy, 0.084e-12)


class TestTPUArchEvaluated(unittest.TestCase):
    """Golden-value tests for TPU arch after evaluation (with einsum context)."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists() or not wl_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(yaml_path, wl_path)._spec_eval_expressions(
            einsum_name="Matmul1"
        )
        cls.arch = cls.spec.arch

    def test_global_buffer_size_evaluated(self):
        gb = self.arch.find("GlobalBuffer")
        self.assertEqual(gb.size, 1024 * 1024 * 128 * 8)

    def test_local_buffer_size_evaluated(self):
        lb = self.arch.find("LocalBuffer")
        self.assertEqual(lb.size, 1024 * 1024 * 4 * 8)

    def test_main_memory_read_latency_evaluated(self):
        mm = self.arch.find("MainMemory")
        read = [a for a in mm.actions if a.name == "read"][0]
        self.assertAlmostEqual(read.latency, 1 / (8 * 614e9), places=20)

    def test_mac_latency_evaluated(self):
        mac = self.arch.find("MAC")
        compute_action = [a for a in mac.actions if a.name == "compute"][0]
        self.assertAlmostEqual(compute_action.latency, 1 / 1.05e9, places=20)

    def test_scalar_unit_latency_evaluated(self):
        su = self.arch.find("ScalarUnit")
        compute_action = [a for a in su.actions if a.name == "compute"][0]
        self.assertAlmostEqual(compute_action.latency, 1 / 1.05e9 / 128, places=20)


# ============================================================================
# unfused_matmuls_to_simple.yaml mapping -- golden values (1 einsum)
# ============================================================================


class TestUnfusedMapping1Einsum(unittest.TestCase):
    """Golden-value tests for unfused mapping with 1 einsum."""

    @classmethod
    def setUpClass(cls):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        map_path = EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml"
        if not all(p.exists() for p in [arch_path, wl_path, map_path]):
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(
            arch_path,
            wl_path,
            map_path,
            jinja_parse_data={"N_EINSUMS": 1, "M": 64, "KN": 32},
        )
        cls.mapping = cls.spec.mapping

    def test_node_count(self):
        # T0@MainMemory, W0@MainMemory, T1@MainMemory,
        # T0+W0+T1@GlobalBuffer,
        # Temporal(m), Temporal(n0), Temporal(n1),
        # Compute(Matmul0)
        self.assertEqual(len(self.mapping.nodes), 8)

    def test_first_three_are_storage_main_memory(self):
        for i in range(3):
            node = self.mapping.nodes[i]
            self.assertIsInstance(node, Storage)
            self.assertEqual(node.component, "MainMemory")

    def test_storage_tensors_at_main_memory(self):
        tensors_at_mm = set()
        for i in range(3):
            tensors_at_mm.update(self.mapping.nodes[i].tensors)
        self.assertEqual(tensors_at_mm, {"T0", "W0", "T1"})

    def test_fourth_is_storage_global_buffer(self):
        node = self.mapping.nodes[3]
        self.assertIsInstance(node, Storage)
        self.assertEqual(node.component, "GlobalBuffer")
        self.assertEqual(set(node.tensors), {"T0", "W0", "T1"})

    def test_temporal_loops(self):
        temporals = [n for n in self.mapping.nodes if isinstance(n, Temporal)]
        self.assertEqual(len(temporals), 3)
        rvs = [t.rank_variable for t in temporals]
        self.assertEqual(rvs, ["m", "n0", "n1"])

    def test_temporal_tile_shapes(self):
        temporals = [n for n in self.mapping.nodes if isinstance(n, Temporal)]
        for t in temporals:
            self.assertEqual(t.tile_shape, 1)

    def test_compute_node(self):
        computes = [n for n in self.mapping.nodes if isinstance(n, MappingCompute)]
        self.assertEqual(len(computes), 1)
        self.assertEqual(computes[0].einsum, "Matmul0")
        self.assertEqual(computes[0].component, "MAC")


# ============================================================================
# unfused_matmuls_to_simple.yaml mapping -- golden values (2 einsums)
# ============================================================================


class TestUnfusedMapping2Einsums(unittest.TestCase):
    """Golden-value tests for unfused mapping with 2 einsums (Sequential)."""

    @classmethod
    def setUpClass(cls):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        map_path = EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml"
        if not all(p.exists() for p in [arch_path, wl_path, map_path]):
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(
            arch_path,
            wl_path,
            map_path,
            jinja_parse_data={"N_EINSUMS": 2},
        )
        cls.mapping = cls.spec.mapping

    def test_top_level_structure(self):
        """Top-level: 5 Storages (T0, T1, T2, W0, W1 at MainMemory) + 1 Sequential."""
        storage_count = sum(1 for n in self.mapping.nodes if isinstance(n, Storage))
        seq_count = sum(1 for n in self.mapping.nodes if isinstance(n, Sequential))
        self.assertEqual(storage_count, 5)
        self.assertEqual(seq_count, 1)

    def test_sequential_has_two_nested(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        self.assertEqual(len(seq.nodes), 2)
        for nested in seq.nodes:
            self.assertIsInstance(nested, Nested)

    def test_nested_0_structure(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested0 = seq.nodes[0]
        # Storage(T0,W0,T1@GlobalBuffer), Temporal(m), Temporal(n0), Temporal(n1), Compute(Matmul0)
        self.assertEqual(len(nested0.nodes), 5)
        self.assertIsInstance(nested0.nodes[0], Storage)
        self.assertEqual(nested0.nodes[0].component, "GlobalBuffer")
        self.assertEqual(set(nested0.nodes[0].tensors), {"T0", "W0", "T1"})

    def test_nested_0_compute(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested0 = seq.nodes[0]
        compute = [n for n in nested0.nodes if isinstance(n, MappingCompute)][0]
        self.assertEqual(compute.einsum, "Matmul0")

    def test_nested_1_compute(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested1 = seq.nodes[1]
        compute = [n for n in nested1.nodes if isinstance(n, MappingCompute)][0]
        self.assertEqual(compute.einsum, "Matmul1")


# ============================================================================
# fused_matmuls_to_simple.yaml mapping -- golden values
# ============================================================================


class TestFusedMapping2Einsums(unittest.TestCase):
    """Golden-value tests for fused mapping with 2 einsums."""

    @classmethod
    def setUpClass(cls):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        map_path = EXAMPLES_DIR / "mappings" / "fused_matmuls_to_simple.yaml"
        if not all(p.exists() for p in [arch_path, wl_path, map_path]):
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(
            arch_path,
            wl_path,
            map_path,
            jinja_parse_data={"N_EINSUMS": 2},
        )
        cls.mapping = cls.spec.mapping

    def test_fused_has_temporal_m_before_sequential(self):
        """Fused mapping: Temporal(m) appears before the Sequential."""
        temporals_before_seq = []
        for node in self.mapping.nodes:
            if isinstance(node, Temporal):
                temporals_before_seq.append(node)
            elif isinstance(node, Sequential):
                break
        self.assertGreater(len(temporals_before_seq), 0)
        self.assertEqual(temporals_before_seq[0].rank_variable, "m")

    def test_fused_sequential_has_two_nested(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        self.assertEqual(len(seq.nodes), 2)


# ============================================================================
# Programmatic construction golden values
# ============================================================================


class TestProgrammaticWorkloadValues(unittest.TestCase):
    """Test that programmatically constructed workloads have expected values."""

    def test_list_projection_becomes_implied(self):
        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertIsInstance(ta.projection, ImpliedProjection)
        self.assertEqual(dict(ta.projection), {"M": "m", "K": "k"})

    def test_dict_projection_stays_dict(self):
        ta = TensorAccess(name="A", projection={"Row": "m", "Col": "k"})
        self.assertNotIsInstance(ta.projection, ImpliedProjection)
        self.assertEqual(ta.projection, {"Row": "m", "Col": "k"})

    def test_tensor_access_ranks(self):
        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertEqual(ta.ranks, ("M", "K"))

    def test_tensor_access_rank_variables(self):
        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertEqual(ta.rank_variables, {"m", "k"})

    def test_tensor_access_rank2rank_variables(self):
        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertEqual(ta.rank2rank_variables, {"M": {"m"}, "K": {"k"}})

    def test_tensor_access_rank_variable2ranks(self):
        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertEqual(ta.rank_variable2ranks, {"m": {"M"}, "k": {"K"}})

    def test_tensor_access_directly_indexing(self):
        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertEqual(ta.directly_indexing_rank_variables, {"m", "k"})

    def test_tensor_access_expression_indexing_empty(self):
        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertEqual(ta.expression_indexing_rank_variables, set())

    def test_einsum_properties(self):
        e = Einsum(
            name="Matmul",
            tensor_accesses=[
                {"name": "A", "projection": ["m", "k"]},
                {"name": "B", "projection": ["k", "n"]},
                {"name": "C", "projection": ["m", "n"], "output": True},
            ],
        )
        self.assertEqual(e.tensor_names, {"A", "B", "C"})
        self.assertEqual(e.input_tensor_names, {"A", "B"})
        self.assertEqual(e.output_tensor_names, {"C"})
        self.assertEqual(e.rank_variables, {"m", "k", "n"})
        self.assertEqual(e.ranks, {"M", "K", "N"})

    def test_einsum_tensor2rank_variables(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                {"name": "X", "projection": ["a", "b"]},
                {"name": "Y", "projection": ["b", "c"], "output": True},
            ],
        )
        t2rv = e.tensor2rank_variables
        self.assertEqual(t2rv["X"], {"a", "b"})
        self.assertEqual(t2rv["Y"], {"b", "c"})

    def test_einsum_tensor2irrelevant_rank_variables(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                {"name": "X", "projection": ["a", "b"]},
                {"name": "Y", "projection": ["b", "c"], "output": True},
            ],
        )
        irr = e.tensor2irrelevant_rank_variables
        self.assertEqual(irr["X"], {"c"})
        self.assertEqual(irr["Y"], {"a"})

    def test_einsum_indexing_expressions(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                {"name": "X", "projection": ["a", "b"]},
                {"name": "Y", "projection": ["b", "c"], "output": True},
            ],
        )
        self.assertEqual(e.indexing_expressions, {"a", "b", "c"})

    def test_workload_properties(self):
        wl = Workload(
            rank_sizes={"M": 10, "K": 20, "N": 30},
            bits_per_value={"All": 16},
            einsums=[
                {
                    "name": "E1",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "k"]},
                        {"name": "B", "projection": ["k", "n"]},
                        {"name": "C", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        )
        self.assertEqual(wl.tensor_names, {"A", "B", "C"})
        self.assertEqual(wl.rank_variables, {"m", "k", "n"})
        self.assertEqual(len(wl.einsums), 1)

    def test_workload_einsums_with_tensor(self):
        wl = Workload(
            rank_sizes={"M": 10, "K": 20, "N": 30},
            bits_per_value={"All": 8},
            einsums=[
                {
                    "name": "E1",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "k"]},
                        {"name": "B", "projection": ["k", "n"]},
                        {"name": "C", "projection": ["m", "n"], "output": True},
                    ],
                },
                {
                    "name": "E2",
                    "tensor_accesses": [
                        {"name": "C", "projection": ["m", "n"]},
                        {"name": "D", "projection": ["n", "k"]},
                        {"name": "E", "projection": ["m", "k"], "output": True},
                    ],
                },
            ],
        )
        c_einsums = wl.einsums_with_tensor("C")
        self.assertEqual({e.name for e in c_einsums}, {"E1", "E2"})
        a_einsums = wl.einsums_with_tensor("A")
        self.assertEqual({e.name for e in a_einsums}, {"E1"})

    def test_rename_list_factory_dict(self):
        rl = RenameList(
            [
                Rename(name="input", source="T0"),
                Rename(name="output", source="T1"),
            ]
        )
        self.assertEqual(len(rl), 2)
        self.assertEqual(rl["input"].source, "T0")

    def test_memory_defaults(self):
        mem = Memory(
            name="TestMem",
            size=1000,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
            leak_power=0,
            area=0,
        )
        self.assertEqual(mem.name, "TestMem")
        self.assertEqual(mem.size, 1000)
        self.assertEqual(len(mem.actions), 2)
        self.assertEqual(len(mem.spatial), 0)

    def test_compute_defaults(self):
        comp = Compute(
            name="MAC",
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
            leak_power=0,
            area=0,
        )
        self.assertEqual(comp.name, "MAC")
        self.assertTrue(comp.enabled)

    def test_mapping_temporal_values(self):
        t = Temporal(rank_variable="m", tile_shape=4)
        self.assertEqual(t.rank_variable, "m")
        self.assertEqual(t.tile_shape, 4)

    def test_mapping_storage_values(self):
        s = Storage(tensors=["A", "B"], component="Mem")
        self.assertEqual(list(s.tensors), ["A", "B"])
        self.assertEqual(s.component, "Mem")

    def test_mapping_compute_values(self):
        c = MappingCompute(einsum="E1", component="MAC")
        self.assertEqual(c.einsum, "E1")
        self.assertEqual(c.component, "MAC")


# ============================================================================
# simple_fused.yaml mapping -- golden values
# ============================================================================


class TestSimpleFusedMappingParsed(unittest.TestCase):
    """Golden-value tests for examples/mappings/simple_fused.yaml."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "mappings" / "simple_fused.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest(f"YAML not found: {yaml_path}")
        cls.spec = Spec.from_yaml(yaml_path)
        cls.mapping = cls.spec.mapping

    def test_first_node_is_storage_offchip(self):
        node = self.mapping.nodes[0]
        self.assertIsInstance(node, Storage)
        self.assertEqual(node.component, "OffChipBuffer")
        self.assertEqual(set(node.tensors), {"I", "WA", "WB", "B"})

    def test_second_node_is_storage_onchip_WA(self):
        node = self.mapping.nodes[1]
        self.assertIsInstance(node, Storage)
        self.assertEqual(node.component, "OnChipBuffer")
        self.assertEqual(list(node.tensors), ["WA"])

    def test_third_is_temporal_nA(self):
        node = self.mapping.nodes[2]
        self.assertIsInstance(node, Temporal)
        self.assertEqual(node.rank_variable, "nA")
        self.assertEqual(node.tile_shape, 1)

    def test_fourth_is_storage_onchip_A(self):
        node = self.mapping.nodes[3]
        self.assertIsInstance(node, Storage)
        self.assertEqual(node.component, "OnChipBuffer")
        self.assertEqual(list(node.tensors), ["A"])

    def test_sequential_present(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)]
        self.assertEqual(len(seq), 1)

    def test_sequential_has_two_nested(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        self.assertEqual(len(seq.nodes), 2)
        for nested in seq.nodes:
            self.assertIsInstance(nested, Nested)

    def test_nested_0_einsum_A(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested0 = seq.nodes[0]
        computes = [n for n in nested0.nodes if isinstance(n, MappingCompute)]
        self.assertEqual(len(computes), 1)
        self.assertEqual(computes[0].einsum, "EinsumA")
        self.assertEqual(computes[0].component, "ComputeUnit")

    def test_nested_1_einsum_B(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested1 = seq.nodes[1]
        computes = [n for n in nested1.nodes if isinstance(n, MappingCompute)]
        self.assertEqual(len(computes), 1)
        self.assertEqual(computes[0].einsum, "EinsumB")
        self.assertEqual(computes[0].component, "ComputeUnit")

    def test_nested_0_temporal_nI(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested0 = seq.nodes[0]
        temporals = [n for n in nested0.nodes if isinstance(n, Temporal)]
        self.assertEqual(len(temporals), 1)
        self.assertEqual(temporals[0].rank_variable, "nI")

    def test_nested_1_temporal_nB(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested1 = seq.nodes[1]
        temporals = [n for n in nested1.nodes if isinstance(n, Temporal)]
        self.assertEqual(len(temporals), 1)
        self.assertEqual(temporals[0].rank_variable, "nB")

    def test_nested_1_storage_tensors(self):
        seq = [n for n in self.mapping.nodes if isinstance(n, Sequential)][0]
        nested1 = seq.nodes[1]
        storages = [n for n in nested1.nodes if isinstance(n, Storage)]
        self.assertEqual(len(storages), 1)
        self.assertEqual(set(storages[0].tensors), {"B", "WB"})


if __name__ == "__main__":
    unittest.main()
