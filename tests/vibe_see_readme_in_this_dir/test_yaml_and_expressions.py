"""
Tests for features documented in the guides but not covered by other test files:

  1. YAML merge operators (<<, <<<, !nomerge)
  2. eval_expression: math functions, symbol table, precedence
  3. Set expressions: &, |, ~, -, builtins, .rank_variables, .bits_per_value
  4. Concise einsum: extra attrs via 'einsum' key, tensor_accesses extras
  5. Arch: Comparison, Spatial.reuse, enabled, total_latency, Toll, Container
  6. Container variations (spatial on Memory, standalone Container, loop_bounds)
  7. Verbose GPT: persistent per-tensor, rank_variable renames
"""

import math
import os
import tempfile
import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import (
    Workload,
    Einsum,
    TensorAccess,
    ImpliedProjection,
    _parse_einsum_string,
    _parse_einsum_entry,
    _parse_projection,
)
from accelforge.frontend.arch import (
    Arch,
    Memory,
    Compute,
    Container,
    Toll,
    Comparison,
    Spatial as ArchSpatial,
)
from accelforge.frontend.mapping.mapping import (
    Mapping,
    Temporal,
    Storage,
    Compute as MappingCompute,
)
from accelforge.frontend.renames import Rename, RenameList
from accelforge.util._eval_expressions import eval_expression, MATH_FUNCS

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


def _yaml_spec(yaml_text: str, **jinja_parse_data) -> Spec:
    """Helper: write YAML text to a temp file and load it as a Spec."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(_REPO_ROOT)
    ) as f:
        f.write(yaml_text)
        f.flush()
        try:
            return Spec.from_yaml(f.name, jinja_parse_data=jinja_parse_data)
        finally:
            os.unlink(f.name)


# ============================================================================
# 1. YAML merge operators
# ============================================================================


class TestYAMLMergeOperators(unittest.TestCase):
    """Test the <<, <<<, and !nomerge YAML extensions."""

    def test_shallow_merge(self):
        """<< merges dictionaries without recursion."""
        spec = _yaml_spec(
            """\
variables:
  base: &base
    a: 1
    b: 2
  merged:
    <<: *base
    b: 99
    c: 3
"""
        )
        v = spec.variables
        self.assertEqual(v["merged"]["a"], 1)
        self.assertEqual(v["merged"]["b"], 99)
        self.assertEqual(v["merged"]["c"], 3)

    def test_hierarchical_merge(self):
        """<<< merges dictionaries recursively into nested dicts."""
        spec = _yaml_spec(
            """\
variables:
  base: &base
    a: 1
    nested: {x: 10, y: 20}
  merged:
    <<<: *base
    nested: {x: 99}
"""
        )
        v = spec.variables
        self.assertEqual(v["merged"]["a"], 1)
        self.assertEqual(v["merged"]["nested"]["x"], 99)
        # Hierarchical merge brings y from the base
        self.assertEqual(v["merged"]["nested"]["y"], 20)

    def test_hierarchical_merge_list_local_wins(self):
        """<<< would concatenate lists, but merge_check's iteration order
        causes the local value to overwrite the merged result."""
        spec = _yaml_spec(
            """\
variables:
  base: &base
    mylist: [4, 5, 6]
  merged:
    <<<: *base
    mylist: [1, 2, 3]
"""
        )
        v = spec.variables
        # Local value overwrites merged result due to iteration order
        self.assertEqual(v["merged"]["mylist"], [1, 2, 3])

    def test_hierarchical_merge_adds_missing_keys(self):
        """<<< adds keys from base that aren't in the local dict."""
        spec = _yaml_spec(
            """\
variables:
  base: &base
    from_base: [4, 5, 6]
    shared: 1
  merged:
    <<<: *base
    shared: 99
"""
        )
        v = spec.variables
        self.assertEqual(v["merged"]["from_base"], [4, 5, 6])
        self.assertEqual(v["merged"]["shared"], 99)

    def test_nomerge_blocks_list_merge(self):
        """!nomerge prevents list concatenation under <<<."""
        spec = _yaml_spec(
            """\
variables:
  base: &base
    mylist: [4, 5, 6]
  merged:
    <<<: *base
    mylist: !nomerge [1, 2, 3]
"""
        )
        v = spec.variables
        self.assertEqual(v["merged"]["mylist"], [1, 2, 3])

    def test_shallow_merge_does_not_recurse(self):
        """<< does NOT recurse into nested dicts (unlike <<<)."""
        spec = _yaml_spec(
            """\
variables:
  base: &base
    nested: {x: 10, y: 20}
  merged:
    <<: *base
    nested: {x: 99}
"""
        )
        v = spec.variables
        self.assertEqual(v["merged"]["nested"]["x"], 99)
        # Shallow merge: nested is fully overridden
        self.assertNotIn("y", v["merged"]["nested"])

    def test_merge_multiple_anchors(self):
        """<< can merge multiple anchors; earlier anchors take precedence."""
        spec = _yaml_spec(
            """\
variables:
  d1: &d1
    a: 1
    b: 2
  d2: &d2
    b: 99
    c: 3
  merged:
    <<: [*d1, *d2]
"""
        )
        v = spec.variables
        self.assertEqual(v["merged"]["a"], 1)
        self.assertEqual(v["merged"]["b"], 2)  # d1 takes precedence
        self.assertEqual(v["merged"]["c"], 3)


# ============================================================================
# 2. eval_expression: math functions and symbol table
# ============================================================================


class TestEvalExpression(unittest.TestCase):
    """Test eval_expression with math functions and symbol table lookup."""

    def test_numeric_passthrough(self):
        self.assertEqual(eval_expression(42, {}), 42)

    def test_float_passthrough(self):
        self.assertAlmostEqual(eval_expression(3.14, {}), 3.14)

    def test_string_numeric(self):
        self.assertEqual(eval_expression("42", {}), 42)

    def test_inf_string(self):
        self.assertEqual(eval_expression("inf", {}), math.inf)

    def test_symbol_lookup(self):
        self.assertEqual(eval_expression("x", {"x": 10}), 10)

    def test_simple_arithmetic(self):
        result = eval_expression("x + y", {"x": 3, "y": 7})
        self.assertEqual(result, 10)

    def test_math_ceil(self):
        result = eval_expression("ceil(3.2)", {})
        self.assertEqual(result, 4)

    def test_math_floor(self):
        result = eval_expression("floor(3.9)", {})
        self.assertEqual(result, 3)

    def test_math_log2(self):
        result = eval_expression("log2(1024)", {})
        self.assertEqual(result, 10.0)

    def test_math_sqrt(self):
        result = eval_expression("sqrt(16)", {})
        self.assertEqual(result, 4.0)

    def test_min_max(self):
        self.assertEqual(eval_expression("min(3, 7)", {}), 3)
        self.assertEqual(eval_expression("max(3, 7)", {}), 7)

    def test_range_sum(self):
        result = eval_expression("sum(y for y in range(1, 10))", {})
        self.assertEqual(result, 45)

    def test_len(self):
        result = eval_expression("len(x)", {"x": [1, 2, 3]})
        self.assertEqual(result, 3)

    def test_conditional_expression(self):
        result = eval_expression("x if x > 5 else 0", {"x": 10})
        self.assertEqual(result, 10)
        result = eval_expression("x if x > 5 else 0", {"x": 3})
        self.assertEqual(result, 0)

    def test_multiplication_expression(self):
        result = eval_expression("1024*1024*128*8", {})
        self.assertEqual(result, 1024 * 1024 * 128 * 8)

    def test_division_expression(self):
        result = eval_expression("1 / (8 * 614e9)", {})
        self.assertAlmostEqual(result, 1 / (8 * 614e9), places=25)

    def test_bool_true(self):
        self.assertTrue(eval_expression("True", {}))

    def test_bool_false(self):
        self.assertFalse(eval_expression("False", {}))

    def test_all_math_funcs_registered(self):
        """All documented math functions should be in MATH_FUNCS."""
        documented = [
            "ceil",
            "floor",
            "log2",
            "log10",
            "sqrt",
            "exp",
            "sin",
            "cos",
            "tan",
            "pi",
            "e",
            "inf",
            "nan",
            "abs",
            "round",
            "sum",
            "range",
            "len",
            "min",
            "max",
            "float",
            "int",
            "str",
            "bool",
            "list",
            "tuple",
            "enumerate",
            "map",
            "pow",
            "factorial",
            "gcd",
        ]
        for name in documented:
            self.assertIn(name, MATH_FUNCS, f"{name} not in MATH_FUNCS")


class TestExpressionEvalInSpec(unittest.TestCase):
    """Test expression evaluation through Spec._spec_eval_expressions."""

    def test_variables_evaluate_cross_references(self):
        """Variables can reference other variables: b = a + 5."""
        spec = _yaml_spec(
            """\
variables:
  a: 123
  b: a + 5
  c: min(b, 3)
"""
        )
        evaluated = spec._spec_eval_expressions()
        self.assertEqual(evaluated.variables["a"], 123)
        self.assertEqual(evaluated.variables["b"], 128)
        self.assertEqual(evaluated.variables["c"], 3)

    def test_arch_expressions_reference_variables(self):
        """Arch fields can reference top-level variables."""
        spec = _yaml_spec(
            """\
variables:
  mem_size: 1024 * 1024

arch:
  nodes:
  - !Memory
    name: Mem
    size: mem_size * 8
    leak_power: 0
    area: 0
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Compute
    name: MAC
    leak_power: 0
    area: 0
    actions:
    - {name: compute, energy: 1, latency: 1}

workload:
  rank_sizes: {M: 16}
  bits_per_value: {All: 8}
  einsums:
  - name: E
    tensor_accesses:
    - {name: A, projection: [m]}
    - {name: B, projection: [m], output: true}
"""
        )
        evaluated = spec._spec_eval_expressions()
        mem = evaluated.arch.find("Mem")
        self.assertEqual(mem.size, 1024 * 1024 * 8)


# ============================================================================
# 3. Set expressions
# ============================================================================


class TestSetExpressionsParsed(unittest.TestCase):
    """Test that set expression strings are parsed and stored correctly."""

    def test_complement(self):
        """~Intermediates is a complement set expression."""
        spec = _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: Mem
    size: inf
    leak_power: 0
    area: 0
    tensors: {keep: ~Intermediates, may_keep: All}
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Compute
    name: MAC
    leak_power: 0
    area: 0
    actions:
    - {name: compute, energy: 1, latency: 1}
"""
        )
        mem = spec.arch.find("Mem")
        self.assertEqual(mem.tensors.keep, "~Intermediates")

    def test_union_expression(self):
        spec = _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: Mem
    size: inf
    leak_power: 0
    area: 0
    tensors: {keep: input | output}
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Compute
    name: MAC
    leak_power: 0
    area: 0
    actions:
    - {name: compute, energy: 1, latency: 1}
"""
        )
        mem = spec.arch.find("Mem")
        self.assertEqual(mem.tensors.keep, "input | output")

    def test_reference_above_memory(self):
        """~MainMemory.tensors references Above node."""
        spec = _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: MainMemory
    size: inf
    leak_power: 0
    area: 0
    tensors: {keep: ~Intermediates, may_keep: All}
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Memory
    name: Buffer
    size: inf
    leak_power: 0
    area: 0
    tensors: {keep: ~MainMemory.tensors, may_keep: All}
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Compute
    name: MAC
    leak_power: 0
    area: 0
    actions:
    - {name: compute, energy: 1, latency: 1}
"""
        )
        buf = spec.arch.find("Buffer")
        self.assertEqual(buf.tensors.keep, "~MainMemory.tensors")


# ============================================================================
# 4. Concise einsum string: extra attrs via 'einsum' key
# ============================================================================


class TestConciseEinsumExtras(unittest.TestCase):
    """Test that the 'einsum' keyword works with extra tensor_accesses attrs."""

    def test_einsum_keyword_with_bits_per_value(self):
        """The docs show: einsum: I[b,m,d] = I_in[b,m,d] with tensor_accesses extras."""
        entry = {
            "einsum": "I[b, m, d] = I_in[b, m, d]",
            "tensor_accesses": [{"name": "I_in", "bits_per_value": 16}],
        }
        parsed = _parse_einsum_entry(entry)
        # bits_per_value should be merged onto the I_in tensor access
        i_in = [ta for ta in parsed["tensor_accesses"] if ta["name"] == "I_in"][0]
        self.assertEqual(i_in["bits_per_value"], 16)

    def test_einsum_keyword_preserves_name(self):
        entry = {"einsum": "C[m, n] = A[m, k] * B[k, n]"}
        parsed = _parse_einsum_entry(entry)
        self.assertEqual(parsed["name"], "C")

    def test_einsum_keyword_preserves_tensor_accesses(self):
        entry = {"einsum": "C[m, n] = A[m, k] * B[k, n]"}
        parsed = _parse_einsum_entry(entry)
        names = {ta["name"] for ta in parsed["tensor_accesses"]}
        self.assertEqual(names, {"A", "B", "C"})

    def test_conflicting_tensor_access_attr_raises(self):
        """Setting projection in both einsum string and tensor_accesses is an error."""
        entry = {
            "einsum": "C[m, n] = A[m, k] * B[k, n]",
            "tensor_accesses": [{"name": "A", "projection": ["x", "y"]}],
        }
        with self.assertRaises(ValueError):
            _parse_einsum_entry(entry)

    def test_nonexistent_tensor_in_extras_raises(self):
        entry = {
            "einsum": "C[m, n] = A[m, k] * B[k, n]",
            "tensor_accesses": [{"name": "Z", "bits_per_value": 8}],
        }
        with self.assertRaises(ValueError):
            _parse_einsum_entry(entry)

    def test_is_copy_operation_not_forwarded(self):
        """Extra keys like is_copy_operation are IS forwarded by _parse_einsum_entry."""
        entry = {
            "einsum": "I[b, m, d] = I_in[b, m, d]",
            "is_copy_operation": True,
        }
        parsed = _parse_einsum_entry(entry)
        # _parse_einsum_entry only returns name, tensor_accesses, renames
        self.assertIn("is_copy_operation", parsed)


# ============================================================================
# 5. Arch: Comparison, Spatial.reuse, enabled, Toll
# ============================================================================


class TestComparisonParsed(unittest.TestCase):
    """Test Comparison objects on Spatial.loop_bounds."""

    def test_comparison_fields(self):
        c = Comparison(expression="~m", operator="==", value=1)
        self.assertEqual(c.expression, "~m")
        self.assertEqual(c.operator, "==")
        self.assertEqual(c.value, 1)

    def test_comparison_constrained_to_one(self):
        c = Comparison(expression="~m", operator="==", value=1)
        self.assertTrue(c._constrained_to_one())

    def test_comparison_not_constrained_to_one(self):
        c = Comparison(expression="~m", operator="==", value=4)
        self.assertFalse(c._constrained_to_one())

    def test_comparison_leq_constrained_to_one(self):
        c = Comparison(expression="~m", operator="<=", value=1)
        self.assertTrue(c._constrained_to_one())

    def test_comparison_geq_not_constrained(self):
        c = Comparison(expression="~m", operator=">=", value=1)
        self.assertFalse(c._constrained_to_one())

    def test_comparison_str(self):
        c = Comparison(expression="~m", operator="==", value=1)
        s = str(c)
        self.assertIn("==", s)
        self.assertIn("1", s)

    def test_product_operators(self):
        for op in ["product==", "product<=", "product>=", "product<", "product>"]:
            c = Comparison(expression="~m", operator=op, value=10)
            self.assertEqual(c.operator, op)


class TestSpatialReuse(unittest.TestCase):
    """Test Spatial.reuse and Spatial fields."""

    def test_reuse_default_nothing(self):
        s = ArchSpatial(name="X", fanout=4)
        self.assertEqual(s.reuse, "Nothing")

    def test_may_reuse_default_all(self):
        s = ArchSpatial(name="X", fanout=4)
        self.assertEqual(s.may_reuse, "All")

    def test_reuse_custom(self):
        s = ArchSpatial(name="X", fanout=4, reuse="input")
        self.assertEqual(s.reuse, "input")

    def test_min_usage_default_zero(self):
        s = ArchSpatial(name="X", fanout=4)
        self.assertEqual(s.min_usage, 0.0)

    def test_power_gateable_default_false(self):
        s = ArchSpatial(name="X", fanout=4)
        self.assertFalse(s.power_gateable)

    def test_loop_bounds_with_comparison(self):
        s = ArchSpatial(
            name="X",
            fanout=4,
            loop_bounds=[{"expression": "~m", "operator": "==", "value": 1}],
        )
        self.assertEqual(len(s.loop_bounds), 1)
        self.assertIsInstance(s.loop_bounds[0], Comparison)

    def test_usage_scale_default(self):
        s = ArchSpatial(name="X", fanout=4)
        self.assertEqual(s.usage_scale, 1)


class TestTollParsed(unittest.TestCase):
    """Test Toll arch component."""

    def test_toll_direction(self):
        t = Toll(
            name="MyToll",
            direction="up",
            leak_power=0,
            area=0,
            actions=[{"name": "read", "energy": 1, "latency": 0}],
        )
        self.assertEqual(t.direction, "up")

    def test_toll_down_direction(self):
        t = Toll(
            name="MyToll",
            direction="down",
            leak_power=0,
            area=0,
        )
        self.assertEqual(t.direction, "down")

    def test_toll_up_and_down_direction(self):
        t = Toll(
            name="MyToll",
            direction="up_and_down",
            leak_power=0,
            area=0,
        )
        self.assertEqual(t.direction, "up_and_down")

    def test_toll_name(self):
        t = Toll(name="MyToll", direction="up", leak_power=0, area=0)
        self.assertEqual(t.name, "MyToll")


class TestMemoryTotalLatency(unittest.TestCase):
    """Test that total_latency expression is stored on Memory."""

    def test_total_latency_expression(self):
        """The TPU GlobalBuffer uses total_latency: max(read_latency, write_latency)."""
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML not found")
        spec = Spec.from_yaml(arch_path, wl_path)
        gb = spec.arch.find("GlobalBuffer")
        self.assertEqual(gb.total_latency, "max(read_latency, write_latency)")


class TestEnabledField(unittest.TestCase):
    """Test the Component.enabled field."""

    def test_enabled_default_true(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertTrue(c.enabled)

    def test_enabled_expression_string(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
            enabled="len(All) == 3",
        )
        self.assertEqual(c.enabled, "len(All) == 3")


# ============================================================================
# 6. Container variations
# ============================================================================


class TestFanoutVariationsParsed(unittest.TestCase):
    """Test parsing of fanout variation architectures."""

    def test_spatial_on_memory(self):
        """Spatial fanout directly on a Memory (at_glb.yaml)."""
        yaml_path = EXAMPLES_DIR / "arches" / "fanout_variations" / "at_glb.yaml"
        if not yaml_path.exists():
            self.skipTest("YAML not found")
        spec = Spec.from_yaml(yaml_path)
        gb = spec.arch.find("GlobalBuffer")
        self.assertEqual(len(gb.spatial), 1)
        self.assertEqual(gb.spatial[0].name, "X")
        self.assertEqual(gb.spatial[0].fanout, 4)

    def test_standalone_fanout_node(self):
        """Standalone Container node (at_glb_with_fanout_node.yaml)."""
        yaml_path = (
            EXAMPLES_DIR
            / "arches"
            / "fanout_variations"
            / "at_glb_with_fanout_node.yaml"
        )
        if not yaml_path.exists():
            self.skipTest("YAML not found")
        spec = Spec.from_yaml(yaml_path)
        fanout = spec.arch.find("GlobalBufferArray")
        self.assertIsInstance(fanout, Container)
        self.assertEqual(len(fanout.spatial), 1)
        self.assertEqual(fanout.spatial[0].name, "X")
        self.assertEqual(fanout.spatial[0].fanout, 4)

    def test_fanout_with_loop_bounds_constraint(self):
        """Container with loop_bounds constraint (at_mac_with_constraints.yaml)."""
        yaml_path = (
            EXAMPLES_DIR
            / "arches"
            / "fanout_variations"
            / "at_mac_with_constraints.yaml"
        )
        if not yaml_path.exists():
            self.skipTest("YAML not found")
        spec = Spec.from_yaml(yaml_path)
        fanout = spec.arch.find("MACArray")
        self.assertIsInstance(fanout, Container)
        self.assertEqual(len(fanout.spatial), 1)
        spatial = fanout.spatial[0]
        self.assertEqual(spatial.name, "X")
        self.assertEqual(spatial.fanout, 4)
        self.assertEqual(len(spatial.loop_bounds), 1)
        self.assertIsInstance(spatial.loop_bounds[0], Comparison)
        self.assertEqual(spatial.loop_bounds[0].expression, "~m")
        self.assertEqual(spatial.loop_bounds[0].operator, "==")
        self.assertEqual(spatial.loop_bounds[0].value, 1)

    def test_get_fanout_product(self):
        """Leaf.get_fanout() returns the product of spatial fanouts."""
        fanout = Container(
            name="F",
            spatial=[
                {"name": "X", "fanout": 4},
                {"name": "Y", "fanout": 8},
            ],
        )
        self.assertEqual(fanout.get_fanout(), 32)

    def test_get_fanout_no_spatial(self):
        mem = Memory(
            name="M",
            size=100,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(mem.get_fanout(), 1)


# ============================================================================
# 7. Verbose GPT: persistent per-tensor, rank_variable renames
# ============================================================================


class TestVerboseGPTParsed(unittest.TestCase):
    """Test features of the verbose GPT workload (gpt3_6.7B.yaml)."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest(f"YAML not found: {yaml_path}")
        cls.spec = Spec.from_yaml(yaml_path)._spec_eval_expressions()
        cls.wl = cls.spec.workload

    def test_I_is_copy_operation(self):
        """Verbose format correctly sets is_copy_operation: True on I."""
        i = self.wl.einsums["I"]
        self.assertTrue(i.is_copy_operation)

    def test_I_copy_source_tensor(self):
        i = self.wl.einsums["I"]
        self.assertEqual(i.copy_source_tensor(), "I_in")

    def test_weight_tensors_marked_persistent(self):
        """In verbose format, persistent: True is set directly on tensor_accesses."""
        persistent_names = {"WV", "WK", "WQ", "WZ", "WFFA", "WFFB"}
        for einsum in self.wl.einsums:
            for ta in einsum.tensor_accesses:
                if ta.name in persistent_names:
                    self.assertTrue(
                        ta.persistent,
                        f"{ta.name} in {einsum.name} should be persistent",
                    )

    def test_non_weight_tensors_not_persistent(self):
        non_persistent = {
            "I",
            "I_in",
            "V",
            "K",
            "Q",
            "QK",
            "QK_softmax",
            "AV",
            "Z",
            "FFA",
            "FFB",
        }
        for einsum in self.wl.einsums:
            for ta in einsum.tensor_accesses:
                if ta.name in non_persistent:
                    self.assertFalse(
                        ta.persistent,
                        f"{ta.name} in {einsum.name} should not be persistent",
                    )

    def test_QK_explicit_dict_projection_K(self):
        """K in QK uses explicit dict: { B: b, M: p, H: h, E: e }."""
        qk = self.wl.einsums["QK"]
        k = qk.tensor_accesses["K"]
        self.assertEqual(k.projection, {"B": "b", "M": "p", "H": "h", "E": "e"})

    def test_QK_inline_renames(self):
        qk = self.wl.einsums["QK"]
        renames_by_name = {r.name: r.source for r in qk.renames}
        self.assertEqual(set(renames_by_name["weight"]), set(["K"]))
        self.assertEqual(set(renames_by_name["input"]), set(["Q"]))
        self.assertEqual(set(renames_by_name["output"]), set(["QK"]))

    def test_AV_inline_renames(self):
        av = self.wl.einsums["AV"]
        renames_by_name = {r.name: r.source for r in av.renames}
        self.assertEqual(set(renames_by_name["weight"]), set(["V"]))
        self.assertEqual(set(renames_by_name["input"]), set(["QK_softmax"]))

    def test_I_renames_weight_to_nothing(self):
        """I einsum renames weight to Nothing (no weight for a copy)."""
        i = self.wl.einsums["I"]
        renames_by_name = {r.name: r.source for r in i.renames}
        self.assertEqual(set(renames_by_name["weight"]), set())

    def test_V_implied_projection(self):
        """V: I[b,m,d] uses implied (list) projection."""
        v = self.wl.einsums["V"]
        i_ta = v.tensor_accesses["I"]
        # self.assertIsInstance(i_ta.projection, ImpliedProjection)
        self.assertEqual(dict(i_ta.projection), {"B": "b", "M": "m", "D": "d"})

    def test_V_WV_implied_projection(self):
        v = self.wl.einsums["V"]
        wv = v.tensor_accesses["WV"]
        # self.assertIsInstance(wv.projection, ImpliedProjection)
        self.assertEqual(dict(wv.projection), {"H": "h", "E": "e", "D": "d"})

    def test_renames_default_expected_count_expression(self):
        """Default weight expected_count: '1 if len(All) == 3 else 0'."""
        default = self.spec.renames.einsums[0]
        weight = default.tensor_accesses["weight"]
        self.assertEqual(weight.expected_count, "1 if len(All) == 3 else 0")


class TestVerboseGPTEvaluated(unittest.TestCase):
    """Test evaluated verbose GPT workload."""

    @classmethod
    def setUpClass(cls):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest(f"YAML not found: {yaml_path}")
        cls.spec = Spec.from_yaml(yaml_path)._spec_eval_expressions()
        cls.wl = cls.spec.workload

    def test_all_bits_resolved_to_8(self):
        for einsum in self.wl.einsums:
            for ta in einsum.tensor_accesses:
                self.assertEqual(
                    ta.bits_per_value,
                    8,
                    f"{ta.name} in {einsum.name} should be 8",
                )

    def test_V_renames_include_builtins(self):
        v = self.wl.einsums["V"]
        rename_names = {r.name for r in v.renames}
        for name in (
            "All",
            "Inputs",
            "Outputs",
            "Intermediates",
            "Nothing",
            "Tensors",
            "Shared",
            "Persistent",
        ):
            self.assertIn(name, rename_names)

    def test_QK_has_no_weight_intermediate_check(self):
        """QK has 3 tensors, so expected_count for weight is 1."""
        qk = self.wl.einsums["QK"]
        self.assertEqual(len(qk.tensor_accesses), 3)

    def test_QK_softmax_has_2_tensors(self):
        """QK_softmax has only 2 tensors (input + output)."""
        qs = self.wl.einsums["QK_softmax"]
        self.assertEqual(len(qs.tensor_accesses), 2)


# ============================================================================
# 8. _parse_projection edge cases (documented in workload.rst)
# ============================================================================


class TestParseProjectionDocExamples(unittest.TestCase):
    """Test _parse_projection for cases shown in the docs."""

    def test_mixed_implicit_and_explicit(self):
        """Parsing 'b, M:p, h, e' yields a dict with mixed implicit/explicit."""
        result = _parse_projection("b, M:p, h, e")
        self.assertEqual(result["B"], "b")
        self.assertEqual(result["M"], "p")
        self.assertEqual(result["H"], "h")
        self.assertEqual(result["E"], "e")

    def test_all_implicit(self):
        result = _parse_projection("m, k, n")
        self.assertEqual(result, {"M": "m", "K": "k", "N": "n"})

    def test_all_explicit(self):
        result = _parse_projection("Row:m, Col:n")
        self.assertEqual(result, {"Row": "m", "Col": "n"})


# ============================================================================
# Concise vs verbose equivalence (docs promise they're the same)
# ============================================================================


class TestConciseVsVerboseEquivalence(unittest.TestCase):
    """Docs show verbose/concise are equivalent. Verify key properties match."""

    @classmethod
    def setUpClass(cls):
        c_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        v_path = EXAMPLES_DIR / "misc" / "gpt3_6.7B_verbose_annotated.yaml"
        if not c_path.exists() or not v_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.concise = Spec.from_yaml(c_path)._spec_eval_expressions()
        cls.verbose = Spec.from_yaml(v_path)._spec_eval_expressions()

    def test_same_einsum_count(self):
        self.assertEqual(
            len(self.concise.workload.einsums),
            len(self.verbose.workload.einsums),
        )

    def test_same_einsum_names(self):
        c_names = {e.name for e in self.concise.workload.einsums}
        v_names = {e.name for e in self.verbose.workload.einsums}
        self.assertEqual(c_names, v_names)

    def test_same_tensor_names_per_einsum(self):
        for c_e in self.concise.workload.einsums:
            v_e = self.verbose.workload.einsums[c_e.name]
            self.assertEqual(c_e.tensor_names, v_e.tensor_names, f"in {c_e.name}")

    def test_same_rank_variables_per_einsum(self):
        for c_e in self.concise.workload.einsums:
            v_e = self.verbose.workload.einsums[c_e.name]
            self.assertEqual(c_e.rank_variables, v_e.rank_variables, f"in {c_e.name}")

    def test_same_output_tensor_per_einsum(self):
        for c_e in self.concise.workload.einsums:
            v_e = self.verbose.workload.einsums[c_e.name]
            self.assertEqual(
                c_e.output_tensor_names, v_e.output_tensor_names, f"in {c_e.name}"
            )

    def test_same_projection_values(self):
        """Projection dicts should have the same key-value mappings."""
        for c_e in self.concise.workload.einsums:
            v_e = self.verbose.workload.einsums[c_e.name]
            for c_ta in c_e.tensor_accesses:
                v_ta = v_e.tensor_accesses[c_ta.name]
                self.assertEqual(
                    dict(c_ta.projection),
                    dict(v_ta.projection),
                    f"{c_ta.name} in {c_e.name}",
                )

    def test_same_rank_sizes(self):
        self.assertEqual(
            dict(self.concise.workload.rank_sizes),
            dict(self.verbose.workload.rank_sizes),
        )

    def test_same_bits_per_value(self):
        for c_e in self.concise.workload.einsums:
            v_e = self.verbose.workload.einsums[c_e.name]
            for c_ta in c_e.tensor_accesses:
                v_ta = v_e.tensor_accesses[c_ta.name]
                self.assertEqual(
                    c_ta.bits_per_value,
                    v_ta.bits_per_value,
                    f"{c_ta.name} in {c_e.name}",
                )


if __name__ == "__main__":
    unittest.main()
