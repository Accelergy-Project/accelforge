"""
Tests for workload model classes:
  - TensorAccess (properties: rank_variables, ranks, rank2rank_variables, etc.)
  - ImpliedProjection
  - Shape
  - Einsum (properties, validation, tensor_accesses)
  - Workload (validation, properties, einsums_with_tensor, etc.)
"""

import unittest

from accelforge.frontend.workload import (
    TensorAccess,
    ImpliedProjection,
    Shape,
    Einsum,
    Workload,
)


# ============================================================================
# TensorAccess
# ============================================================================


class TestTensorAccess(unittest.TestCase):
    """Tests for TensorAccess model and its properties."""

    def _make_access(self, name="T", projection=None, output=False, **kwargs):
        if projection is None:
            projection = ["m", "n"]
        return TensorAccess(name=name, projection=projection, output=output, **kwargs)

    def test_list_projection_becomes_implied(self):
        ta = self._make_access(projection=["a", "b", "c"])
        self.assertIsInstance(ta.projection, ImpliedProjection)
        self.assertEqual(ta.projection, {"A": "a", "B": "b", "C": "c"})

    def test_dict_projection_kept(self):
        ta = self._make_access(projection={"M": "m", "N": "n"})
        self.assertNotIsInstance(ta.projection, ImpliedProjection)
        self.assertEqual(ta.projection, {"M": "m", "N": "n"})

    def test_ranks_property(self):
        ta = self._make_access(projection=["a", "b", "c"])
        self.assertEqual(ta.ranks, ("A", "B", "C"))

    def test_ranks_property_dict(self):
        ta = self._make_access(projection={"X": "x", "Y": "y"})
        self.assertEqual(ta.ranks, ("X", "Y"))

    def test_rank_variables_property(self):
        ta = self._make_access(projection=["m", "n", "k"])
        self.assertEqual(ta.rank_variables, {"m", "n", "k"})

    def test_rank_variables_with_expression(self):
        ta = self._make_access(projection={"M": "m+n", "K": "k"})
        self.assertEqual(ta.rank_variables, {"m", "n", "k"})

    def test_directly_indexing_rank_variables(self):
        ta = self._make_access(projection={"M": "m", "N": "n"})
        self.assertEqual(ta.directly_indexing_rank_variables, {"m", "n"})

    def test_directly_indexing_simple_only(self):
        """Only simple single-identifier projections count as directly indexing."""
        ta = self._make_access(projection={"M": "m", "N": "n"})
        # Both are simple direct indices
        self.assertEqual(ta.directly_indexing_rank_variables, {"m", "n"})
        self.assertEqual(ta.expression_indexing_rank_variables, set())

    def test_expression_indexing_rank_variables(self):
        """Rank variables that appear only in expressions (not as standalone indices)."""
        ta = self._make_access(projection={"M": "m", "K": "k"})
        self.assertEqual(ta.expression_indexing_rank_variables, set())

    def test_expression_indexing_with_complex_projection(self):
        """Expression-based projections have rank variables extracted from expressions."""
        ta = self._make_access(projection={"M": "m", "N": "n"})
        # All are direct, none are expression-based
        all_rv = ta.rank_variables
        direct = ta.directly_indexing_rank_variables
        expr = ta.expression_indexing_rank_variables
        self.assertEqual(expr, all_rv - direct)

    def test_rank2rank_variables(self):
        ta = self._make_access(projection={"M": "m", "N": "n"})
        result = ta.rank2rank_variables
        self.assertEqual(result["M"], {"m"})
        self.assertEqual(result["N"], {"n"})

    def test_rank_variable2ranks(self):
        ta = self._make_access(projection={"M": "m", "N": "m"})
        result = ta.rank_variable2ranks
        self.assertEqual(result["m"], {"M", "N"})

    def test_default_output_false(self):
        ta = self._make_access()
        self.assertFalse(ta.output)

    def test_output_true(self):
        ta = self._make_access(output=True)
        self.assertTrue(ta.output)

    def test_default_persistent_false(self):
        ta = self._make_access()
        self.assertFalse(ta.persistent)

    def test_bits_per_value_default_none(self):
        ta = self._make_access()
        self.assertIsNone(ta.bits_per_value)

    def test_bits_per_value_set(self):
        ta = self._make_access(bits_per_value=16)
        self.assertEqual(ta.bits_per_value, 16)

    def test_backing_storage_size_scale_default(self):
        ta = self._make_access()
        self.assertEqual(ta.backing_storage_size_scale, 1.0)

    def test_to_formatted_string(self):
        ta = self._make_access(name="W", projection=["k", "n"])
        s = ta._to_formatted_string()
        self.assertIn("W", s)

    def test_to_formatted_string_with_dict_projection(self):
        ta = self._make_access(name="K", projection={"B": "b", "M": "p"})
        s = ta._to_formatted_string()
        self.assertIn("K", s)


# ============================================================================
# ImpliedProjection
# ============================================================================


class TestImpliedProjection(unittest.TestCase):
    """Tests for ImpliedProjection subclass of dict."""

    def test_is_dict(self):
        ip = ImpliedProjection({"A": "a", "B": "b"})
        self.assertIsInstance(ip, dict)
        self.assertIsInstance(ip, ImpliedProjection)

    def test_values(self):
        ip = ImpliedProjection({"M": "m", "N": "n"})
        self.assertEqual(ip["M"], "m")
        self.assertEqual(ip["N"], "n")

    def test_empty(self):
        ip = ImpliedProjection()
        self.assertEqual(len(ip), 0)


# ============================================================================
# Shape
# ============================================================================


class TestShape(unittest.TestCase):
    """Tests for the Shape class (list of ISL expressions)."""

    def test_rank_variables_from_expressions(self):
        s = Shape(["0 <= m < 10", "0 <= n < 20"])
        self.assertEqual(s.rank_variables, {"m", "n"})

    def test_rank_variables_complex(self):
        s = Shape(["0 <= a + b < 10"])
        self.assertEqual(s.rank_variables, {"a", "b"})

    def test_empty_shape(self):
        s = Shape()
        self.assertEqual(s.rank_variables, set())

    def test_single_expression(self):
        s = Shape(["0 <= x < 5"])
        self.assertEqual(s.rank_variables, {"x"})


# ============================================================================
# Einsum
# ============================================================================


class TestEinsum(unittest.TestCase):
    """Tests for the Einsum model."""

    def _make_einsum(self, name="Matmul", **kwargs):
        defaults = {
            "tensor_accesses": [
                {"name": "A", "projection": ["m", "k"]},
                {"name": "B", "projection": ["k", "n"]},
                {"name": "C", "projection": ["m", "n"], "output": True},
            ],
        }
        defaults.update(kwargs)
        return Einsum(name=name, **defaults)

    def test_basic_creation(self):
        e = self._make_einsum()
        self.assertEqual(e.name, "Matmul")
        self.assertEqual(len(e.tensor_accesses), 3)

    def test_rank_variables(self):
        e = self._make_einsum()
        self.assertEqual(e.rank_variables, {"m", "k", "n"})

    def test_ranks(self):
        e = self._make_einsum()
        self.assertEqual(e.ranks, {"M", "K", "N"})

    def test_input_tensor_names(self):
        e = self._make_einsum()
        self.assertEqual(e.input_tensor_names, {"A", "B"})

    def test_output_tensor_names(self):
        e = self._make_einsum()
        self.assertEqual(e.output_tensor_names, {"C"})

    def test_tensor_names(self):
        e = self._make_einsum()
        self.assertEqual(e.tensor_names, {"A", "B", "C"})

    def test_tensor2rank_variables(self):
        e = self._make_einsum()
        t2rv = e.tensor2rank_variables
        self.assertEqual(t2rv["A"], {"m", "k"})
        self.assertEqual(t2rv["B"], {"k", "n"})
        self.assertEqual(t2rv["C"], {"m", "n"})

    def test_tensor2directly_indexing_rank_variables(self):
        e = self._make_einsum()
        t2di = e.tensor2directly_indexing_rank_variables
        self.assertEqual(t2di["A"], {"m", "k"})

    def test_tensor2irrelevant_rank_variables(self):
        e = self._make_einsum()
        t2irr = e.tensor2irrelevant_rank_variables
        self.assertEqual(t2irr["A"], {"n"})
        self.assertEqual(t2irr["B"], {"m"})
        self.assertEqual(t2irr["C"], {"k"})

    def test_rank_variable2ranks(self):
        e = self._make_einsum()
        rv2r = e.rank_variable2ranks
        self.assertEqual(rv2r["m"], {"M"})
        self.assertEqual(rv2r["k"], {"K"})
        self.assertEqual(rv2r["n"], {"N"})

    def test_indexing_expressions(self):
        e = self._make_einsum()
        self.assertEqual(e.indexing_expressions, {"m", "k", "n"})

    def test_reserved_name_total_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._make_einsum(name="Total")
        self.assertIn("reserved", str(ctx.exception).lower())

    def test_n_instances_default(self):
        e = self._make_einsum()
        self.assertEqual(e.n_instances, 1)

    def test_n_instances_custom(self):
        e = self._make_einsum(n_instances=5)
        self.assertEqual(e.n_instances, 5)

    def test_is_copy_operation_default_false(self):
        e = self._make_einsum()
        self.assertFalse(e.is_copy_operation)

    def test_copy_source_tensor_none_if_not_copy(self):
        e = self._make_einsum()
        self.assertIsNone(e.copy_source_tensor())

    def test_copy_source_tensor(self):
        e = Einsum(
            name="Copy",
            tensor_accesses=[
                {"name": "Src", "projection": ["m"]},
                {"name": "Dst", "projection": ["m"], "output": True},
            ],
            is_copy_operation=True,
        )
        self.assertEqual(e.copy_source_tensor(), "Src")

    def test_copy_with_multiple_inputs_raises(self):
        e = Einsum(
            name="Copy",
            tensor_accesses=[
                {"name": "Src1", "projection": ["m"]},
                {"name": "Src2", "projection": ["m"]},
                {"name": "Dst", "projection": ["m"], "output": True},
            ],
            is_copy_operation=True,
        )
        with self.assertRaises(ValueError):
            e.copy_source_tensor()

    def test_renames_as_dict(self):
        e = self._make_einsum(renames={"input": "A", "output": "C"})
        self.assertEqual(len(e.renames), 2)

    def test_renames_as_list(self):
        e = self._make_einsum(renames=[])
        self.assertEqual(len(e.renames), 0)

    def test_empty_tensor_accesses_rank_variables(self):
        e = Einsum(name="Empty", tensor_accesses=[])
        self.assertEqual(e.rank_variables, set())
        self.assertEqual(e.ranks, set())

    def test_to_formatted_string(self):
        e = self._make_einsum()
        s = e._to_formatted_string()
        self.assertIn("C", s)
        self.assertIn("A", s)


# ============================================================================
# Workload
# ============================================================================


class TestWorkload(unittest.TestCase):
    """Tests for the Workload model."""

    def _make_workload(self, **kwargs):
        defaults = {
            "rank_sizes": {"M": 64, "K": 32, "N": 64},
            "bits_per_value": {"All": 8},
            "einsums": [
                {
                    "name": "Matmul",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "k"]},
                        {"name": "B", "projection": ["k", "n"]},
                        {"name": "C", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        }
        defaults.update(kwargs)
        return Workload(**defaults)

    def test_basic_creation(self):
        w = self._make_workload()
        self.assertEqual(len(w.einsums), 1)
        self.assertEqual(w.einsums[0].name, "Matmul")

    def test_einsum_names(self):
        w = self._make_workload()
        self.assertEqual(w.einsum_names, ["Matmul"])

    def test_tensor_names(self):
        w = self._make_workload()
        self.assertEqual(w.tensor_names, {"A", "B", "C"})

    def test_rank_variables(self):
        w = self._make_workload()
        self.assertEqual(w.rank_variables, {"m", "k", "n"})

    def test_einsums_with_tensor(self):
        w = self._make_workload()
        self.assertEqual(len(w.einsums_with_tensor("A")), 1)
        self.assertEqual(w.einsums_with_tensor("A")[0].name, "Matmul")
        self.assertEqual(len(w.einsums_with_tensor("NotHere")), 0)

    def test_einsums_with_tensor_as_input(self):
        w = self._make_workload()
        self.assertEqual(len(w.einsums_with_tensor_as_input("A")), 1)
        self.assertEqual(len(w.einsums_with_tensor_as_input("C")), 0)

    def test_einsums_with_tensor_as_output(self):
        w = self._make_workload()
        self.assertEqual(len(w.einsums_with_tensor_as_output("C")), 1)
        self.assertEqual(len(w.einsums_with_tensor_as_output("A")), 0)

    def test_accesses_for_tensor(self):
        w = self._make_workload()
        accesses = w.accesses_for_tensor("A")
        self.assertEqual(len(accesses), 1)
        self.assertEqual(accesses[0].name, "A")

    def test_n_instances_default(self):
        w = self._make_workload()
        self.assertEqual(w.n_instances, 1)

    def test_duplicate_einsum_names_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Workload(
                rank_sizes={"B": 4, "M": 8},
                bits_per_value={"All": 8},
                einsums=[
                    "V[b, m] = I[b]",
                    "V[b, m] = X[b]",
                ],
            )
        self.assertIn("not unique", str(ctx.exception))

    def test_inconsistent_tensor_ranks_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Workload(
                rank_sizes={"B": 4, "M": 8, "D": 16},
                bits_per_value={"All": 8},
                einsums=[
                    "V[b, m] = I[b]",
                    "K[b, m, d] = I[b, d]",
                ],
            )
        self.assertIn("inconsistent ranks", str(ctx.exception))

    def test_concise_string_einsums(self):
        w = Workload(
            rank_sizes={"B": 4, "M": 8, "D": 16},
            bits_per_value={"All": 8},
            einsums=[
                "V[b, m] = I[b, m] * W[m]",
            ],
        )
        self.assertEqual(len(w.einsums), 1)
        self.assertEqual(w.einsums[0].name, "V")

    def test_multi_einsum_workload(self):
        w = Workload(
            rank_sizes={"M": 64, "N0": 32, "N1": 32, "N2": 32},
            bits_per_value={"All": 8},
            einsums=[
                {
                    "name": "Matmul1",
                    "tensor_accesses": [
                        {"name": "T0", "projection": ["m", "n0"]},
                        {"name": "W0", "projection": ["n0", "n1"]},
                        {"name": "T1", "projection": ["m", "n1"], "output": True},
                    ],
                },
                {
                    "name": "Matmul2",
                    "tensor_accesses": [
                        {"name": "T1", "projection": ["m", "n1"]},
                        {"name": "W1", "projection": ["n1", "n2"]},
                        {"name": "T2", "projection": ["m", "n2"], "output": True},
                    ],
                },
            ],
        )
        self.assertEqual(len(w.einsums), 2)
        self.assertEqual(w.tensor_names_used_in_multiple_einsums, {"T1"})

    def test_tensor_copies(self):
        w = Workload(
            rank_sizes={"M": 64},
            bits_per_value={"All": 8},
            einsums=[
                {
                    "name": "Copy",
                    "tensor_accesses": [
                        {"name": "Src", "projection": ["m"]},
                        {"name": "Dst", "projection": ["m"], "output": True},
                    ],
                    "is_copy_operation": True,
                },
            ],
        )
        copies = w.get_tensor_copies()
        self.assertIn("Src", copies)
        self.assertIn("Dst", copies["Src"])

    def test_empty_workload(self):
        w = Workload()
        self.assertEqual(len(w.einsums), 0)
        self.assertEqual(w.tensor_names, set())
        self.assertEqual(w.rank_variables, set())

    def test_iteration_space_shape(self):
        w = Workload(
            iteration_space_shape={"m": "0 <= m < 64", "n": "0 <= n < 32"},
            rank_sizes={"M": 64, "N": 32},
            bits_per_value={"All": 8},
            einsums=[
                {
                    "name": "E",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m"]},
                        {"name": "B", "projection": ["m", "n"], "output": True},
                    ],
                },
            ],
        )
        self.assertIn("m", w.iteration_space_shape)

    def test_persistent_tensors_field(self):
        w = Workload(
            rank_sizes={"M": 64},
            bits_per_value={"All": 8},
            persistent_tensors="weight - Intermediates",
            einsums=[],
        )
        self.assertEqual(w.persistent_tensors, "weight - Intermediates")


if __name__ == "__main__":
    unittest.main()
