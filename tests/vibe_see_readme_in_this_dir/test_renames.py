"""
Tests for rename frontend classes:
  - Rename
  - RenameList
  - rename_list_factory
  - EinsumRename
  - Renames
"""

import unittest

from accelforge.frontend.renames import (
    Rename,
    RenameList,
    rename_list_factory,
    EinsumRename,
    Renames,
)


# ============================================================================
# Rename
# ============================================================================


class TestRename(unittest.TestCase):
    """Tests for the Rename model."""

    def test_basic_creation(self):
        r = Rename(name="weight", source="W")
        self.assertEqual(r.name, "weight")
        self.assertEqual(r.source, "W")

    def test_expected_count_none_default(self):
        r = Rename(name="input", source="I")
        self.assertIsNone(r.expected_count)

    def test_expected_count_set(self):
        r = Rename(name="output", source="Outputs", expected_count=1)
        self.assertEqual(r.expected_count, 1)

    def test_set_expression_source(self):
        r = Rename(name="input", source="Inputs & Intermediates")
        self.assertEqual(r.source, "Inputs & Intermediates")

    def test_complement_set_expression(self):
        r = Rename(name="weight", source="~(input | output)")
        self.assertEqual(r.source, "~(input | output)")


# ============================================================================
# rename_list_factory
# ============================================================================


class TestRenameListFactory(unittest.TestCase):
    """Tests for rename_list_factory."""

    def test_dict_input(self):
        result = rename_list_factory({"input": "I", "output": "O"})
        self.assertIsInstance(result, RenameList)
        self.assertEqual(len(result), 2)
        names = {r.name for r in result}
        self.assertEqual(names, {"input", "output"})

    def test_list_input(self):
        renames = [
            Rename(name="input", source="I"),
            Rename(name="output", source="O"),
        ]
        result = rename_list_factory(renames)
        self.assertIsInstance(result, RenameList)
        self.assertEqual(len(result), 2)

    def test_empty_dict(self):
        result = rename_list_factory({})
        self.assertIsInstance(result, RenameList)
        self.assertEqual(len(result), 0)

    def test_empty_list(self):
        result = rename_list_factory([])
        self.assertIsInstance(result, RenameList)
        self.assertEqual(len(result), 0)

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            rename_list_factory(123)

    def test_invalid_type_string_raises(self):
        with self.assertRaises(TypeError):
            rename_list_factory("not_valid")

    def test_dict_preserves_sources(self):
        result = rename_list_factory({"weight": "~(input | output)"})
        self.assertEqual(result[0].source, "~(input | output)")
        self.assertIsNone(result[0].expected_count)


# ============================================================================
# RenameList
# ============================================================================


class TestRenameList(unittest.TestCase):
    """Tests for RenameList."""

    def test_creation(self):
        rl = RenameList(
            [
                Rename(name="a", source="A"),
                Rename(name="b", source="B"),
            ]
        )
        self.assertEqual(len(rl), 2)

    def test_dict_conversion(self):
        rl = RenameList(
            [
                Rename(name="a", source="A"),
                Rename(name="b", source="B"),
            ]
        )
        d = rl.__dict__()
        self.assertEqual(d, {"a": "A", "b": "B"})

    def test_empty_rename_list(self):
        rl = RenameList()
        self.assertEqual(len(rl), 0)

    def test_contains_by_name(self):
        rl = RenameList(
            [
                Rename(name="input", source="I"),
            ]
        )
        # The __contains__ for EvalableList uses name
        self.assertIn("input", rl)
        self.assertNotIn("output", rl)


# ============================================================================
# EinsumRename
# ============================================================================


class TestEinsumRename(unittest.TestCase):
    """Tests for EinsumRename."""

    def test_basic_creation(self):
        er = EinsumRename(name="Matmul")
        self.assertEqual(er.name, "Matmul")
        self.assertEqual(len(er.tensor_accesses), 0)
        self.assertEqual(len(er.rank_variables), 0)

    def test_with_tensor_accesses_dict(self):
        er = EinsumRename(
            name="Matmul",
            tensor_accesses={"input": "A", "output": "C"},
        )
        self.assertEqual(len(er.tensor_accesses), 2)

    def test_with_tensor_accesses_list(self):
        er = EinsumRename(
            name="Matmul",
            tensor_accesses=[
                Rename(name="input", source="A"),
                Rename(name="output", source="C"),
            ],
        )
        self.assertEqual(len(er.tensor_accesses), 2)

    def test_with_rank_variables_dict(self):
        er = EinsumRename(
            name="Matmul",
            rank_variables={"reduced": "k"},
        )
        self.assertEqual(len(er.rank_variables), 1)

    def test_default_name(self):
        er = EinsumRename(name="default")
        self.assertEqual(er.name, "default")


# ============================================================================
# Renames
# ============================================================================


class TestRenames(unittest.TestCase):
    """Tests for the top-level Renames model."""

    def test_empty_renames(self):
        r = Renames()
        self.assertEqual(len(r.einsums), 0)

    def test_with_default_einsum_rename(self):
        r = Renames(
            einsums=[
                EinsumRename(
                    name="default",
                    tensor_accesses={"input": "Inputs & Intermediates"},
                ),
            ]
        )
        self.assertEqual(len(r.einsums), 1)

    def test_get_renames_for_einsum_default_applied(self):
        r = Renames(
            einsums=[
                EinsumRename(
                    name="default",
                    tensor_accesses=[
                        Rename(name="input", source="Inputs & Intermediates"),
                    ],
                ),
            ]
        )
        result = r.get_renames_for_einsum("SomeEinsum")
        self.assertEqual(result.name, "SomeEinsum")
        # The default tensor rename should be applied
        self.assertIn("input", result.tensor_accesses)

    def test_get_renames_for_einsum_without_default(self):
        r = Renames()
        result = r.get_renames_for_einsum("SomeEinsum")
        self.assertEqual(result.name, "SomeEinsum")
        self.assertEqual(len(result.tensor_accesses), 0)

    def test_non_default_einsum_renames_applied_at_eval_time(self):
        """Non-default einsum renames are only resolved during full spec
        evaluation (name-based lookup requires EvalableList). Pre-evaluation,
        get_renames_for_einsum only applies defaults."""
        r = Renames(
            einsums=[
                EinsumRename(
                    name="Matmul",
                    tensor_accesses=[
                        Rename(name="weight", source="W"),
                    ],
                ),
            ]
        )
        # Without evaluation, 'Matmul' is not found in the plain list,
        # so a fresh EinsumRename is created with no tensor_accesses.
        result = r.get_renames_for_einsum("Matmul")
        self.assertEqual(result.name, "Matmul")

    def test_default_applied_when_no_specific_match(self):
        """When a specific einsum is not found, defaults are still applied."""
        r = Renames(
            einsums=[
                EinsumRename(
                    name="default",
                    tensor_accesses=[
                        Rename(name="input", source="DefaultInput"),
                    ],
                ),
            ]
        )
        result = r.get_renames_for_einsum("SomeOtherEinsum")
        input_rename = [rn for rn in result.tensor_accesses if rn.name == "input"]
        self.assertEqual(len(input_rename), 1)
        self.assertEqual(input_rename[0].source, "DefaultInput")


if __name__ == "__main__":
    unittest.main()
