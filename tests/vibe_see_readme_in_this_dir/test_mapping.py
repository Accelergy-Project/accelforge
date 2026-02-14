"""
Tests for mapping frontend classes:
  - MappingNode, MappingNodeWithChildren
  - Temporal, Spatial (mapping loops)
  - Storage, Toll (mapping tensor holders)
  - Compute (mapping compute)
  - TilePattern
  - Nested, Split, Sequential, Pipeline
  - Reservation
  - Mapping (top-level)
"""

import unittest

from accelforge.frontend.mapping.mapping import (
    Compute,
    Loop,
    Mapping,
    MappingNode,
    MappingNodeWithChildren,
    Nested,
    Pipeline,
    Reservation,
    Sequential,
    Spatial,
    Split,
    Storage,
    Temporal,
    TensorHolder,
    TilePattern,
    Toll,
)


# ============================================================================
# TilePattern
# ============================================================================


class TestTilePattern(unittest.TestCase):
    """Tests for TilePattern dataclass."""

    def test_default_creation(self):
        tp = TilePattern()
        self.assertEqual(tp.tile_shape, "symbol")
        self.assertEqual(tp.initial_tile_shape, "symbol")
        self.assertIsNone(tp.calculated_n_iterations)

    def test_with_values(self):
        tp = TilePattern(tile_shape=4, initial_tile_shape=8)
        self.assertEqual(tp.tile_shape, 4)
        self.assertEqual(tp.initial_tile_shape, 8)

    def test_str_with_iterations(self):
        tp = TilePattern(tile_shape=4, initial_tile_shape=8, calculated_n_iterations=3)
        s = str(tp)
        self.assertIn("3", s)
        self.assertIn("4", s)
        self.assertIn("8", s)

    def test_str_symbol_only(self):
        tp = TilePattern()
        s = str(tp)
        # Should be empty since all are "symbol"
        self.assertEqual(s, "")

    def test_update(self):
        tp = TilePattern(tile_shape=4)
        tp2 = tp.update(tile_shape=8)
        self.assertEqual(tp2.tile_shape, 8)
        self.assertEqual(tp.tile_shape, 4)  # Original unchanged

    def test_equality(self):
        tp1 = TilePattern(tile_shape=4, initial_tile_shape=8)
        tp2 = TilePattern(tile_shape=4, initial_tile_shape=8)
        self.assertEqual(tp1, tp2)

    def test_inequality(self):
        tp1 = TilePattern(tile_shape=4)
        tp2 = TilePattern(tile_shape=8)
        self.assertNotEqual(tp1, tp2)

    def test_hash(self):
        tp1 = TilePattern(tile_shape=4, initial_tile_shape=8)
        tp2 = TilePattern(tile_shape=4, initial_tile_shape=8)
        self.assertEqual(hash(tp1), hash(tp2))

    def test_clear_symbols(self):
        tp = TilePattern(tile_shape="x", initial_tile_shape="y")
        cleared = tp._clear_symbols()
        self.assertIsNone(cleared.tile_shape)
        self.assertIsNone(cleared.initial_tile_shape)

    def test_clear_symbols_keeps_ints(self):
        tp = TilePattern(tile_shape=4, initial_tile_shape=8)
        cleared = tp._clear_symbols()
        self.assertEqual(cleared.tile_shape, 4)
        self.assertEqual(cleared.initial_tile_shape, 8)

    def test_as_str_selective(self):
        tp = TilePattern(tile_shape=4, initial_tile_shape=8, calculated_n_iterations=2)
        s = tp.as_str(with_initial_tile_shape=False, with_tile_shape=True)
        self.assertNotIn("initial=", s)
        self.assertIn("tile_shape=4", s)

    def test_symbol_attrs(self):
        tp = TilePattern()
        attrs = tp._symbol_attrs()
        self.assertIn("tile_shape", attrs)
        self.assertIn("initial_tile_shape", attrs)
        self.assertIn("calculated_n_iterations", attrs)


# ============================================================================
# Temporal
# ============================================================================


class TestTemporal(unittest.TestCase):
    """Tests for Temporal loop node."""

    def test_basic_creation(self):
        t = Temporal(rank_variable="m", tile_shape=4)
        self.assertEqual(t.rank_variable, "m")
        self.assertEqual(t.tile_shape, 4)

    def test_str_representation(self):
        t = Temporal(rank_variable="m", tile_shape=4)
        s = str(t)
        self.assertIn("for", s)
        self.assertIn("m", s)

    def test_compact_str(self):
        t = Temporal(rank_variable="m", tile_shape=4)
        s = t.compact_str()
        self.assertIn("T-", s)
        self.assertIn("m", s)

    def test_tile_pattern(self):
        t = Temporal(rank_variable="m", tile_shape=4, initial_tile_shape=8)
        tp = t.tile_pattern
        self.assertEqual(tp.tile_shape, 4)
        self.assertEqual(tp.initial_tile_shape, 8)

    def test_equality(self):
        t1 = Temporal(rank_variable="m", tile_shape=4)
        t2 = Temporal(rank_variable="m", tile_shape=4)
        self.assertEqual(t1, t2)

    def test_inequality_different_rv(self):
        t1 = Temporal(rank_variable="m", tile_shape=4)
        t2 = Temporal(rank_variable="n", tile_shape=4)
        self.assertNotEqual(t1, t2)

    def test_inequality_different_tile(self):
        t1 = Temporal(rank_variable="m", tile_shape=4)
        t2 = Temporal(rank_variable="m", tile_shape=8)
        self.assertNotEqual(t1, t2)

    def test_merge_different_tile_raises(self):
        t1 = Temporal(rank_variable="m", tile_shape=4)
        t2 = Temporal(rank_variable="n", tile_shape=8)
        with self.assertRaises(ValueError):
            t1._merge(t2)

    def test_merge_non_temporal_raises(self):
        t1 = Temporal(rank_variable="m", tile_shape=4)
        s1 = Spatial(rank_variable="n", tile_shape=4, name="X", component="Mem")
        with self.assertRaises(ValueError):
            t1._merge(s1)


# ============================================================================
# Spatial (mapping)
# ============================================================================


class TestMappingSpatial(unittest.TestCase):
    """Tests for Spatial loop node in mapping."""

    def test_basic_creation(self):
        s = Spatial(rank_variable="m", tile_shape=4, name="X", component="Register")
        self.assertEqual(s.rank_variable, "m")
        self.assertEqual(s.name, "X")
        self.assertEqual(s.component, "Register")

    def test_str_representation(self):
        s = Spatial(rank_variable="m", tile_shape=4, name="X", component="Register")
        result = str(s)
        self.assertIn("S-X", result)
        self.assertIn("m", result)

    def test_compact_str(self):
        s = Spatial(rank_variable="m", tile_shape=4, name="X", component="Register")
        result = s.compact_str()
        self.assertIn("S-X", result)
        self.assertIn("m", result)

    def test_equality(self):
        s1 = Spatial(rank_variable="m", tile_shape=4, name="X", component="Register")
        s2 = Spatial(rank_variable="m", tile_shape=4, name="X", component="Register")
        self.assertEqual(s1, s2)

    def test_inequality_different_name(self):
        s1 = Spatial(rank_variable="m", tile_shape=4, name="X", component="Register")
        s2 = Spatial(rank_variable="m", tile_shape=4, name="Y", component="Register")
        self.assertNotEqual(s1, s2)

    def test_inequality_different_component(self):
        s1 = Spatial(rank_variable="m", tile_shape=4, name="X", component="Register")
        s2 = Spatial(rank_variable="m", tile_shape=4, name="X", component="Buffer")
        self.assertNotEqual(s1, s2)

    def test_merge_different_name_raises(self):
        s1 = Spatial(rank_variable="m", tile_shape=4, name="X", component="Reg")
        s2 = Spatial(rank_variable="n", tile_shape=4, name="Y", component="Reg")
        with self.assertRaises(ValueError):
            s1._merge(s2)


# ============================================================================
# Storage
# ============================================================================


class TestStorage(unittest.TestCase):
    """Tests for Storage mapping node."""

    def test_basic_creation(self):
        s = Storage(tensors=["A", "B"], component="MainMem")
        self.assertEqual(s.tensors, ["A", "B"])
        self.assertEqual(s.component, "MainMem")

    def test_str_representation(self):
        s = Storage(tensors=["A"], component="MainMem")
        result = str(s)
        self.assertIn("MainMem", result)
        self.assertIn("A", result)

    def test_compact_str(self):
        s = Storage(tensors=["A"], component="MainMem")
        result = s.compact_str()
        self.assertIn("A", result)
        self.assertIn("MainMem", result)

    def test_tensor_property_single(self):
        s = Storage(tensors=["A"], component="MainMem")
        self.assertEqual(s.tensor, "A")

    def test_tensor_property_multiple_raises(self):
        s = Storage(tensors=["A", "B"], component="MainMem")
        with self.assertRaises(ValueError):
            _ = s.tensor

    def test_equality(self):
        s1 = Storage(tensors=["A", "B"], component="MainMem")
        s2 = Storage(tensors=["A", "B"], component="MainMem")
        self.assertEqual(s1, s2)

    def test_equality_order_matters_for_lists(self):
        s1 = Storage(tensors=["A", "B"], component="MainMem")
        s2 = Storage(tensors=["B", "A"], component="MainMem")
        # set comparison in __eq__
        self.assertEqual(s1, s2)

    def test_inequality_different_component(self):
        s1 = Storage(tensors=["A"], component="MainMem")
        s2 = Storage(tensors=["A"], component="Buffer")
        self.assertNotEqual(s1, s2)

    def test_merge_different_component_raises(self):
        s1 = Storage(tensors=["A"], component="MainMem")
        s2 = Storage(tensors=["B"], component="Buffer")
        with self.assertRaises(ValueError):
            s1._merge(s2)

    def test_render_node_shape(self):
        s = Storage(tensors=["A"], component="MainMem")
        self.assertEqual(s._render_node_shape(), "cylinder")

    def test_persistent_default_false(self):
        s = Storage(tensors=["A"], component="MainMem")
        self.assertFalse(s.persistent)


# ============================================================================
# Toll (mapping)
# ============================================================================


class TestMappingToll(unittest.TestCase):
    """Tests for Toll mapping node."""

    def test_basic_creation(self):
        t = Toll(tensors=["A"], component="Network")
        self.assertEqual(t.component, "Network")

    def test_str_representation(self):
        t = Toll(tensors=["A"], component="Network")
        result = str(t)
        self.assertIn("Network", result)
        self.assertIn("passes", result)

    def test_render_node_shape(self):
        t = Toll(tensors=["A"], component="Network")
        self.assertEqual(t._render_node_shape(), "rarrow")


# ============================================================================
# Compute (mapping)
# ============================================================================


class TestMappingCompute(unittest.TestCase):
    """Tests for Compute mapping node."""

    def test_basic_creation(self):
        c = Compute(einsum="Matmul", component="MAC")
        self.assertEqual(c.einsum, "Matmul")
        self.assertEqual(c.component, "MAC")

    def test_str_representation(self):
        c = Compute(einsum="Matmul", component="MAC")
        result = str(c)
        self.assertIn("MAC", result)
        self.assertIn("Matmul", result)

    def test_compact_str(self):
        c = Compute(einsum="Matmul", component="MAC")
        result = c.compact_str()
        self.assertIn("MAC", result)
        self.assertIn("Matmul", result)

    def test_render_node_shape(self):
        c = Compute(einsum="Matmul", component="MAC")
        self.assertEqual(c._render_node_shape(), "ellipse")


# ============================================================================
# Reservation
# ============================================================================


class TestReservation(unittest.TestCase):
    """Tests for Reservation mapping node."""

    def test_basic_creation(self):
        r = Reservation(purposes=["A"], resource="MainMem")
        self.assertEqual(r.purposes, ["A"])
        self.assertEqual(r.resource, "MainMem")

    def test_purpose_single(self):
        r = Reservation(purposes=["A"], resource="MainMem")
        self.assertEqual(r.purpose, "A")

    def test_purpose_multiple_raises(self):
        r = Reservation(purposes=["A", "B"], resource="MainMem")
        with self.assertRaises(ValueError):
            _ = r.purpose

    def test_str_representation(self):
        r = Reservation(purposes=["A"], resource="MainMem")
        result = str(r)
        self.assertIn("MainMem", result)
        self.assertIn("A", result)

    def test_compact_str(self):
        r = Reservation(purposes=["A"], resource="MainMem")
        result = r.compact_str()
        self.assertIn("A", result)
        self.assertIn("MainMem", result)

    def test_equality(self):
        r1 = Reservation(purposes=["A"], resource="MainMem")
        r2 = Reservation(purposes=["A"], resource="MainMem")
        self.assertEqual(r1, r2)

    def test_inequality(self):
        r1 = Reservation(purposes=["A"], resource="MainMem")
        r2 = Reservation(purposes=["B"], resource="MainMem")
        self.assertNotEqual(r1, r2)

    def test_persistent_default(self):
        r = Reservation(purposes=["A"], resource="MainMem")
        self.assertFalse(r.persistent)

    def test_render_node_shape(self):
        r = Reservation(purposes=["A"], resource="MainMem")
        self.assertEqual(r._render_node_shape(), "component")


# ============================================================================
# Nested
# ============================================================================


class TestNested(unittest.TestCase):
    """Tests for Nested mapping node."""

    def test_basic_creation(self):
        n = Nested(nodes=[])
        self.assertEqual(len(n.nodes), 0)

    def test_with_children(self):
        n = Nested(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Storage(tensors=["A"], component="MainMem"),
                Compute(einsum="Matmul", component="MAC"),
            ]
        )
        self.assertEqual(len(n.nodes), 3)

    def test_get_nodes_of_type(self):
        n = Nested(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Temporal(rank_variable="n", tile_shape=2),
                Storage(tensors=["A"], component="MainMem"),
                Compute(einsum="Matmul", component="MAC"),
            ]
        )
        temporals = n.get_nodes_of_type(Temporal)
        self.assertEqual(len(temporals), 2)

    def test_flatten(self):
        n = Nested(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Storage(tensors=["A"], component="MainMem"),
            ]
        )
        flat = n._flatten()
        self.assertGreaterEqual(len(flat), 3)  # Nested + 2 children

    def test_compact_str(self):
        n = Nested(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Storage(tensors=["A"], component="MainMem"),
            ]
        )
        result = n.compact_str()
        self.assertIn("m", result)


# ============================================================================
# Split, Sequential, Pipeline
# ============================================================================


class TestSplit(unittest.TestCase):
    """Tests for Split node."""

    def test_basic_creation(self):
        s = Split(nodes=[])
        self.assertEqual(len(s.nodes), 0)

    def test_str_representation(self):
        s = Split(nodes=[])
        self.assertEqual(str(s), "Split")

    def test_render_node_shape(self):
        s = Split(nodes=[])
        self.assertEqual(s._render_node_shape(), "hexagon")


class TestSequential(unittest.TestCase):
    """Tests for Sequential split node."""

    def test_basic_creation(self):
        s = Sequential(nodes=[])
        self.assertEqual(len(s.nodes), 0)

    def test_with_nested_children(self):
        s = Sequential(
            nodes=[
                Nested(nodes=[Compute(einsum="E1", component="MAC")]),
                Nested(nodes=[Compute(einsum="E2", component="MAC")]),
            ]
        )
        self.assertEqual(len(s.nodes), 2)


class TestPipeline(unittest.TestCase):
    """Tests for Pipeline split node."""

    def test_basic_creation(self):
        p = Pipeline(nodes=[])
        self.assertEqual(len(p.nodes), 0)


# ============================================================================
# Mapping (top-level)
# ============================================================================


class TestMapping(unittest.TestCase):
    """Tests for the top-level Mapping class."""

    def test_empty_mapping(self):
        m = Mapping()
        self.assertEqual(len(m.nodes), 0)

    def test_mapping_with_nodes(self):
        m = Mapping(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Storage(tensors=["A"], component="MainMem"),
                Compute(einsum="Matmul", component="MAC"),
            ]
        )
        self.assertEqual(len(m.nodes), 3)

    def test_loops_property(self):
        m = Mapping(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Temporal(rank_variable="n", tile_shape=2),
                Compute(einsum="Matmul", component="MAC"),
            ]
        )
        self.assertEqual(len(m.loops), 2)

    def test_remove_reservations(self):
        m = Mapping(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Reservation(purposes=["A"], resource="MainMem"),
                Compute(einsum="Matmul", component="MAC"),
            ]
        )
        m.remove_reservations()
        self.assertEqual(len(m.nodes), 2)
        self.assertFalse(any(isinstance(n, Reservation) for n in m.nodes))

    def test_split_loop_with_multiple_rank_variables(self):
        m = Mapping(
            nodes=[
                Temporal(rank_variable={"m", "n"}, tile_shape=4),
                Compute(einsum="E", component="MAC"),
            ]
        )
        m.split_loop_with_multiple_rank_variables(stride_and_halo={"m": (1, 0)})
        temporals = [n for n in m.nodes if isinstance(n, Temporal)]
        self.assertEqual(len(temporals), 2)

    def test_split_tensor_holders_with_multiple_tensors(self):
        m = Mapping(
            nodes=[
                Storage(tensors=["A", "B"], component="MainMem"),
                Compute(einsum="E", component="MAC"),
            ]
        )
        m.split_tensor_holders_with_multiple_tensors()
        storages = [n for n in m.nodes if isinstance(n, Storage)]
        self.assertEqual(len(storages), 2)

    def test_get_nodes_of_type(self):
        m = Mapping(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Storage(tensors=["A"], component="MainMem"),
                Compute(einsum="Matmul", component="MAC"),
            ]
        )
        storages = m.get_nodes_of_type(Storage)
        self.assertEqual(len(storages), 1)

    def test_mapping_node_hash(self):
        """Mapping nodes use id-based hashing."""
        n1 = Temporal(rank_variable="m", tile_shape=4)
        n2 = Temporal(rank_variable="m", tile_shape=4)
        # Different objects should have different hashes (identity-based)
        self.assertNotEqual(hash(n1), hash(n2))

    def test_mapping_node_identity_equality(self):
        """Mapping nodes use identity-based equality."""
        n1 = Temporal(rank_variable="m", tile_shape=4)
        n2 = Temporal(rank_variable="m", tile_shape=4)
        # But Loop.__eq__ compares by value, so they're equal by value
        self.assertEqual(n1, n2)


# ============================================================================
# MappingNodeWithChildren utilities
# ============================================================================


class TestMappingNodeWithChildrenUtils(unittest.TestCase):
    """Tests for utility methods on MappingNodeWithChildren."""

    def test_clear_nodes_of_type(self):
        n = Nested(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                Reservation(purposes=["A"], resource="MainMem"),
                Storage(tensors=["A"], component="MainMem"),
                Compute(einsum="E", component="MAC"),
            ]
        )
        n.clear_nodes_of_type(Reservation)
        self.assertFalse(any(isinstance(x, Reservation) for x in n.nodes))
        self.assertEqual(len(n.nodes), 3)

    def test_clear_nodes_specific(self):
        r = Reservation(purposes=["A"], resource="MainMem")
        n = Nested(
            nodes=[
                Temporal(rank_variable="m", tile_shape=4),
                r,
                Compute(einsum="E", component="MAC"),
            ]
        )
        n.clear_nodes(r)
        self.assertEqual(len(n.nodes), 2)


if __name__ == "__main__":
    unittest.main()
