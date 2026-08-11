"""
Regression tests for contract fixes to ``distributed_buffers.py``'s multicast
models (see the ``_mesh_node_tuple``/``_covered_fills``/
``identify_mesh_casts`` docstrings in ``mesh_casts.py`` for the full
rationale). Each class below pins down one previously-broken-or-undocumented
contract so a future change cannot silently regress it:

- ``TestTupleNameGenericity``: a spacetime/node tuple named anything
  other than the literal ``'noc'`` (e.g. ``pe[x, y]``) used to crash deep
  inside an opaque ISL assertion in ``XYRoutingMulticastModel``'s and
  ``StarMulticastModel``'s helpers, which hardcoded ``'noc'`` into every map
  string they built. Both now read the tuple name off the caller's own maps.
- ``TestXYDimensionalityGuard``: a non-2-D node tuple through
  ``XYRoutingMulticastModel`` now raises a clear ``ValueError`` instead of the
  same kind of opaque ISL assertion.
- ``TestUnfulfilledPartition``: ``fulfilled_fill``/``unfulfilled_fill``
  are a true partition of ``fills`` by whether ``identify_mesh_casts`` found a
  matched source, for all four models -- previously ``unfulfilled_fill`` was
  unconditionally empty regardless of whether a fill actually had a source.
- ``TestAsymmetricDistFnOrientation``: ``identify_mesh_casts``'s
  ``dist_fn`` is documented to be applied in the
  ``{ [dst -> src] -> [hops] }`` orientation; this pins down that an
  asymmetric ``dist_fn`` (direction-dependent cost) actually selects the
  source that is *cheapest to reach from the destination*, not some other
  pairing that a swapped orientation would silently produce.

See accelforge/model/_looptree/reuse/isl/distributed/README.md for the models'
background.
"""

import unittest

import islpy as isl

from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import (
    Fill,
    Occupancy,
    SpatialTag,
)
from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    HypercubeMulticastModel,
    FullyConnectedMulticastModel,
    XYRoutingMulticastModel,
    StarMulticastModel,
)
from accelforge.model._looptree.reuse.isl.distributed.edge_pressure import _eval_const

_CTX = isl.DEFAULT_CONTEXT


def _manhattan_2d(name: str) -> isl.Map:
    """
    Build a 2-D Manhattan ``{ [dst -> src] -> [hops] }`` distance map over a
    node tuple named ``name`` (e.g. ``"noc"`` or ``"pe"``), identical in shape
    to the ``2d_manhattan`` fixture in ``xy_routing/test_cases.yaml`` but with
    the tuple name parametrized so it can be reused for the tuple-name
    genericity regression tests below.
    """
    return isl.Map.read_from_str(
        _CTX,
        (
            "{{ [{n}[xd, yd] -> {n}[xs, ys]] -> hops[(xd - xs) + (yd - ys)] :"
            "     xd >= xs and yd >= ys;"
            "   [{n}[xd, yd] -> {n}[xs, ys]] -> hops[-(xd - xs) + -(yd - ys)] :"
            "     xd < xs and yd < ys;"
            "   [{n}[xd, yd] -> {n}[xs, ys]] -> hops[-(xd - xs) + (yd - ys)] :"
            "     xd < xs and yd >= ys;"
            "   [{n}[xd, yd] -> {n}[xs, ys]] -> hops[(xd - xs) + -(yd - ys)] :"
            "     xd >= xs and yd < ys }}"
        ).format(n=name),
    )


def _manhattan_1d(name: str) -> isl.Map:
    """
    Build a 1-D Manhattan ``{ [dst -> src] -> [hops] }`` distance map over a
    node tuple named ``name``, for the line-of-nodes fixtures below.
    """
    return isl.Map.read_from_str(
        _CTX,
        (
            "{{ [{n}[nd] -> {n}[ns]] -> hops[nd - ns] : nd >= ns;"
            "   [{n}[nd] -> {n}[ns]] -> hops[ns - nd] : nd <  ns }}"
        ).format(n=name),
    )


def _all_to_all(name: str, n: int) -> tuple[Fill, Occupancy, isl.Map]:
    """
    Build ``(fill, occ, dist_fn)`` for an ``n``-node all-to-all on a 1-D line
    of nodes tagged under the tuple name ``name``.

    Node ``g`` starts holding only ``data[g]`` and requests every other node's
    datum, exactly the ``_all_to_all`` fixture in ``test_edge_pressure.py``
    but with the tuple name parametrized for the genericity checks.
    """
    tags = [SpatialTag(0, 0)]
    occ = Occupancy(
        tags,
        isl.Map.read_from_str(
            _CTX, f"{{ {name}[g] -> data[d] : 0 <= g < {n} and d = g }}"
        ),
    )
    fill = Fill(
        tags,
        isl.Map.read_from_str(
            _CTX,
            f"{{ {name}[g] -> data[d] : 0 <= g < {n} and 0 <= d < {n} and d != g }}",
        ),
    )
    return fill, occ, _manhattan_1d(name)


class TestTupleNameGenericity(unittest.TestCase):
    """
    A node tuple named anything other than ``'noc'`` must produce the same
    numbers as the ``'noc'``-named equivalent, for every model whose helpers
    used to hardcode the literal ``'noc'`` into ISL map strings.
    """

    def test_xy_non_noc_tuple_name_matches_noc(self):
        """
        XY routing case B (source (1,0) -> (0,2),(2,2), 6 hops -- see
        ``xy_routing/test_cases.yaml``) run under both ``noc[x, y]`` and
        ``pe[x, y]`` must produce the identical hop count. Previously, `'pe'`
        crashed inside ``_directed_mesh_links``'s hardcoded ``'noc[...]'`` map
        strings with a raw ISL tuple-name-mismatch assertion.
        """
        tags = [SpatialTag(0, 0), SpatialTag(1, 0)]
        for name in ("noc", "pe"):
            with self.subTest(name=name):
                occ = Occupancy(
                    tags,
                    isl.Map.read_from_str(
                        _CTX, f"{{ {name}[x, y] -> data[c] : c = 0 and x = 1 and y = 0 }}"
                    ),
                )
                fill = Fill(
                    tags,
                    isl.Map.read_from_str(
                        _CTX,
                        f"{{ {name}[x, y] -> data[c] : c = 0 and"
                        f"    ((x = 0 and y = 2) or (x = 2 and y = 2)) }}",
                    ),
                )
                model = XYRoutingMulticastModel(_manhattan_2d(name))
                hops = _eval_const(model.apply(0, fill, occ).hops)
                self.assertEqual(hops, 6)

    def test_star_and_fully_connected_non_noc_tuple_name_match_noc(self):
        """
        The 8-node all-to-all oracle (FullyConnected 56, Star 64 -- see
        ``TestStarSpokePressure`` in ``test_edge_pressure.py``) run under both
        ``noc[g]`` and ``pe[g]`` must produce identical numbers. Previously,
        ``StarMulticastModel._spoke_loads`` hardcoded ``'noc'`` the same way
        ``_directed_mesh_links`` did (``FullyConnectedMulticastModel`` never
        hardcoded a tuple name, so it is included here only as a same-input
        cross-check, not because it was previously broken).
        """
        for name in ("noc", "pe"):
            with self.subTest(name=name):
                fill, occ, dist_fn = _all_to_all(name, 8)
                fc_hops = _eval_const(
                    FullyConnectedMulticastModel(dist_fn).apply(0, fill, occ).hops
                )
                star_hops = _eval_const(
                    StarMulticastModel(dist_fn).apply(0, fill, occ).hops
                )
                self.assertEqual(fc_hops, 56)
                self.assertEqual(star_hops, 64)


class TestXYDimensionalityGuard(unittest.TestCase):
    """
    ``XYRoutingMulticastModel`` only supports exactly 2-D node tuples, and
    must say so with a clear ``ValueError`` rather than an opaque ISL abort.
    """

    def test_three_dimensional_node_tuple_raises_value_error(self):
        """
        A 3-D node tuple (``noc[x, y, z]``) must raise ``ValueError`` from
        ``apply`` (via ``_directed_mesh_links``), naming the offending tuple
        and its dimensionality, instead of failing deep inside an
        ``isl.Map.read_from_str``/``apply_range`` call the way it did before
        ``_mesh_node_tuple``'s dimensionality check was added. The
        `dist_fn` here is a trivial constant (correctness of routing over a
        3-D mesh is out of scope -- this test only exercises the guard).
        """
        tags = [SpatialTag(0, 0), SpatialTag(1, 0), SpatialTag(2, 0)]
        occ = Occupancy(
            tags,
            isl.Map.read_from_str(
                _CTX, "{ noc[x, y, z] -> data[c] : c = 0 and x = 0 and y = 0 and z = 0 }"
            ),
        )
        fill = Fill(
            tags,
            isl.Map.read_from_str(
                _CTX, "{ noc[x, y, z] -> data[c] : c = 0 and x = 1 and y = 0 and z = 0 }"
            ),
        )
        dist_fn = isl.Map.read_from_str(
            _CTX, "{ [noc[xd, yd, zd] -> noc[xs, ys, zs]] -> hops[0] }"
        )
        model = XYRoutingMulticastModel(dist_fn)
        with self.assertRaisesRegex(
            ValueError, r"2-D node tuple.*'noc'.*3 dimensions"
        ):
            model.apply(0, fill, occ)


class TestUnfulfilledPartition(unittest.TestCase):
    """
    ``fulfilled_fill``/``unfulfilled_fill`` must be a true partition of
    ``fills`` by whether ``identify_mesh_casts`` matched a source, for every
    model. Shared geometry: node 0 holds only ``data[0]``; node 1 requests
    both ``data[0]`` (has a source -- fulfilled) and ``data[1]`` (no source
    anywhere -- unfulfilled). Previously, every model reported
    ``unfulfilled_fill`` as unconditionally empty (``fills - fills``) and
    ``fulfilled_fill`` as the entire (uncovered) fill map.
    """

    @staticmethod
    def _geometry(name: str, two_d: bool):
        """
        Build ``(tags, occ, fill, dist_fn, expected_fulfilled,
        expected_unfulfilled)`` for the shared node0/node1 geometry, in either
        a 1-D node tuple (for models with no dimensionality requirement) or a
        2-D one (required by ``XYRoutingMulticastModel``).
        """
        if two_d:
            tags = [SpatialTag(0, 0), SpatialTag(1, 0)]
            occ = Occupancy(
                tags,
                isl.Map.read_from_str(
                    _CTX, f"{{ {name}[x, y] -> data[c] : x = 0 and y = 0 and c = 0 }}"
                ),
            )
            fill = Fill(
                tags,
                isl.Map.read_from_str(
                    _CTX,
                    f"{{ {name}[x, y] -> data[c] : x = 1 and y = 0 and"
                    f"    (c = 0 or c = 1) }}",
                ),
            )
            dist_fn = _manhattan_2d(name)
            expected_fulfilled = isl.Map.read_from_str(
                _CTX, f"{{ {name}[x, y] -> data[c] : x = 1 and y = 0 and c = 0 }}"
            )
            expected_unfulfilled = isl.Map.read_from_str(
                _CTX, f"{{ {name}[x, y] -> data[c] : x = 1 and y = 0 and c = 1 }}"
            )
        else:
            tags = [SpatialTag(0, 0)]
            occ = Occupancy(
                tags, isl.Map.read_from_str(_CTX, f"{{ {name}[n] -> data[c] : n = 0 and c = 0 }}")
            )
            fill = Fill(
                tags,
                isl.Map.read_from_str(
                    _CTX, f"{{ {name}[n] -> data[c] : n = 1 and (c = 0 or c = 1) }}"
                ),
            )
            dist_fn = _manhattan_1d(name)
            expected_fulfilled = isl.Map.read_from_str(
                _CTX, f"{{ {name}[n] -> data[c] : n = 1 and c = 0 }}"
            )
            expected_unfulfilled = isl.Map.read_from_str(
                _CTX, f"{{ {name}[n] -> data[c] : n = 1 and c = 1 }}"
            )
        return tags, occ, fill, dist_fn, expected_fulfilled, expected_unfulfilled

    def test_partition_across_models(self):
        """
        For each model, ``unfulfilled_fill.map_`` must be exactly
        ``{ dst -> data[1] }`` (node 1's unmatched request), and
        ``fulfilled_fill.map_`` must be exactly ``{ dst -> data[0] }`` (the
        covered rest). ``hops`` is checked too, as a sanity cross-check that
        only the matched (data[0]) delivery is costed -- 1 for the models that
        count a single crossing, 2 for ``StarMulticastModel`` (its `hops` is
        injection + delivery, i.e. egress + ingress, for the one crossing
        delivery).
        """
        # (model class, needs-2-D node tuple, expected `hops` for the one
        # matched delivery).
        cases = [
            (HypercubeMulticastModel, False, 1),
            (FullyConnectedMulticastModel, False, 1),
            (StarMulticastModel, False, 2),
            (XYRoutingMulticastModel, True, 1),
        ]
        for model_cls, two_d, expected_hops in cases:
            with self.subTest(model=model_cls.__name__):
                _, occ, fill, dist_fn, expected_fulfilled, expected_unfulfilled = (
                    self._geometry("noc", two_d)
                )
                info = model_cls(dist_fn).apply(0, fill, occ)
                self.assertTrue(
                    info.unfulfilled_fill.map_.is_equal(expected_unfulfilled),
                    f"unfulfilled_fill was {info.unfulfilled_fill.map_}, "
                    f"expected {expected_unfulfilled}",
                )
                self.assertTrue(
                    info.fulfilled_fill.map_.is_equal(expected_fulfilled),
                    f"fulfilled_fill was {info.fulfilled_fill.map_}, "
                    f"expected {expected_fulfilled}",
                )
                self.assertEqual(_eval_const(info.hops), expected_hops)


class TestAsymmetricDistFnOrientation(unittest.TestCase):
    """
    ``identify_mesh_casts`` applies ``dist_fn`` in the
    ``{ [dst -> src] -> [hops] }`` orientation (the docstring previously said
    the opposite, though the code itself was never wrong -- the fix was
    documentation-only). This pins down the now-correctly-documented
    behavior with a ``dist_fn`` that is *directionally* asymmetric (cost
    depends on which side is source vs. destination, not just on distance),
    so a caller who mis-orients their own ``dist_fn`` -- or a future change
    that swaps `identify_mesh_casts`'s composition order -- would select the
    wrong source and fail this test.
    """

    def test_nearest_source_selected_by_directional_cost_not_distance(self):
        """
        Three nodes on a line: 0, 1, 2. Nodes 0 and 2 both hold ``data[0]``;
        node 1 requests it. Both candidate sources are Euclidean-distance 1
        away, so a distance-based tiebreak could not distinguish them -- the
        ``dist_fn`` instead makes the *direction* the deciding factor: a
        source below the destination (``src < dst``) costs 1, a source above
        (``src > dst``) costs 5. The nearest (min-cost) source is therefore
        node 0, not node 2. `identify_mesh_casts`'s `parent_reads` records
        exactly which source served the delivery, so this is checked directly
        against the expected ``{ data[0] -> [noc[1] -> noc[0]] }`` mapping,
        not just inferred from a hop count.
        """
        tags = [SpatialTag(0, 0)]
        occ = Occupancy(
            tags, isl.Map.read_from_str(_CTX, "{ noc[n] -> data[c] : c = 0 and (n = 0 or n = 2) }")
        )
        fill = Fill(tags, isl.Map.read_from_str(_CTX, "{ noc[n] -> data[c] : c = 0 and n = 1 }"))
        # `{ [dst -> src] -> hops }`: cheap (1) if src is below dst, expensive
        # (5) if src is above -- direction, not magnitude, decides.
        dist_fn = isl.Map.read_from_str(
            _CTX,
            "{ [noc[nd] -> noc[ns]] -> hops[0] : nd = ns;"
            "  [noc[nd] -> noc[ns]] -> hops[1] : ns < nd;"
            "  [noc[nd] -> noc[ns]] -> hops[5] : ns > nd }",
        )
        model = FullyConnectedMulticastModel(dist_fn)
        info = model.apply(0, fill, occ)

        expected_match = isl.Map.read_from_str(_CTX, "{ data[0] -> [noc[1] -> noc[0]] }")
        self.assertTrue(
            info.parent_reads.map_.is_equal(expected_match),
            f"parent_reads was {info.parent_reads.map_}, expected {expected_match} "
            "(source node 0, the directionally-cheap side)",
        )
        # A single crossing delivery, regardless of the asymmetric magnitude.
        self.assertEqual(_eval_const(info.hops), 1)


if __name__ == "__main__":
    unittest.main()
