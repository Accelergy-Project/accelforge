"""
Tests for the TorusMulticastModel (3-D torus, dimension-order routing x, y, z,
each axis a shortest-arc ring), a composition of the XY mesh's dimension-order
tree and the ring model's link decomposition.

- ``TestTorusHandDerived``: one hand-derived tree, the antipodal tie and a
  self-delivery.
- ``TestTorusBruteForce``: every link load equals a plain-Python dimension-order
  enumerator on seeded random multicasts (4 x 4 x 4 and 3 x 4 x 5), with
  ``hops == Σ load``.
- ``TestTorusComposition``: deliveries along a single axis load exactly what
  ``RingMulticastModel`` gives on that axis.
- ``TestTorusContract``: constructor guards, the 3-D tuple check and
  ``torus_dist_fn``.

See accelforge/model/_looptree/reuse/isl/distributed/README.md.
"""

import random
import unittest
from collections import Counter

import islpy as isl

from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import (
    Fill,
    Occupancy,
    SpatialTag,
)
from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    RingMulticastModel,
    TorusMulticastModel,
)
from accelforge.model._looptree.reuse.isl.distributed.edge_pressure import _eval_const

_CTX = isl.DEFAULT_CONTEXT
_TAGS = [SpatialTag(0, 0), SpatialTag(1, 0), SpatialTag(2, 0)]
_AXES = ("x", "y", "z")


def _casts(dims, casts):
    """
    (fill, occ) with one datum data[i] per cast ``(src, [dst, ...])``, held at
    the source and wanted at every destination, over 3-D nodes noc[x, y, z].
    """
    dom = " and ".join(f"0 <= {v} < {n}" for v, n in zip("xyz", dims))
    at = lambda p: " and ".join(f"{v} = {c}" for v, c in zip("xyz", p))  # noqa: E731
    occ = " or ".join(f"(i = {i} and {at(s)})" for i, (s, _) in enumerate(casts))
    fill = " or ".join(
        f"(i = {i} and {at(d)})" for i, (_, ds) in enumerate(casts) for d in ds
    )
    occ_m = isl.Map.read_from_str(_CTX, f"{{ noc[x, y, z] -> data[i] : {dom} and ({occ}) }}")
    fill_m = isl.Map.read_from_str(_CTX, f"{{ noc[x, y, z] -> data[i] : {dom} and ({fill}) }}")
    return Fill(_TAGS, fill_m), Occupancy(_TAGS, occ_m)


def _model(dims) -> TorusMulticastModel:
    return TorusMulticastModel(TorusMulticastModel.torus_dist_fn(dims), dims)


def _arc_links(n, s, dsts):
    """Directed ring links `(i, +1 or -1)` an n-ring multicast from `s` crosses (antipode clockwise)."""
    links = set()
    for t in dsts:
        d = (t - s) % n
        if d == 0:
            continue
        step, hops = (1, d) if 2 * d <= n else (-1, n - d)
        links |= {((s + step * h) % n, step) for h in range(hops)}
    return links


def _brute_loads(dims, casts):
    """{(edge name, (x, y, z)): trees crossing it}, by plain dimension-order enumeration."""
    loads = Counter()
    for s, dsts in casts:
        for k in range(3):
            # Lines of phase k: earlier axes at the destination, later ones at the source.
            lines = {}
            for d in dsts:
                line = tuple(d[j] if j < k else s[j] for j in range(3) if j != k)
                lines.setdefault(line, []).append(d[k])
            for line, coords in lines.items():
                for e, step in _arc_links(dims[k], s[k], coords):
                    c = list(line)
                    c.insert(k, e)
                    loads[f"{_AXES[k]}_{'cw' if step == 1 else 'ccw'}", tuple(c)] += 1
    return loads


def _all_edges(dims):
    return [
        (name, (x, y, z))
        for name in TorusMulticastModel.EDGE_TYPES
        for x in range(dims[0])
        for y in range(dims[1])
        for z in range(dims[2])
    ]


class TestTorusHandDerived(unittest.TestCase):
    def test_tree(self):
        # (0,0,0) -> {(2,1,0), (2,3,1)} on 4x4x4: x cw 0->1->2 in row (y=0, z=0);
        # column x=2 at z=0: y cw 0->1 and y ccw 0->3; pillar (2,3): z cw 0->1.
        fill, occ = _casts((4, 4, 4), [((0, 0, 0), [(2, 1, 0), (2, 3, 1)])])
        ep = _model((4, 4, 4)).apply(0, fill, occ).edge_pressure
        want = {("x_cw", (0, 0, 0)), ("x_cw", (1, 0, 0)), ("y_cw", (2, 0, 0)),
                ("y_ccw", (2, 0, 0)), ("z_cw", (2, 3, 0))}
        for name, c in _all_edges((4, 4, 4)):
            self.assertEqual(int(ep.eval_edge(name, list(c))), int((name, c) in want), (name, c))
        self.assertEqual(int(ep.total()), 5)

    def test_antipode_goes_clockwise(self):
        fill, occ = _casts((4, 4, 4), [((1, 1, 1), [(1, 3, 1)])])
        ep = _model((4, 4, 4)).apply(0, fill, occ).edge_pressure
        self.assertEqual(int(ep.eval_edge("y_cw", [1, 1, 1])), 1)
        self.assertEqual(int(ep.eval_edge("y_cw", [1, 2, 1])), 1)
        self.assertEqual(int(ep.total()), 2)

    def test_self_delivery_loads_nothing(self):
        fill, occ = _casts((4, 4, 4), [((1, 2, 3), [(1, 2, 3)])])
        info = _model((4, 4, 4)).apply(0, fill, occ)
        self.assertEqual(int(info.edge_pressure.total()), 0)
        self.assertTrue(info.unfulfilled_fill.map_.is_empty())


class TestTorusBruteForce(unittest.TestCase):
    def _check(self, dims, n_casts, seed):
        rng = random.Random(seed)
        nodes = [(x, y, z) for x in range(dims[0]) for y in range(dims[1]) for z in range(dims[2])]
        casts = [(rng.choice(nodes), rng.sample(nodes, rng.randint(1, 9))) for _ in range(n_casts)]
        fill, occ = _casts(dims, casts)
        info = _model(dims).apply(0, fill, occ)
        ep = info.edge_pressure
        want = _brute_loads(dims, casts)
        for name, c in _all_edges(dims):
            self.assertEqual(int(ep.eval_edge(name, list(c))), want.get((name, c), 0), (name, c))
        self.assertEqual(int(ep.total()), sum(want.values()))
        self.assertEqual(_eval_const(info.hops), sum(want.values()))
        self.assertEqual(int(ep.bottleneck()), max(want.values()))

    def test_4x4x4(self):
        self._check((4, 4, 4), 24, seed=1)

    def test_3x4x5(self):
        self._check((3, 4, 5), 24, seed=2)


class TestTorusComposition(unittest.TestCase):
    def test_single_axis_equals_ring(self):
        dims = (4, 5, 3)
        for k in range(3):
            n = dims[k]
            fixed = [1, 2, 0]

            def pt(v):
                p = list(fixed)
                p[k] = v
                return tuple(p)

            casts1 = [(1, [3, 4 % n, 0]), (0, [2]), (2, [1, 0])]
            fill, occ = _casts(dims, [(pt(s), [pt(t % n) for t in ts]) for s, ts in casts1])
            ep = _model(dims).apply(0, fill, occ).edge_pressure
            occ_r = " or ".join(f"(i = {i} and g = {s})" for i, (s, _) in enumerate(casts1))
            fill_r = " or ".join(f"(i = {i} and g = {t % n})" for i, (_, ts) in enumerate(casts1) for t in ts)
            tags = [SpatialTag(0, 0)]
            ring = RingMulticastModel(RingMulticastModel.ring_dist_fn(n), n)
            rp = ring.apply(
                0,
                Fill(tags, isl.Map.read_from_str(_CTX, f"{{ noc[g] -> data[i] : 0 <= g < {n} and ({fill_r}) }}")),
                Occupancy(tags, isl.Map.read_from_str(_CTX, f"{{ noc[g] -> data[i] : 0 <= g < {n} and ({occ_r}) }}")),
            ).edge_pressure
            for e in range(n):
                for d, rname in (("cw", "ring_cw"), ("ccw", "ring_ccw")):
                    self.assertEqual(
                        int(ep.eval_edge(f"{_AXES[k]}_{d}", list(pt(e)))),
                        int(rp.eval_edge(rname, [e])),
                        (k, d, e),
                    )
            self.assertEqual(int(ep.total()), int(rp.total()))


class TestTorusContract(unittest.TestCase):
    def test_dims_guard(self):
        dist = TorusMulticastModel.torus_dist_fn((4, 4, 4))
        with self.assertRaises(ValueError):
            TorusMulticastModel(dist, (4, 4))
        with self.assertRaises(ValueError):
            TorusMulticastModel(dist, (4, 1, 4))

    def test_requires_3d_tuple(self):
        tags = [SpatialTag(0, 0), SpatialTag(1, 0)]
        occ = isl.Map.read_from_str(_CTX, "{ noc[x, y] -> data[0] : x = 0 and y = 0 }")
        fill = isl.Map.read_from_str(_CTX, "{ noc[x, y] -> data[0] : x = 1 and y = 0 }")
        dist = isl.Map.read_from_str(
            _CTX, "{ [noc[a, b] -> noc[c, d]] -> hops[h] : 0 <= a, b, c, d < 4 and h = 0 }"
        )
        with self.assertRaises(ValueError):
            TorusMulticastModel(dist, (4, 4, 4)).apply(0, Fill(tags, fill), Occupancy(tags, occ))

    def test_dist_fn(self):
        dist = TorusMulticastModel.torus_dist_fn((4, 4, 5))
        for d, s, h in [((0, 0, 0), (0, 0, 0), 0), ((3, 0, 0), (0, 0, 0), 1), ((2, 2, 0), (0, 0, 0), 4),
                        ((1, 3, 4), (0, 0, 0), 3), ((0, 0, 3), (0, 0, 0), 2)]:
            pt = isl.Map.read_from_str(
                _CTX, "{ [noc[%d, %d, %d] -> noc[%d, %d, %d]] -> hops[h] }" % (*d, *s)
            )
            got = dist.intersect_domain(pt.domain()).range()
            self.assertTrue(got.is_equal(isl.Set.read_from_str(_CTX, f"{{ hops[{h}] }}")), (d, s, got))
        self.assertTrue(dist.is_single_valued())


if __name__ == "__main__":
    unittest.main()
