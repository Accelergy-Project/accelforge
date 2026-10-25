"""
Tests for the DonutMulticastModel (K rings of R GPUs, ring links inside a ring,
complete bipartite lanes between consecutive rings), a composition of the ring
and star models.

- ``TestDonutMulticastModel.test_gamut``: total links crossed (``hops``) for the
  hand-derived cases in ``donut/test_cases.yaml`` (K = 9, R = 8).
- ``TestDonutLinkLoads``: closed-form per-link loads of the study's MoE layer
  step (unicast, multicast, broadcast) at K = 9, R = 8 and K = 3, R = 4, with
  ``hops == Σ load``.
- ``TestDonutComposition``: the ring and lane loads equal what the Ring and Star
  primitives give on the equivalent 1-D flows.
- ``TestDonutContract``: the unsupported cases, constructor guards, and
  ``donut_dist_fn``.

See accelforge/model/_looptree/reuse/isl/distributed/README.md.
"""

import time
import unittest
from pathlib import Path

import islpy as isl

from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import (
    Fill,
    Occupancy,
    SpatialTag,
)
from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    DonutMulticastModel,
    RingMulticastModel,
    StarMulticastModel,
)
from accelforge.model._looptree.reuse.isl.distributed.edge_pressure import _eval_const

from .helpers import run_hops_gamut

_CTX = isl.DEFAULT_CONTEXT
_TAGS = [SpatialTag(0, 0), SpatialTag(1, 0)]


def Donut98(dist_fn: isl.Map) -> DonutMulticastModel:
    """`run_hops_gamut` builds models as `model_cls(dist_fn)`; the yaml cases are K = 9, R = 8."""
    assert dist_fn.is_equal(DonutMulticastModel.donut_dist_fn(8, 9))
    return DonutMulticastModel(dist_fn, 8, 9)


def _casts(K: int, R: int, casts):
    """
    (fill, occ) with one datum data[i] per cast ``((l, g), [(l2, g2), ...])``,
    held at the source and wanted at every destination, over 2-D nodes noc[l, g].
    """
    del K, R  # the nodes are listed explicitly; kept for a uniform signature
    occ_c = " or ".join(f"(c = {i} and l = {s[0]} and g = {s[1]})" for i, (s, _) in enumerate(casts))
    fill_c = " or ".join(
        f"(c = {i} and ({' or '.join(f'(l = {a} and g = {b})' for a, b in dsts)}))"
        for i, (_, dsts) in enumerate(casts))
    occ = Occupancy(_TAGS, isl.Map.read_from_str(_CTX, f"{{ noc[l, g] -> data[c] : {occ_c} }}"))
    fill = Fill(_TAGS, isl.Map.read_from_str(_CTX, f"{{ noc[l, g] -> data[c] : {fill_c} }}"))
    return fill, occ


def _flows(K: int, R: int, pairs):
    """(fill, occ) with one datum per (src, dst) pair ``((l, g), (l2, g2))``."""
    return _casts(K, R, [(s, [t]) for s, t in pairs])


def _moe_step(K: int, R: int, m: int, q: int, variant: str):
    """
    The study's MoE layer step in the complement structured encoding.

    A class is identified by its kk = R - 1 - m missing offsets x1 < ... < xkk;
    "d in class" is 1 <= d <= R - 1 and d != x_j for every j. Dispatch (kind 0)
    sends a token from (l, g) to (l, g + d mod R) for d in its class; combine
    (kind 1) sends each expert GPU's output to (l + 1 mod K, g).
    ``unicast`` is one dispatch datum per (src, dst) pair, ``broadcast`` the
    fan-out form with every offset, ``multicast`` uses ``m``.
    """
    if variant == "unicast":
        kk, m = 0, 1
    elif variant == "broadcast":
        kk = 0
    else:
        kk = R - 1 - m
    xs = [f"x{j}" for j in range(1, kk + 1)]
    X = "".join(f"{x}, " for x in xs)
    b0 = [f"0 <= l < {K}", f"0 <= g < {R}", f"0 <= i < {q}"]
    if xs:
        b0 += ["1 <= x1"] + [f"{a} < {b}" for a, b in zip(xs, xs[1:])] + [f"{xs[-1]} <= {R - 1}"]
    in_class = [f"1 <= d <= {R - 1}"] + [f"d != {x}" for x in xs]
    B0 = " and ".join(b0)
    B = " and ".join(b0 + in_class)
    if variant == "unicast":
        disp_occ = f"{{ data[0, l, g, {X}i, d] -> noc[l, g] : {B} }}"
        disp_fill = f"{{ data[0, l, g, {X}i, d] -> noc[l, (g + d) mod {R}] : {B} }}"
    else:
        disp_occ = f"{{ data[0, l, g, {X}i, 0] -> noc[l, g] : {B0} }}"
        disp_fill = (f"{{ data[0, l, g, {X}i, 0] -> [tok[l, g] -> off[d]] : {B} }}")
    comb_occ = f"{{ data[1, l, g, {X}i, d] -> noc[l, (g + d) mod {R}] : {B} }}"
    comb_fill = f"{{ data[1, l, g, {X}i, d] -> noc[(l + 1) mod {K}, g] : {B} }}"

    fill_m = isl.Map.read_from_str(_CTX, disp_fill)
    if variant != "unicast":
        fill_m = fill_m.apply_range(isl.Map.read_from_str(
            _CTX, f"{{ [tok[l, g] -> off[d]] -> noc[l, (g + d) mod {R}] }}"))
    fill_m = fill_m.union(isl.Map.read_from_str(_CTX, comb_fill))
    occ_m = isl.Map.read_from_str(_CTX, disp_occ).union(isl.Map.read_from_str(_CTX, comb_occ))
    return Fill(_TAGS, fill_m.reverse()), Occupancy(_TAGS, occ_m.reverse())


def _donut(K: int, R: int) -> DonutMulticastModel:
    return DonutMulticastModel(DonutMulticastModel.donut_dist_fn(R, K), R, K)


def _check(K, R, variant, m, cw, ccw, fwd, bwd, total):
    """Closed-form loads of one MoE step: every ring link, every lane, the total."""
    fill, occ = _moe_step(K, R, m, 1, variant)
    info = _donut(K, R).apply(0, fill, occ)
    ep = info.edge_pressure
    for l in range(K):
        for e in range(R):
            assert ep.eval_edge("ring_cw", [l, e]) == cw, (variant, "cw", l, e)
            assert ep.eval_edge("ring_ccw", [l, e]) == ccw, (variant, "ccw", l, e)
        for g in range(R):
            for h in range(R):
                assert ep.eval_edge("bip_fwd", [l, g, h]) == (fwd if h != g else 0), (variant, "fwd", l, g, h)
                assert ep.eval_edge("bip_bwd", [l, g, h]) == bwd, (variant, "bwd", l, g, h)
    assert ep.bottleneck() == max(cw, ccw, fwd, bwd)
    assert _eval_const(info.hops) == ep.total() == total, (variant, ep.total())
    assert info.unfulfilled_fill.map_.is_empty()


class TestDonutMulticastModel(unittest.TestCase):
    """Total links crossed for the hand-derived yaml cases."""

    TEST_CASES_FILE: Path = Path(__file__).parent / "donut" / "test_cases.yaml"

    def test_gamut(self):
        run_hops_gamut(Donut98, self.TEST_CASES_FILE, "donut_hops")


class TestDonutLinkLoads(unittest.TestCase):
    """Per-link loads of the MoE layer step against closed forms."""

    def test_unicast(self):
        """Every ring is a personalised all-to-all (cw 10, ccw 6); each combine datum takes one lane."""
        _check(9, 8, "unicast", 1, 10, 6, 1, 0, 1656)

    def test_multicast(self):
        _check(9, 8, "multicast", 5, 77, 56, 15, 0, 17136)

    def test_broadcast(self):
        _check(9, 8, "broadcast", 7, 4, 3, 1, 0, 1008)

    def test_small_donut_multicast(self):
        """K = 3, R = 4, m = 2: 3 * 4 * (5 + 2) ring crossings plus 3 * 4 * 3 * 2 lane crossings."""
        _check(3, 4, "multicast", 2, 5, 2, 2, 0, 156)

    def test_apply_time(self):
        """The K = 9, R = 8 multicast step finishes within the guard (the probe took ~2 s)."""
        fill, occ = _moe_step(9, 8, 5, 1, "multicast")
        start = time.perf_counter()
        _donut(9, 8).apply(0, fill, occ)
        elapsed = time.perf_counter() - start
        print(f"donut K=9 R=8 multicast apply: {elapsed:.2f} s")
        assert elapsed < 20.0, elapsed


class TestDonutComposition(unittest.TestCase):
    """The composite equals the primitives on the equivalent flows."""

    def test_ring_equals_ring_model(self):
        """All-to-all plus a 3-leaf multicast on ring 3 loads ring 3 exactly like the Ring model."""
        R = 8
        casts_1d = [(s, [t]) for s in range(R) for t in range(R) if s != t]
        casts_1d.append((2, [3, 4, 5]))
        occ_c = " or ".join(f"(c = {i} and g = {s})" for i, (s, _) in enumerate(casts_1d))
        fill_c = " or ".join(
            f"(c = {i} and ({' or '.join(f'g = {t}' for t in ts)}))" for i, (_, ts) in enumerate(casts_1d))
        tags1 = [SpatialTag(0, 0)]
        occ1 = Occupancy(tags1, isl.Map.read_from_str(_CTX, f"{{ noc[g] -> data[c] : {occ_c} }}"))
        fill1 = Fill(tags1, isl.Map.read_from_str(_CTX, f"{{ noc[g] -> data[c] : {fill_c} }}"))
        ref = RingMulticastModel(RingMulticastModel.ring_dist_fn(R), R).apply(0, fill1, occ1).edge_pressure

        casts_2d = [((3, s), [(3, t) for t in ts]) for s, ts in casts_1d]
        fill2, occ2 = _casts(9, R, casts_2d)
        got = _donut(9, R).apply(0, fill2, occ2).edge_pressure
        for e in range(R):
            assert got.eval_edge("ring_cw", [3, e]) == ref.eval_edge("ring_cw", [e]), e
            assert got.eval_edge("ring_ccw", [3, e]) == ref.eval_edge("ring_ccw", [e]), e
        assert got.total() == ref.total()
        for l in range(9):
            if l == 3:
                continue
            for e in range(R):
                assert got.eval_edge("ring_cw", [l, e]) == 0
                assert got.eval_edge("ring_ccw", [l, e]) == 0

    def test_lanes_equal_star_spokes(self):
        """One hub (4, 5) with lane traffic loads bip_fwd[4, 5, h] like the star's ingress spoke h."""
        R = 8
        casts = [((4, 5), [(5, 0), (5, 2), (5, 6)]), ((4, 5), [(5, 1)]), ((4, 5), [(5, 2)])]
        fill, occ = _casts(9, R, casts)
        got = _donut(9, R).apply(0, fill, occ).edge_pressure

        # Equivalent star: leaves 0..R-1 (ring 5) plus the hub as extra node R.
        occ_c = " or ".join(f"(c = {i} and g = {R})" for i in range(len(casts)))
        fill_c = " or ".join(
            f"(c = {i} and ({' or '.join(f'g = {b}' for _, b in dsts)}))" for i, (_, dsts) in enumerate(casts))
        tags1 = [SpatialTag(0, 0)]
        occ1 = Occupancy(tags1, isl.Map.read_from_str(_CTX, f"{{ noc[g] -> data[c] : {occ_c} }}"))
        fill1 = Fill(tags1, isl.Map.read_from_str(_CTX, f"{{ noc[g] -> data[c] : {fill_c} }}"))
        star_dist = isl.Map.read_from_str(
            _CTX, f"{{ [noc[a] -> noc[b]] -> hops[1] : 0 <= a <= {R} and 0 <= b <= {R} and a != b; "
                  f"[noc[a] -> noc[b]] -> hops[0] : 0 <= a <= {R} and a = b }}")
        star = StarMulticastModel(star_dist).apply(0, fill1, occ1).edge_pressure
        expected = {0: 1, 1: 1, 2: 2, 6: 1}
        for h in range(R):
            assert got.eval_edge("bip_fwd", [4, 5, h]) == star.eval_edge("spoke_in", [h]) == expected.get(h, 0), h
        assert got.total() == 5  # five lane crossings; the ring and the hub egress add nothing

    def test_backward_lanes(self):
        """(4, 5) -> {(3, 0), (3, 7)} loads only bip_bwd[4, 5, 0] and bip_bwd[4, 5, 7]."""
        fill, occ = _casts(9, 8, [((4, 5), [(3, 0), (3, 7)])])
        ep = _donut(9, 8).apply(0, fill, occ).edge_pressure
        for h in range(8):
            assert ep.eval_edge("bip_bwd", [4, 5, h]) == (1 if h in (0, 7) else 0)
            assert ep.eval_edge("bip_fwd", [4, 5, h]) == 0
        assert ep.total() == 2

    def test_wrap_forward_lane(self):
        """(8, 1) -> (0, 2) loads bip_fwd[8, 1, 2]."""
        fill, occ = _flows(9, 8, [((8, 1), (0, 2))])
        ep = _donut(9, 8).apply(0, fill, occ).edge_pressure
        assert ep.eval_edge("bip_fwd", [8, 1, 2]) == 1
        assert ep.total() == 1


class TestDonutContract(unittest.TestCase):
    def test_rejects_non_adjacent_rings(self):
        fill, occ = _flows(9, 8, [((0, 0), (2, 0))])
        with self.assertRaisesRegex(ValueError, "non-adjacent rings"):
            _donut(9, 8).apply(0, fill, occ)

    def test_rejects_small_k_and_n(self):
        with self.assertRaises(ValueError):
            DonutMulticastModel(DonutMulticastModel.donut_dist_fn(4, 2), 4, 2)
        with self.assertRaises(ValueError):
            DonutMulticastModel(DonutMulticastModel.donut_dist_fn(1, 3), 1, 3)

    def test_rejects_1d_nodes(self):
        tags = [SpatialTag(0, 0)]
        occ = Occupancy(tags, isl.Map.read_from_str(_CTX, "{ noc[g] -> data[c] : c = 0 and g = 0 }"))
        fill = Fill(tags, isl.Map.read_from_str(_CTX, "{ noc[g] -> data[c] : c = 0 and g = 1 }"))
        dist = isl.Map.read_from_str(_CTX, "{ [noc[a] -> noc[b]] -> hops[1] }")
        with self.assertRaisesRegex(ValueError, "2-D node tuple"):
            DonutMulticastModel(dist, 8, 9).apply(0, fill, occ)

    def test_dist_fn_is_total_single_valued_and_closed_form(self):
        for n, k in ((4, 3), (8, 9)):
            dist = DonutMulticastModel.donut_dist_fn(n, k)
            assert dist.is_single_valued()
            assert int(str(dist.domain().count_val())) == (k * n) ** 2
        n, k = 4, 3
        dist = DonutMulticastModel.donut_dist_fn(n, k)
        for ld in range(k):
            for gd in range(n):
                for ls in range(k):
                    for gs in range(n):
                        if ld == ls:
                            d = (gd - gs) % n
                            want = min(d, n - d)
                        else:
                            d = (ld - ls) % k
                            want = min(d, k - d)
                        pair = isl.Set.read_from_str(_CTX, f"{{ [noc[{ld}, {gd}] -> noc[{ls}, {gs}]] }}")
                        got = dist.intersect_domain(pair).range()
                        assert got.is_equal(isl.Set.read_from_str(_CTX, f"{{ hops[{want}] }}")), (ld, gd, ls, gs)


if __name__ == "__main__":
    unittest.main()
