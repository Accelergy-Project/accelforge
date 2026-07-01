# Transfer models: `HypercubeMulticastModel` and how to write a new one

This directory holds **distributed / network cost models** for spatial reuse analysis. This
document explains how `HypercubeMulticastModel` works and gives a step-by-step recipe (plus a fully
worked example) for adding your own model.

---

## 1. Overview

A **transfer model** estimates the on-chip data-movement cost of a mapping: given where data
*lives* (occupancy) and where it is *needed* (fills), it computes how many network *hops* the
delivery costs and which transfers are fulfilled peer-to-peer vs. read from a parent.

All transfer models implement one abstract interface, `TransferModel`. The concrete
implementations:

| Model | File | Shape |
|-------|------|-------|
| `SimpleLinkTransferModel` | [`../spatial.py`](../spatial.py) | Neighbor-to-neighbor mesh; no constructor state |
| `HypercubeMulticastModel` | [`distributed_buffers.py`](distributed_buffers.py) | Distance-aware worst-case multicast; takes a `dist_fn` |
| `FullyConnectedMulticastModel` | [`distributed_buffers.py`](distributed_buffers.py) | Fully-connected fabric; 1 hop per fabric crossing (see §6) |
| `XYRoutingMulticastModel` | [`distributed_buffers.py`](distributed_buffers.py) | XY / dimension-order routing on a 2-D mesh; X-then-Y multicast tree; per-link `EdgePressure` (see §7–§8) |
| `StarMulticastModel` | [`distributed_buffers.py`](distributed_buffers.py) | Star / central-switch — the spokes realization of a fully-connected fabric; per-spoke `EdgePressure` (see §8). Parameter-free constant regime only, like XY |

> **There is no registry or factory.** Models are constructed directly
> (`HypercubeMulticastModel(dist_fn)`) and applied via `.apply(...)`. The only current usage is the
> test suite — [`tests/not_working/distribuffers/test_multicast.py`](../../../../../../tests/not_working/distribuffers/test_multicast.py).
> The `not_working/` location signals these models are work-in-progress.

---

## 2. The `TransferModel` contract

Defined in [`../spatial.py`](../spatial.py). You implement exactly one abstract method:

```python
class TransferModel(ABC):
    @abstractmethod
    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        ...
```

### Inputs

- **`buff: MappingNode`** — the buffer being analyzed. (`HypercubeMulticastModel` ignores it and
  relies on its `dist_fn` instead; `SimpleLinkTransferModel` uses it to find spatial dims.)
- **`fills: Fill`** — a *tagged* `isl.Map` `{ [spacetime] -> [data] }` describing what each element
  needs from its parent over time.
- **`occs: Occupancy`** — a *tagged* `isl.Map` `{ [spacetime] -> [data] }` describing what each
  element holds over time.

`Fill` and `Occupancy` are `TaggedMap`s (see
[`../mapping_to_isl/types.py`](../mapping_to_isl/types.py)): the raw relation is `.map_`, and
`.tags` is a list of `Tag`s labeling each **input** dimension. Both constructors assert
`len(tags) == map_.dim(isl.dim_type.in_)`. The relevant tags are:

- `TemporalTag()` — that dimension spreads over time.
- `SpatialTag(spatial_dim, buffer)` — that dimension spreads over space, in `buffer`.

### Output: `TransferInfo`

A frozen dataclass (in `../spatial.py`) you must fully populate:

| Field | Type | Meaning |
|-------|------|---------|
| `fulfilled_fill` | `Transfers` | Fills satisfied by peer-to-peer transfers (a tagged map) — the fills *covered* by a matched multicast source. |
| `unfulfilled_fill` | `Fill` | Fills *not* satisfied — no source held the datum, so it must come from higher in the hierarchy. |
| `parent_reads` | `Reads` | Fills satisfied by parent-to-child reads. |
| `hops` | `isl.PwQPolynomial` | The transfer cost metric across spacetime. |
| `link_transfer` | `bool` | Metadata flag — whether this used link transfers. |
| `edge_pressure` | `Optional[EdgePressure]` | Per-directed-edge load backing `hops` (see §8), for models that define a per-link topology. Defaults to `None`. |

`fulfilled_fill` and `unfulfilled_fill` **partition the fills exactly**: with
`covered = _covered_fills(mcs)` (the `{ dst -> data }` pairs `identify_mesh_casts` matched to a
source), `fulfilled = fills ∩ covered` and `unfulfilled = fills − covered`. A destination requesting
a datum that *no* source holds simply never appears in the multicast networks, so it lands in
`unfulfilled_fill` — it is never silently treated as fulfilled.

`edge_pressure` is populated by the models with an explicit per-link topology
(`XYRoutingMulticastModel`: mesh links; `StarMulticastModel`: spokes) inside `apply()`, from the
same `identify_mesh_casts` result as `hops`. `HypercubeMulticastModel` and
`FullyConnectedMulticastModel` leave it `None` — a convex bounding box has no notion of individual
links, and a contention-free full mesh (one dedicated link per pair) has no shared-link pressure to
report.

`Transfers` and `Reads` are thin `TaggedMap` subclasses; construct them as
`Transfers(tags, map_)` / `Reads(tags, map_)`.

---

## 3. How `HypercubeMulticastModel` works

> **Worst-case multicast.** It assumes every multicast broadcasts to the *convex hypercube* that
> encloses all of its sources and destinations — an upper bound on the real cost.

### Constructor

```python
HypercubeMulticastModel(dist_fn: isl.Map)
```

`dist_fn` is a distance function `{ [dst -> src] -> [hops] }` — note the orientation: its domain is
a **`[dst -> src]` pair** (destination first), because `identify_mesh_casts` composes it as
`fills_to_matches.apply_range(dist_fn)` onto `{ ... -> [dst -> src] }` maps. Two further caller
contracts: the tuple names in `dist_fn`'s domain must match the spacetime tuple names of
`fills`/`occs` (ISL raises on a name mismatch when `dist_fn` is applied), and its range tuple must
be named `hops` (the fully-connected and star models filter on `{ hops[h] : h >= 1 }` to tell
self-deliveries apart from fabric crossings). Two assumptions are baked in:

1. **Orthogonal dimensions / Manhattan distance** — each unit move along a dimension costs 1 hop,
   and dimensions are orthogonal in the metric space.
2. **Translational invariance** — distance depends only on the displacement: if
   `|src − dst| = |src' − dst'|` then `dist_fn(dst, src) = dist_fn(dst', src')`.

These come from `calculate_extents_per_dim`, which the model relies on.

### `apply(...)` data flow

```
occs.map_  { [spacetime] -> [data] }   fills.map_  { [spacetime] -> [data] }
         \                                   /
          \                                 /
        identify_mesh_casts(occs, fills, dist_fn)
                          |
                          v
            mcs : { [data] -> [dst -> src] }    (each datum's nearest-source multicast network)
                          |
            _cost_mesh_cast_hypercube(mcs)
                          |
                          v
            hops : isl.PwQPolynomial  (total upper-bound hop count)
```

**Step A — `identify_mesh_casts(src_occupancy, dst_fill, dist_fn)`**
([`distributed_buffers.py`](distributed_buffers.py)).

For every datum, it pairs the destinations that request it with the *nearest* source that holds it.
Conceptually:

1. Reverse occupancy to `{ [data] -> [src] }` (which elements hold each datum).
2. Match each fill `{ [dst] -> [data] }` against the sources holding that datum →
   `{ [dst -> data] -> [dst -> src] }`.
3. Apply `dist_fn` and take `lexmin` over distance to keep only the closest source per
   `(dst, data)`.
4. Regroup as `{ [data] -> [dst -> src] }` and `lexmin` again to *devolve to a single source* when
   several are equidistant.

The result is the set of **multicast networks** (`mcns`): per datum, the destinations grouped with
their chosen source.

**Step B — extents per dimension: `calculate_extents_per_dim(mcns)`**
([`distributed_buffers.py`](distributed_buffers.py)).

For each multicast network it unions the sources with the destinations, then for each NoC dimension
projects away the others and takes `dim_max − dim_min`. That difference is the **extent** (the side
length of the bounding box) along that dimension, returned as one `isl.PwAff` per dimension.

**Step C — hypercube cost: `_cost_mesh_cast_hypercube(mcns)`**
([`distributed_buffers.py`](distributed_buffers.py)).

It folds the per-dimension extents into a single cost, then `.sum()`s over all networks.

> ⚠️ **Known discrepancy — this is the bug currently being tracked.**
> The docstring/comment says the cost is
> `(∏_i extent_i) − 1` (the number of *interior* points of the bounding box minus one — the
> standard hypercube broadcast cost). **But the code actually computes**
>
> ```
> cost = (∏_i (extent_i + 1)) − 1
> ```
>
> because the loop multiplies by `dim_extent.add(one)` (`extent_i + 1`), not `extent_i`. Since
> `extent_i = max − min` is already *one less* than the number of points along a dimension, adding
> 1 back double-counts the span. This is consistent with the observed **~3× overestimate of hops in
> all-to-all** topologies noted in recent work. Treat the formula in the code as *not yet correct*;
> a new template should decide deliberately whether it wants `extent_i` or `extent_i + 1`.

### Return value

```python
# { dst -> data } fills actually covered by a matched source (D2).
covered: isl.Map = _covered_fills(mcs)

return TransferInfo(
    fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
    parent_reads=Reads(occs.tags, mcs),                          # the multicast map
    unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
    hops=result,                                                 # the PwQPolynomial cost
    link_transfer=True,
    # No per-link decomposition defined for the hypercube abstraction.
    edge_pressure=None,
)
```

Note the partition idiom: `_covered_fills(mcs)` reshapes the multicast networks
`{ [data] -> [dst -> src] }` into `{ dst -> data }` (via
`range_reverse().uncurry().domain_factor_domain().reverse()` — "forget which source served it, keep
dst and data"), which lines up with `fills.map_` so `intersect`/`subtract` split the fills into the
covered and uncovered halves. Earlier revisions returned the *entire* fill map as `fulfilled_fill`
and an always-empty `unfulfilled_fill` (`fills.map_.subtract(fills.map_)`); that silently
mis-reported fills whose datum no source holds, and was fixed in D2 — do not copy that idiom.

---

## 4. Contrast: `SimpleLinkTransferModel`

For comparison (the *other* valid shape — no constructor state), `SimpleLinkTransferModel`
([`../spatial.py`](../spatial.py)):

- Takes **no `dist_fn`**; constructed as `SimpleLinkTransferModel()`.
- Asserts `fills.tags == occs.tags`.
- Builds a **neighbor-to-neighbor mesh** via `make_mesh_connectivity` (only 1 or 2 spatial dims
  supported — raises otherwise).
- Data reachable from a neighbor is `fulfilled_fill`; the rest is `unfulfilled_fill`.
- `hops` is **1 per neighbor-filled element** (`PwQPolynomial.one_on_domain(...)`), not a
  distance-weighted sum.
- Has an early-out: if there is no temporal dimension or no spatial dimension, nothing moves, so it
  returns an empty/zero `TransferInfo`.

So these two models bracket the design space: **stateful + distance-aware** (hypercube) vs.
**stateless + fixed-topology** (simple link).

---

## 5. How to create a new template

1. **Pick a home.** Add your class under `reuse/isl/`. Put distributed / NoC / mesh models in this
   `distributed/` directory next to `distributed_buffers.py`.
2. **Subclass `TransferModel`** and decide your constructor state — a `dist_fn`, topology
   parameters, bandwidth, or nothing (like `SimpleLinkTransferModel`).
3. **Implement `apply(self, buff, fills, occs) -> TransferInfo`.** If your model is mesh/multicast
   shaped, reuse the existing kernels:
   - `identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)` to get `{ [data] -> [dst -> src] }`.
   - `_covered_fills(mcs)` to partition the fills into fulfilled/unfulfilled.
   - `calculate_extents_per_dim(mcns)` if you want per-dimension bounding-box extents.
   - `_edge_pressure_from_links(...)` / `_const_pwq(...)` / `_eval_const(...)` if you report a
     per-link `EdgePressure` (see §8) and derive `hops` from it.
   Otherwise write your own cost kernel over the ISL maps. Call `identify_mesh_casts` **once** per
   `apply` and derive everything (`hops`, `edge_pressure`, the fill partition) from that single
   result, so the outputs can never disagree.
4. **Honor the invariants.** Assert `fills.tags == occs.tags` if your model needs aligned tags.
   Build `hops` as an `isl.PwQPolynomial` over the correct domain, and partition the fills with
   `_covered_fills` (`fulfilled = fills ∩ covered`, `unfulfilled = fills − covered`) rather than
   declaring everything fulfilled. Decide deliberately whether your topology defines an
   `edge_pressure` (per-link decomposition) or leaves it `None`.
5. **Add a test** mirroring
   [`tests/not_working/distribuffers/test_multicast.py`](../../../../../../tests/not_working/distribuffers/test_multicast.py):
   a YAML-driven gamut of `(dims, fill, occ, dist_fn, expected_hops)` cases.

### Copy-paste skeleton

```python
import islpy as isl

from accelforge.frontend.mapping import MappingNode
from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import Fill, Occupancy
from accelforge.model._looptree.reuse.isl.spatial import (
    Reads,
    Transfers,
    TransferInfo,
    TransferModel,
)
# Reuse these if your model is mesh/multicast-shaped:
from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    identify_mesh_casts,
    calculate_extents_per_dim,
    _covered_fills,
)


class MyTransferModel(TransferModel):
    """One-line description of the topology/assumptions this model encodes."""

    def __init__(self, dist_fn: isl.Map):
        # TODO: store whatever state your cost kernel needs (or drop the arg entirely).
        self.dist_fn = dist_fn

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        # TODO (optional): assert fills.tags == occs.tags

        # 1. Group destinations with their nearest source per datum. Call this
        #    ONCE and derive every output from the same `mcs`.
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)

        # 2. TODO: compute your cost as an isl.PwQPolynomial.
        hops: isl.PwQPolynomial = self._cost(mcs)

        # 3. Partition the fills by whether a source was matched (D2).
        covered: isl.Map = _covered_fills(mcs)  # { dst -> data }

        # 4. Assemble the result.
        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
            hops=hops,
            link_transfer=True,
            # TODO: an EdgePressure if your topology defines per-link loads
            # (see §8); None if it has no per-link decomposition.
            edge_pressure=None,
        )

    def _cost(self, mcns: isl.Map) -> isl.PwQPolynomial:
        # TODO: your cost kernel.
        raise NotImplementedError
```

**Checklist**

- [ ] Subclasses `TransferModel`, implements `apply`.
- [ ] Constructor state matches what the cost kernel needs.
- [ ] `hops` is an `isl.PwQPolynomial` over the right domain.
- [ ] `fulfilled_fill` + `unfulfilled_fill` partition the fills via `_covered_fills(mcs)`
      (`fills ∩ covered` / `fills − covered`), not "everything fulfilled / empty unfulfilled".
- [ ] `identify_mesh_casts` is called once per `apply`; `hops`, `edge_pressure`, and the fill
      partition all derive from the same result.
- [ ] `edge_pressure` is a deliberate choice: an `EdgePressure` if the topology has per-link loads,
      `None` (with a comment saying why) if not.
- [ ] A YAML-driven gamut test exists.

---

## 6. Worked example: `FullyConnectedMulticastModel`

A real, tested model for a **fully-connected fabric** (e.g. an NVSwitch-style all-to-all). It lives
next to the hypercube model in [`distributed_buffers.py`](distributed_buffers.py).

On a fully-connected fabric every cross-node delivery costs **one hop regardless of distance**, and a
self-delivery (source node == destination node) costs **zero**. So the cost of a mapping is just the
number of deliveries that actually traverse the fabric:

```
cost = | { (data, dst, src) ∈ mcs : dist_fn(dst, src) ≥ 1 } |
```

`dist_fn` is used *only* to tell self-deliveries (0 hops) apart from fabric-crossing ones — its hop
*magnitude* never enters the cost. This sidesteps the hypercube extent overestimate (see §3) entirely.

```python
class FullyConnectedMulticastModel(TransferModel):
    """Multicast cost on a fully-connected fabric: 1 hop per fabric crossing."""

    def __init__(self, dist_fn: isl.Map):
        self.dist_fn = dist_fn

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        result: isl.PwQPolynomial = self._cost_fully_connected(mcs)
        # { dst -> data } fills actually covered by a matched source (D2).
        covered: isl.Map = _covered_fills(mcs)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
            hops=result,
            link_transfer=True,
            # No per-link decomposition defined for the full-mesh abstraction.
            edge_pressure=None,
        )

    def _cost_fully_connected(self, mcns: isl.Map) -> isl.PwQPolynomial:
        """Count the deliveries in ``mcns`` that traverse the fabric (dist >= 1)."""
        # [dst -> src] pairs that actually traverse the fabric (>= 1 hop).
        crossing_hops: isl.Set = isl.Set.read_from_str(
            isl.DEFAULT_CONTEXT, "{ hops[h] : h >= 1 }"
        )
        crossing_pairs: isl.Set = self.dist_fn.intersect_range(crossing_hops).domain()
        crossing: isl.Map = mcns.intersect_range(crossing_pairs)
        return crossing.wrap().card()
```

How the kernel works: `dist_fn.intersect_range({ hops[h] : h ≥ 1 }).domain()` is the set of
`[dst -> src]` pairs that cross the fabric; intersecting `mcns`' range with it keeps only crossing
deliveries; `.wrap().card()` counts the `(data, dst, src)` points as an `isl.PwQPolynomial`
(constant when there are no parameters). `intersect_range`, `domain`, `wrap`, and `card` are all
standard `islpy`/barvinok operations already used elsewhere in this subsystem.

**Verified numbers** (8-GPU one-hot encoding, from the test below):

| Case | `FullyConnectedMulticastModel` | `HypercubeMulticastModel` |
|------|-------------------------------|---------------------------|
| all-to-all (64 chunks, 8 self) | **56** | 168 |
| single unicast GPU0 → GPU3 | **1** | 3 |
| self chunk GPU5 → GPU5 | **0** | 0 |

The all-to-all column is the headline: **56 vs 168 is exactly the ~3× overestimate** the hypercube
model incurs on a fully-connected fabric (each unicast costed as a `(1+1)(1+1) − 1 = 3` hypercube
instead of a single crossing — see the discrepancy in §3).

This example is exercised by a real test:
[`tests/not_working/distribuffers/test_fully_connected.py`](../../../../../../tests/not_working/distribuffers/test_fully_connected.py)
with cases in
[`tests/not_working/distribuffers/fully_connected/test_cases.yaml`](../../../../../../tests/not_working/distribuffers/fully_connected/test_cases.yaml).
Run it (m4 on `PATH` per the islpy-barvinok setup):

```bash
PATH="$HOME/.local/bin:$PATH" .venv/bin/python -m pytest \
  tests/not_working/distribuffers/test_fully_connected.py -q
```

---

## 7. XY (dimension-order) routing: `XYRoutingMulticastModel`

`XYRoutingMulticastModel` ([`distributed_buffers.py`](distributed_buffers.py)) models **XY routing**
on a 2-D mesh: every packet travels **along X first, then Y**. A multicast from one source is
therefore a rigid tree:

1. an **X segment** along the source's row, reaching every column that holds a destination, then
2. an independent **Y segment** down each of those columns, *starting from the source's row*.

The cost of one tree (source `s = (xs, ys)`, destinations `D`) is:

```
x_extent({xs} ∪ {dst columns})
  + Σ over destination columns xd of  y_extent({ys} ∪ {dst y's in column xd})
```

Source selection is **per destination** (`identify_mesh_casts` pairs each destination with its
nearest source, devolving ties); destinations sharing a source form one tree, and the model sums
over all trees and all data.

### Where it sits: `extent_DOR` (floor) ≤ XY ≤ hypercube

Because each column's Y segment restarts from the source row instead of sharing a trunk, XY is an
**upper bound on free routing** and a **lower bound on the hypercube** (which reaches every node in
the bounding box). Note `extent_DOR_hops` in the test yamls is the **free-routing floor**, *not* the
XY cost — it assumes routing can move freely between dimensions. Worked example (verified by the
model): source `(1,0)` casting to `(0,2)` and `(2,2)`:

```
  y=2  D . D
  y=1  | . |
  y=0  +-S-+        S=(1,0); X covers cols 0..2 (2 links)
       x=0 1 2      then Y down col 0 and col 2 from row 0 (2 + 2 links)

  floor (extent_DOR) = 4   ≤   XY = 6   ≤   hypercube = 8
```

### Cost kernel

There is no standalone cost method: the cost *is* the per-edge decomposition, aggregated. `apply()`
decomposes every tree onto directed mesh links, wraps the per-link loads as an `EdgePressure`
(see §8), and derives `hops` from its total — one aggregation path, so the scalar and the per-edge
view can never disagree:

```python
def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
    # `identify_mesh_casts` is called exactly once; `hops`, `edge_pressure`,
    # and the fill partition all derive from this single `mcs` (D4/D5).
    mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
    links: list[isl.Map] = self._directed_mesh_links(mcs)   # see §8
    pressure: EdgePressure = _edge_pressure_from_links(links)
    # `hops` == the sum of per-edge loads (Σ_edges load == total link count),
    # wrapped as the constant PwQPolynomial `TransferInfo.hops` expects.
    hops: isl.PwQPolynomial = _const_pwq(pressure.total())
    covered: isl.Map = _covered_fills(mcs)                   # { dst -> data }

    return TransferInfo(
        fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
        parent_reads=Reads(occs.tags, mcs),
        unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
        hops=hops,
        link_transfer=True,
        edge_pressure=pressure,
    )
```

`_directed_mesh_links` builds the X segments explicitly along each source row (out to every
destination column, split rightward/leftward at the source column `xs`) and the Y segments down
each destination column from the source row (split upward/downward at `ys`), keeping every link's
identity — see §8 for the edge naming.

Notes / limitations:
- **2-D node tuples only** (no temporal dims); `apply` raises a `ValueError` ("XYRoutingMulticastModel
  requires a 2-D node tuple (X then Y)...") for any other arity — N-D dimension-order routing is a
  TODO. The tuple *name* is generic: it is read off the maps at `apply` time (via
  `_mesh_node_tuple`), so `noc[x, y]`, `pe[x, y]`, etc. all work as long as `fills`, `occs`, and
  `dist_fn` agree on it (see the caller contract in §3).
- Each per-column/per-row link set is built explicitly and counted via `card()` rather than summing
  a min/max polynomial — the latter trips a barvinok `summate` assertion at scale (e.g. the 8×8
  case).
- Returns a **parameter-free constant** (the validated regime); parametric spacetimes are future work.

### Tested

[`tests/not_working/distribuffers/test_xy_routing.py`](../../../../../../tests/not_working/distribuffers/test_xy_routing.py)
with hand-derived cases in
[`tests/not_working/distribuffers/xy_routing/test_cases.yaml`](../../../../../../tests/not_working/distribuffers/xy_routing/test_cases.yaml)
(there is **no XY oracle in the repo**, so the expected values are hand-derived and each case carries
its geometry). Cases: unicast `4`, the `(1,0)` discriminator `6`, three-corner `6`, 1-D column `3`,
replicated-source `4`, and an 8×8 scale case `448`. Run:

```bash
PATH="$HOME/.local/bin:$PATH" .venv/bin/python -m pytest \
  tests/not_working/distribuffers/test_xy_routing.py -q
```

---

## 8. Per-edge memory pressure (link load): `EdgePressure`

`hops` collapses a whole routing to one scalar, which hides a real constraint: **each physical edge
has a finite bandwidth**, so the *busiest* link is what saturates first. `EdgePressure` keeps the
load broken out per directed physical edge — `{ edge -> number-of-trees-crossing-it }` — which is
exactly the quantity the production symbolic path calls `max_traffic` and feeds into the `Network`
latency formula `max(max_hops·latency, max_link_traffic / throughput)`
([`frontend/arch/components.py`](../../../../frontend/arch/components.py)). It is exposed as the
`TransferInfo.edge_pressure` field: `model.apply(buff, fills, occs).edge_pressure`, an
`Optional[EdgePressure]` populated by `XYRoutingMulticastModel` and `StarMulticastModel` inside
`apply()` (from the same `identify_mesh_casts` result as `hops`, so the two are always consistent)
and `None` for `HypercubeMulticastModel` / `FullyConnectedMulticastModel` (see §2). There is no
standalone `edge_pressure(fills, occs)` method.

### The primitive

After `identify_mesh_casts` fixes the `[dst -> src]` pairs, build, per directed edge type, a map
`M = { [data -> src] -> edge }` associating each multicast **tree** with every edge its route
crosses. Then (`_edge_pressure_from_links` does exactly this, unioning the per-type results)

```python
load = M.reverse().card()        # { edge -> #trees crossing it }
```

The key is the tree `(data, src)`, **not** the destination: within one tree a link is traversed once
regardless of how many leaves hang off it, so this counts *pressure* (distinct flows on a link), not
summed hops. `EdgePressure.total()` sums the load over every edge (this is what XY/Star `hops` is
built from, via `_const_pwq`); `bottleneck()` returns the max load (enumerating the finite,
parameter-free edge domain); `eval_edge(name, coords)` looks up one edge.

### XY routing → directed mesh edges

`XYRoutingMulticastModel._directed_mesh_links` decomposes each tree onto four directed link types:
`xedge_r`/`xedge_l` along the source row (split at the source column `xs`) and `yedge_u`/`yedge_d`
down each destination column (split at the source row `ys`). This **is** the §7 cost kernel — every
link is built explicitly with its identity kept (not via `calculate_extents_per_dim`, which only
yields a scalar length and discards *which* links are used), and the scalar `hops` is just
`EdgePressure.total()` over these maps.

Decisive cross-check (no oracle needed): **`Σ_edges load == total XY hops`**. The per-edge loads must
sum back to the already-trusted A–F totals (4/6/6/3/4/448), so a wrong decomposition fails. Worked F
geometry (8×8, datum `(d0,d1)` held at node `(d0,d1)`, requested by all of column `x=d0`): every
datum of column `c` floods all 7 vertical links of column `c`, so `yedge_u[c,t]` load = `t+1` and
`yedge_d[c,t]` = `7−t`; the busiest directed link is `7` (the physical link total is `8`). Cases A–E
are single trees, so every edge load is `1`.

### Star / fully-connected → spokes

`StarMulticastModel` **is** the spokes realization of the fully-connected fabric (NVSwitch-style: each
node has one link to a central switch). A delivery routes `src → switch → dst`, so each datum loads
its source's egress spoke once (multicast fans out *at the switch*) and each destination's ingress
spoke once; self-deliveries (0 hops) load nothing. `_spoke_loads` returns `spoke_in[n]` (ingress) and
`spoke_out[n]` (egress) loads. For an N-way all-to-all every node receives `N−1` and sources `1`, so
the **ingress spokes are hottest at `N−1`** — this is where bandwidth actually bites, whereas
`FullyConnectedMulticastModel`'s contention-free full-mesh view has no hotspot. The two are tied by
**`Σ_nodes ingress == FullyConnected crossing count`** (every crossing delivery is one node's
ingress; e.g. 8-GPU all-to-all: `56`). The star scalar `hops` = `Σ egress + Σ ingress` (injections +
deliveries).

### Tested

[`tests/not_working/distribuffers/test_edge_pressure.py`](../../../../../../tests/not_working/distribuffers/test_edge_pressure.py)
— the XY `Σ load == hops` invariant over A–F (with the pressure taken from
`apply(...).edge_pressure`), F's bottleneck/edge loads (`7`, `yedge_u[0,6]=7`, `yedge_d[0,1]=6`),
single-tree unit bottlenecks, and the star spoke loads / `Σ ingress == FC count` invariant for 4-
and 8-GPU all-to-all (star `hops` = injections + deliveries: `16`/`64`, ingress bottleneck
`3`/`7`). Run:

```bash
PATH="$HOME/.local/bin:$PATH" .venv/bin/python -m pytest \
  tests/not_working/distribuffers/test_edge_pressure.py -q
```

---

## 9. References

- Interface + `SimpleLinkTransferModel` + `TransferInfo`: [`../spatial.py`](../spatial.py)
- `HypercubeMulticastModel`, `identify_mesh_casts`, `calculate_extents_per_dim`:
  [`distributed_buffers.py`](distributed_buffers.py)
- Tagged-map / tag types: [`../mapping_to_isl/types.py`](../mapping_to_isl/types.py)
- Example test harness:
  [`tests/not_working/distribuffers/test_multicast.py`](../../../../../../tests/not_working/distribuffers/test_multicast.py)
