"""
Draw a fused mapping whose two Einsums bind the tensor they share two different ways.

A *binding* says how a physically-distributed memory spreads one tensor over its
physical instances: each spatial loop below the memory's Storage either partitions the
tensor along one of the tensor's ranks or replicates it across that spatial dimension.
A binding therefore belongs to one Einsum, not to the tensor -- and two Einsums fused
through a tensor are free to disagree about it, with an adapter shuffling the data
between the two layouts in between.

The picture of a fused mapping has to say which Einsum asked for which binding. Where
the two agree, the tensor they share is drawn once, above the split, as the one thing
both branches hold. Where they disagree, neither binding may speak for the other, so
each is drawn inside the branch of the Einsum that asked for it. This script renders
both, on the same mapping, so the difference is visible side by side.

Run it as::

    python experiments/binding_adapter_visualization.py

which writes one SVG per case plus an ``index.html`` that shows them together, and
prints what each case cost. The cases differ only in the ``binding`` of T1 in the two
branches of ``tests/input_files/mapping/fused_matmuls_to_binding_adapter.yaml``;
everything else about the mapping, and every action of the arch but a network hop, is
held fixed, so the energy column is a hop count and the difference between cases is the
adapter alone.
"""

import argparse
import html
import sys
from dataclasses import dataclass, field
from pathlib import Path

import accelforge as af
from accelforge.frontend.mapping.mapping import (
    Binding,
    Compute,
    MappingNodeWithChildren,
    Nested,
    _binding_str,
)
from accelforge.frontend.spec import Spec

REPO = Path(__file__).resolve().parent.parent
INPUT_FILES = REPO / "tests" / "input_files"
ARCH = INPUT_FILES / "binding_adapter.arch.yaml"
MAPPING = INPUT_FILES / "mapping" / "fused_matmuls_to_binding_adapter.yaml"

M = 8
KN = 8
BITS_PER_VALUE = 8
SHARED_TENSOR_BITS = M * KN * BITS_PER_VALUE
"""The fixture's shared tensor is an 8x8 tile of 8-bit values, so a full shuffle of it
moves this many bits -- the unit the adapter's energy comes out in."""


# =================================================================================================
# The cases
# =================================================================================================
@dataclass
class Case:
    """One pair of bindings of the shared tensor, and what to say about it."""

    name: str
    title: str
    story: str
    jinja: dict = field(default_factory=dict)
    """Overrides of the mapping fixture's binding variables. ``MATMUL0_X`` and
    ``MATMUL0_Y`` name the rank Matmul0 partitions T1 along across each spatial
    dimension of the SharedBuffer (``DUPLICATE`` to replicate instead); likewise
    ``MATMUL1_*`` for Matmul1. The fixture's own defaults are ``X:M, Y:N1`` for Matmul0
    and ``X:M, Y:*`` for Matmul1."""


CASES = [
    Case(
        name="agree",
        title="The two Einsums agree",
        story=(
            "Both Einsums leave T1 partitioned along N1 across Y. There is one layout "
            "of T1 in the SharedBuffer, so it is drawn once, above the split, and "
            "nothing has to move between the two Einsums."
        ),
        jinja={"MATMUL1_Y": "N1"},
    ),
    Case(
        name="gather",
        title="A partition gathered into a replica",
        story=(
            "Matmul0 leaves T1 split two ways along Y; Matmul1 wants every instance "
            "along Y to hold all of it. Each binding is drawn in its own branch, and "
            "the adapter gathers the halves in each of the two X columns."
        ),
        jinja={},
    ),
    Case(
        name="already-there",
        title="Bindings that differ for free",
        story=(
            "Matmul0 replicates T1 across Y and Matmul1 partitions it along N1. The "
            "two bindings are still different layouts, drawn one per branch -- but an "
            "instance holding a whole copy already holds any slice of it, so the "
            "adapter has nothing to move. Being drawn apart is about which Einsum "
            "asked for what, not about what the shuffle costs."
        ),
        jinja={"MATMUL0_Y": "DUPLICATE", "MATMUL1_Y": "N1"},
    ),
    Case(
        name="transpose",
        title="A partition moved between dimensions",
        story=(
            "Matmul0 partitions T1 along M across X and N1 across Y; Matmul1 wants the "
            "two the other way around. Every instance holds the right amount of T1 "
            "either way, but not the right part of it, so the adapter transposes it "
            "across the array."
        ),
        jinja={"MATMUL1_X": "N1", "MATMUL1_Y": "M"},
    ),
]


# =================================================================================================
# Evaluating one case
# =================================================================================================
def evaluate(**jinja):
    """The fixture's mapping under one pair of bindings, evaluated -- which runs the
    join, and the adapter with it."""
    settings = {
        "N_EINSUMS": 2,
        "M": M,
        "KN": KN,
        "M_TILE": M // 2,
        "N_TILE": KN // 2,
        "MAY_ADAPT": "All",
        "HOP_LATENCY": 1,
    }
    settings.update(jinja)
    spec = Spec.from_yaml(
        ARCH, af.examples.workloads.basic.matmuls_any_einsums, jinja_parse_data=settings
    )
    spec.mapping = Spec.from_yaml(
        MAPPING,
        jinja_parse_data={
            "M_TILE": M // 2,
            "N_TILE": KN // 2,
            **{k: v for k, v in jinja.items() if k.startswith("MATMUL")},
        },
    ).mapping
    return spec.evaluate_mapping().data


def bindings_by_einsum(mapping) -> dict[str | None, list[str]]:
    """
    The binding of every bound (memory, tensor) pair of a mapping, keyed by the Einsum
    whose branch it sits in -- or by ``None`` for the ones above the split, which belong
    to every Einsum at once. A ``None`` key is the picture saying the two Einsums
    agreed.
    """
    result: dict[str | None, list[str]] = {}

    def walk(node, einsum):
        for child in node.nodes:
            if isinstance(child, Binding):
                result.setdefault(einsum, []).extend(
                    _binding_str(loops) for _, _, loops in child.entries()
                )
            if isinstance(child, Nested):
                computes = child.get_nodes_of_type(Compute)
                walk(child, computes[0].einsum if computes else einsum)
            elif isinstance(child, MappingNodeWithChildren):
                walk(child, einsum)

    walk(mapping, None)
    return result


def einsum_hops(data) -> float:
    """The hops the two Einsums are charged for on their own, which the adapter's are
    deliberately not part of. Every other action of the fixture is free, so the total
    energy less these is what the adapter cost."""
    return sum(data[c].iloc[0] for c in data.columns if "action<SEP>NoC" in c)


def einsum_latency(data) -> float:
    """The latency the two Einsums are charged for on their own: each one's longest
    component, summed, which is what the joiner adds up."""
    total = 0.0
    for einsum in ("Matmul0", "Matmul1"):
        prefix = f"{einsum}<SEP>latency<SEP>"
        total += max(data[c].iloc[0] for c in data.columns if c.startswith(prefix))
    return total


@dataclass
class Result:
    case: Case
    svg: str
    bindings: dict
    adapter_energy: float
    adapter_latency: float

    @property
    def drawn_once_above_the_split(self) -> bool:
        return None in self.bindings

    def binding_of(self, einsum: str) -> str:
        """What the picture says this Einsum's binding of T1 is, wherever it is drawn."""
        if self.drawn_once_above_the_split:
            return self.bindings[None][0]
        return self.bindings[einsum][0]


def run(case: Case) -> Result:
    data = evaluate(**case.jinja)
    mapping = data["Total<SEP>mapping"].iloc[0]()
    return Result(
        case=case,
        svg=str(mapping.render()),
        bindings=bindings_by_einsum(mapping),
        adapter_energy=data["Total<SEP>energy"].iloc[0] - einsum_hops(data),
        adapter_latency=data["Total<SEP>latency"].iloc[0] - einsum_latency(data),
    )


# =================================================================================================
# Output
# =================================================================================================# =================================================================================================
# Output
# =================================================================================================
FONTS = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>\n"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>\n"
    "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;500&"
    "family=IBM+Plex+Sans:wght@500;600&"
    "family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap'>"
)

INDEX_STYLE = """
:root {
  --paper: #f3f6f8;
  --card: #ffffff;
  --ink: #141a21;
  --muted: #5a6673;
  --line: #d8dee6;
  --accent: #1f6f8b;
  --accent-soft: #e3eff4;
  --charged: #855511;
  --charged-soft: #f6ecd7;
  --free: #2d6a4d;
  --free-soft: #e0efe6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0e1317;
    --card: #161d24;
    --ink: #e3e9ef;
    --muted: #94a2b1;
    --line: #26313b;
    --accent: #7cc0d6;
    --accent-soft: #143440;
    --charged: #d8a55a;
    --charged-soft: #322718;
    --free: #7fc39d;
    --free-soft: #16301f;
  }
}
:root[data-theme="dark"] {
  --paper: #0e1317;
  --card: #161d24;
  --ink: #e3e9ef;
  --muted: #94a2b1;
  --line: #26313b;
  --accent: #7cc0d6;
  --accent-soft: #143440;
  --charged: #d8a55a;
  --charged-soft: #322718;
  --free: #7fc39d;
  --free-soft: #16301f;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 400 17px/1.62 "IBM Plex Serif", Georgia, serif;
}
.wrap {
  max-width: 62rem;
  margin: 0 auto;
  padding: clamp(2rem, 5vw, 4.5rem) clamp(1rem, 4vw, 2.5rem) 5rem;
  display: flex;
  flex-direction: column;
  gap: 3.5rem;
}
h1, h2, h3, .eyebrow, .label, th { font-family: "IBM Plex Sans", system-ui, sans-serif; }
h1 {
  font-size: clamp(2rem, 5vw, 2.9rem);
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.1;
  text-wrap: balance;
  margin: 0;
}
h2 {
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  text-wrap: balance;
  margin: 0;
}
h3 {
  font-size: 0.95rem;
  font-weight: 600;
  margin: 0;
}
p { margin: 0; max-width: 40em; }
.eyebrow, .label {
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--muted);
}
code, .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.88em; }
code {
  background: var(--accent-soft);
  color: var(--ink);
  padding: 0.08em 0.34em;
  border-radius: 2px;
}
a { color: var(--accent); }

/* ---- masthead ---------------------------------------------------------------- */
header { display: flex; flex-direction: column; gap: 1rem; }
header .rule { height: 3px; width: 4rem; background: var(--accent); }
.standfirst { font-size: 1.13rem; color: var(--ink); }
.deck { color: var(--muted); }

/* ---- glossary ---------------------------------------------------------------- */
.glossary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
}
.glossary > div {
  background: var(--card);
  padding: 1.1rem 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.glossary p { font-size: 0.94rem; color: var(--muted); }

/* ---- summary table ------------------------------------------------------------ */
.tablewrap { overflow-x: auto; border: 1px solid var(--line); background: var(--card); }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th, td {
  text-align: left;
  padding: 0.65rem 1rem;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
th {
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}
tbody tr:last-child td { border-bottom: none; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
td.mono, th.num { font-variant-numeric: tabular-nums; }

/* ---- one case ----------------------------------------------------------------- */
.case {
  border-top: 1px solid var(--line);
  padding-top: 1.75rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}
.case-head { display: flex; flex-direction: column; gap: 0.45rem; }

/* the two layouts, and what stands between them */
.layouts {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 1.25rem;
}
.layouts.single { grid-template-columns: minmax(0, 22rem); justify-content: start; }
@media (max-width: 46rem) {
  .layouts, .layouts.single { grid-template-columns: 1fr; }
}
.layout {
  background: var(--card);
  border: 1px solid var(--line);
  padding: 1rem 1.1rem 1.15rem;
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}
.layout-head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
.layout-head .binding { font-family: "IBM Plex Mono", monospace; font-size: 0.9rem; color: var(--accent); }
.grid2 {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 3px;
  background: var(--line);
  border: 1px solid var(--line);
}
.cell {
  background: var(--paper);
  padding: 0.55rem 0.6rem;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  min-height: 3.6rem;
}
.cell .coord {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.66rem;
  color: var(--muted);
}
.cell .holds { font-family: "IBM Plex Mono", monospace; font-size: 0.82rem; }
.axes { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); }
.arrow { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; text-align: center; }
.arrow .glyph { font-size: 1.5rem; color: var(--muted); line-height: 1; }
.chip {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  padding: 0.15rem 0.5rem;
  border-radius: 2px;
  white-space: nowrap;
}
.chip.charged { background: var(--charged-soft); color: var(--charged); }
.chip.free { background: var(--free-soft); color: var(--free); }

/* the mapping picture itself */
figure { margin: 0; display: flex; flex-direction: column; gap: 0.6rem; }
.pic {
  overflow-x: auto;
  background: #ffffff;
  border: 1px solid var(--line);
  padding: 1rem;
  display: flex;
  justify-content: center;
}
.pic svg { max-width: 100%; height: auto; }
figcaption { font-size: 0.88rem; color: var(--muted); max-width: 44em; }

/* ---- colophon ----------------------------------------------------------------- */
.colophon {
  border-top: 1px solid var(--line);
  padding-top: 1.75rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
pre {
  margin: 0;
  overflow-x: auto;
  background: var(--card);
  border: 1px solid var(--line);
  padding: 0.85rem 1rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.85rem;
  line-height: 1.5;
}
ul { margin: 0; padding-left: 1.1rem; color: var(--muted); font-size: 0.94rem; }
li { margin: 0.25rem 0; max-width: 44em; }
"""

DUPLICATE_IN_PICTURE = "*"
"""What :func:`~accelforge.frontend.mapping.mapping._binding_str` writes where a binding
loop replicates rather than partitions."""


def parse_binding(binding: str) -> list[tuple[str, str]]:
    """``"X:M, Y:N1"`` into ``[("X", "M"), ("Y", "N1")]``."""
    return [tuple(part.split(":", 1)) for part in binding.split(", ")]


def fraction_held(binding: str) -> str:
    """What share of the tensor one instance holds under `binding`."""
    partitioned = sum(1 for _, rank in parse_binding(binding) if rank != DUPLICATE_IN_PICTURE)
    return {0: "all of T1", 1: "half of T1", 2: "a quarter of T1"}[partitioned]


def inline_svg(svg: str) -> str:
    """
    A standalone SVG made safe to drop into an HTML page: Graphviz writes an XML
    prolog and a DOCTYPE, which belong to a file of its own and not to the middle of
    a document.
    """
    start = svg.index("<svg")
    return svg[start:]


def layout_html(binding: str, einsum: str) -> str:
    """
    The 2x2 array of SharedBuffer instances, each cell saying which part of T1 that
    instance holds under `binding` -- which is what a binding *is*, said without the
    mapping tree around it.
    """
    dims = parse_binding(binding)
    cells = []
    for y in range(2):
        for x in range(2):
            holds = [
                f"{rank}[{coordinate}]"
                for (_, rank), coordinate in zip(dims, (x, y))
                if rank != DUPLICATE_IN_PICTURE
            ]
            cells.append(
                "<div class='cell'>"
                f"<span class='coord'>X{x} Y{y}</span>"
                f"<span class='holds'>{html.escape(' · '.join(holds) or 'all of T1')}</span>"
                "</div>"
            )
    return (
        "<div class='layout'>"
        "<div class='layout-head'>"
        f"<span class='label'>{html.escape(einsum)}</span>"
        f"<span class='binding'>{html.escape(binding)}</span>"
        "</div>"
        f"<div class='grid2'>{''.join(cells)}</div>"
        "<div class='axes'>"
        "<span>X across, Y down</span>"
        f"<span>each instance holds {fraction_held(binding)}</span>"
        "</div>"
        "</div>"
    )


def cost_chip(result: "Result") -> str:
    if result.adapter_energy == 0:
        return "<span class='chip free'>nothing to move</span>"
    return (
        f"<span class='chip charged'>{result.adapter_energy:g} bits · "
        f"{result.adapter_latency:g} cycle{'s' if result.adapter_latency != 1 else ''}</span>"
    )


def layouts_html(result: "Result") -> str:
    """The two Einsums' layouts of T1 side by side, with the adapter between them --
    or one layout on its own where the two agree."""
    left = result.binding_of("Matmul0")
    right = result.binding_of("Matmul1")
    if left == right:
        return (
            "<div class='layouts single'>"
            + layout_html(left, "Matmul0 and Matmul1")
            + "</div>"
        )
    return (
        "<div class='layouts'>"
        + layout_html(left, "Matmul0 leaves T1")
        + "<div class='arrow'>"
        "<span class='label'>adapter</span>"
        "<span class='glyph'>&rarr;</span>"
        f"{cost_chip(result)}"
        "</div>"
        + layout_html(right, "Matmul1 wants T1")
        + "</div>"
    )


def summary_table(results: list["Result"]) -> str:
    rows = []
    for r in results:
        drawn = (
            "once, above the split"
            if r.drawn_once_above_the_split
            else "once per branch"
        )
        rows.append(
            "<tr>"
            f"<td class='mono'>{html.escape(r.case.name)}</td>"
            f"<td class='mono'>{html.escape(r.binding_of('Matmul0'))}</td>"
            f"<td class='mono'>{html.escape(r.binding_of('Matmul1'))}</td>"
            f"<td>{drawn}</td>"
            f"<td class='num'>{r.adapter_energy:g}</td>"
            f"<td class='num'>{r.adapter_latency:g}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table>"
        "<thead><tr>"
        "<th>Case</th><th>Matmul0 binds T1</th><th>Matmul1 binds T1</th>"
        "<th>T1 drawn</th><th class='num'>Adapter bits</th>"
        "<th class='num'>Adapter cycles</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def write_index(results: list["Result"], outdir: Path) -> Path:
    parts = [
        "<title>Bindings That Disagree</title>",
        FONTS,
        f"<style>{INDEX_STYLE}</style>",
        "<div class='wrap'>",
        # ---- masthead
        "<header>",
        "<span class='eyebrow'>accelforge &middot; fused mapping visualization</span>",
        "<h1>Bindings that disagree</h1>",
        "<div class='rule'></div>",
        "<p class='standfirst'>A binding belongs to one Einsum, not to the tensor. Two "
        "Einsums fused through <code>T1</code> may lay it out across the "
        "<code>SharedBuffer</code>'s instances two different ways &mdash; so the picture "
        "of the fused mapping has to say which Einsum asked for which layout, and an "
        "adapter shuffles the data in between.</p>",
        "<p class='deck'>The same two-matmul mapping below, under four pairs of "
        "bindings. Everything else is held fixed, and every action of the architecture "
        "but a network hop is free, so the energy column is a count of bits crossing "
        "the NoC and the difference between cases is the adapter alone.</p>",
        "</header>",
        # ---- glossary
        "<section class='glossary'>",
        "<div><h3>Partition</h3><p>A binding loop naming a rank splits the tensor along "
        "it, one slice per instance down that spatial dimension. <code>Y:N1</code> "
        "leaves each row of the array holding half of T1.</p></div>",
        "<div><h3>Replicate</h3><p>A binding loop naming no rank &mdash; drawn "
        "<code>*</code> &mdash; gives every instance down that dimension a whole copy "
        "of what it would otherwise have split.</p></div>",
        "<div><h3>Adapt</h3><p>Where the producer's layout is not the consumer's, the "
        "joiner inserts an adapter: the data is shuffled between instances, and the "
        "mapping is charged what the shuffle costs on the NoC.</p></div>",
        "</section>",
        # ---- summary
        "<section style='display:flex;flex-direction:column;gap:1rem'>",
        "<h2>What each pair of bindings costs</h2>",
        summary_table(results),
        "<p class='deck' style='font-size:0.94rem'>A full shuffle of T1 &mdash; the "
        f"fixture's 8&times;8 tile of 8-bit values &mdash; is {SHARED_TENSOR_BITS} bits. "
        "Being drawn apart and being charged are two different things: the third case "
        "below is drawn once per branch and still costs nothing.</p>",
        "</section>",
    ]
    for r in results:
        parts.append("<section class='case'>")
        parts.append("<div class='case-head'>")
        parts.append(f"<span class='eyebrow'>{html.escape(r.case.name)}</span>")
        parts.append(f"<h2>{html.escape(r.case.title)}</h2>")
        parts.append(f"<p>{html.escape(r.case.story)}</p>")
        parts.append("</div>")
        parts.append(layouts_html(r))
        parts.append("<figure>")
        parts.append(f"<div class='pic'>{inline_svg(r.svg)}</div>")
        drawn = (
            "The two branches hold the same layout of T1, so the mapping draws it once, "
            "above the split, as the one thing both Einsums share."
            if r.drawn_once_above_the_split
            else "Neither binding may speak for the other, so T1 is drawn twice &mdash; "
            "once inside each branch, carrying the binding of the Einsum that asked "
            "for it."
        )
        parts.append(f"<figcaption>{drawn}</figcaption>")
        parts.append("</figure>")
        parts.append("</section>")

    parts += [
        "<section class='colophon'>",
        "<h2>Reproducing this</h2>",
        "<pre>python experiments/binding_adapter_visualization.py</pre>",
        "<ul>",
        "<li>Architecture: <code>tests/input_files/binding_adapter.arch.yaml</code> "
        "&mdash; a 2&times;2 array whose <code>SharedBuffer</code> is physically "
        "distributed over it, with a NoC below it to move data between instances.</li>",
        "<li>Mapping: <code>tests/input_files/mapping/"
        "fused_matmuls_to_binding_adapter.yaml</code> &mdash; two matmuls fused through "
        "T1, each backing it in the SharedBuffer under spatial loops of its own. The "
        "cases differ only in the <code>!Binding</code> node of each branch.</li>",
        "<li>The rule the pictures follow lives in <code>Split._pop_agreed_bindings"
        "</code>: the Storage of a shared tensor is always lifted above the split, and "
        "its binding joins it there only when the branches agree on it.</li>",
        "<li>Locked down by <code>TestAdapterInTheVisualization</code> in "
        "<code>tests/test_binding_adapter.py</code>.</li>",
        "</ul>",
        "</section>",
        "</div>",
    ]
    path = outdir / "index.html"
    path.write_text("\n".join(parts))
    return path


def print_summary(results: list[Result]) -> None:
    header = f"{'case':16s} {'Matmul0':12s} {'Matmul1':12s} {'T1 drawn':22s} {'energy':>8s} {'latency':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        drawn = "above the split" if r.drawn_once_above_the_split else "once per branch"
        print(
            f"{r.case.name:16s} {r.binding_of('Matmul0'):12s} "
            f"{r.binding_of('Matmul1'):12s} {drawn:22s} "
            f"{r.adapter_energy:8g} {r.adapter_latency:8g}"
        )
    print()
    print(
        f"Energy is in bits moved over the NoC; a full shuffle of T1 is "
        f"{SHARED_TENSOR_BITS} bits."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "binding_adapter_visualization",
        help="Where to write the SVGs and the index page.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[c.name for c in CASES],
        default=[c.name for c in CASES],
        help="Which cases to render. Defaults to all of them.",
    )
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in [c for c in CASES if c.name in args.cases]:
        print(f"rendering {case.name}...", file=sys.stderr)
        result = run(case)
        (args.outdir / f"{case.name}.svg").write_text(result.svg)
        results.append(result)

    index = write_index(results, args.outdir)
    print_summary(results)
    print(f"\nWrote {len(results)} SVGs and {index} to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
