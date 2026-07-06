"""
Shared test helpers for the ISL distributed-buffer multicast model test suite.

Factors out two pieces of logic that used to be byte-for-byte duplicated across
``test_multicast.py``, ``test_fully_connected.py``, and ``test_xy_routing.py``:

1. ``construct_spacetime`` -- turning a yaml ``dims`` list into ``Tag`` objects.
2. ``run_hops_gamut`` -- the "load a yaml of test cases, build a ``Fill``
   /``Occupancy``/``dist_fn`` triple, run one ``TransferModel``, and compare
   ``.hops`` against an expected key" loop, which is identical across the
   three hop-oracle test files and differs only in which model class, which
   yaml directory, and which ``expected`` key (``hypercube_hops`` /
   ``fully_connected_hops`` / ``xy_routing_hops``) is used.

With this module in place, ``test_multicast.py``, ``test_fully_connected.py``,
and ``test_xy_routing.py`` reduce to thin parametrizations: a ``TestCase`` whose
single ``test_gamut`` method calls ``run_hops_gamut`` with its model class, yaml
path, and expected-key string.

Import note
-----------
This module (and every test module in this package) imports the canonical
``load_solutions`` helper via the *absolute* path ``tests.isl.util`` rather than
a local copy. That import only resolves if the repository root is on
``sys.path``, which is the case when the suite is invoked from the repo root as

    PATH="$HOME/.local/bin:$PATH" .venv/bin/python -m pytest tests/isl/distributed/ -q

(pytest inserts the current working directory / rootdir onto ``sys.path``).
Running this module or its dependents from a different working directory, or
via a bare module path without the repo root on ``sys.path``, will raise
``ModuleNotFoundError: No module named 'tests'``.
"""

from pathlib import Path

import islpy as isl

from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    # Design (D6): the "evaluate a parameter-free PwQPolynomial at the zero
    # point" idiom lives in exactly one place now -- see `_eval_const`'s
    # docstring for the history of it drifting across four call sites with
    # inconsistent return types. This module used to inline its own copy
    # (`info.hops.eval(isl.Point.zero(...))`, which returns an `isl.Val`
    # despite looking `int`-shaped); importing the canonical helper instead
    # keeps this the single non-test call site left outside
    # `distributed_buffers.py` and guarantees a real Python `int`.
    _eval_const,
)
from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import (
    # Data movement descriptors.
    Fill,
    Occupancy,
    # Tags
    Tag,
    SpatialTag,
    TemporalTag,
)
from accelforge.model._looptree.reuse.isl.spatial import TransferInfo, TransferModel

# Design: this package (formerly `tests/not_working/distribuffers/`) used to
# carry a byte-identical copy of `tests/isl/util.py` as its own `util.py`.
# Rather than keep two copies of `load_solutions` in sync by hand, it imports
# the canonical one directly -- see the "Import note" above for the
# run-from-repo-root requirement this implies.
from tests.isl.util import load_solutions


def construct_spacetime(dims: list[dict]) -> list[Tag]:
    """
    Convert a yaml ``dims`` list (each entry a dict with a ``type`` key, plus
    ``spatial_dim``/``target`` for spatial entries) into the corresponding
    ``Tag`` objects, in order.

    Parameters
    ----------
    dims:
        The list of dim-tag dicts as loaded from a test-case yaml, e.g.
        ``[{"type": "Spatial", "spatial_dim": 0, "target": 0}, ...]``.

    Returns
    -------
    ``list[Tag]`` where ``list[i]`` is the tag corresponding to ``dims[i]``.
    """
    spacetime: list[Tag] = []
    for dim in dims:
        if dim["type"] == "Temporal":
            spacetime.append(TemporalTag())
        elif dim["type"] == "Spatial":
            spacetime.append(SpatialTag(dim["spatial_dim"], dim["target"]))

    return spacetime


def run_hops_gamut(model_cls: type[TransferModel], yaml_path: Path, expected_key: str) -> None:
    """
    Run every test case in a yaml gamut file through ``model_cls`` and assert
    the resulting ``TransferInfo.hops`` matches the case's expected value.

    # Design: `test_multicast.py`, `test_fully_connected.py`, and
    # `test_xy_routing.py` each had their own copy of this loop; the only
    # differences were the model class being exercised, the yaml directory the
    # cases were loaded from, and which key of `test["expected"]` held the
    # oracle value (`hypercube_hops` / `fully_connected_hops` /
    # `xy_routing_hops`). Parametrizing on those three makes the loop itself
    # single-sourced.

    Parameters
    ----------
    model_cls:
        A ``TransferModel`` subclass constructible as ``model_cls(dist_fn)``
        (every model in ``distributed_buffers.py`` follows this signature).
    yaml_path:
        Path to a yaml file of test cases, each a dict with ``dims``, ``fill``,
        ``occ``, ``dist_fn``, and ``expected`` keys (see any
        ``*/test_cases.yaml`` under this package for the schema).
    expected_key:
        The key within each case's ``expected`` dict holding the oracle hop
        count for this model (e.g. ``"xy_routing_hops"``). A ``None`` value at
        this key marks a case as still in progress: rather than failing, the
        case's inputs/output are printed for manual inspection (preserving the
        original tests' "unimplemented case" debugging affordance).

    Raises
    ------
    AssertionError
        If a case's expected value is not ``None`` and the model's computed
        ``hops`` does not match it.
    """
    testcases: dict = load_solutions(yaml_path)
    for test in testcases:
        # Reads test case parameters and constructs the necessary objects.
        dim_tags: list[Tag] = construct_spacetime(test["dims"])
        fill: Fill = Fill(dim_tags, test["fill"])
        occ: Occupancy = Occupancy(dim_tags, test["occ"])
        dist_fn: isl.Map = test["dist_fn"]
        model: TransferModel = model_cls(dist_fn)

        # Applies the model.
        info: TransferInfo = model.apply(0, fill, occ)
        # Checks the results. `hops` is parameter-free (the validated regime
        # every distributed model targets, see `_eval_const`'s precondition),
        # so this is a scalar extraction, not a real evaluation-at-a-point.
        sum_extract: int = _eval_const(info.hops)

        expected = test["expected"][expected_key]
        # The block is used for debugging test cases not yet implemented.
        if expected is None:
            print("~~~Test case in progress:~~~")
            print(f"Fill: {fill}")
            print(f"Occ: {occ}")
            print(f"Dist Fn: {dist_fn}")
            print(f"Returned: {sum_extract}")
        else:
            assert sum_extract == expected, (
                f"{model_cls.__name__} hops mismatch: got {sum_extract}, "
                f"expected {expected} ({expected_key})\n"
                f"Fill: {fill}\nOcc: {occ}\nDist Fn: {dist_fn}"
            )
