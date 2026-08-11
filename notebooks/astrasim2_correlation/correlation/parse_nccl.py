"""Parse raw NCCL collective-communication profiling logs into tidy CSVs.

This module turns the stdout of two profiling tools into a single, unified
CSV schema that the correlation notebook (``correlation.ipynb``) consumes:

1. **nccl-tests** (upstream NVIDIA binaries: ``all_reduce_perf``,
   ``all_gather_perf``, ``reduce_scatter_perf``, ``alltoall_perf``,
   ``broadcast_perf``, ``sendrecv_perf``) run against the fully-connected
   (FC) NVSwitch fabric.
2. ``torus_bench`` (a custom binary built by a sibling work package) run
   against a torus-topology emulation on the same physical fabric.

Both tools are parsed independently (:func:`parse_nccl_tests` and
:func:`parse_torus_bench`) and their results are reshaped into the shared
schema documented at :data:`UNIFIED_CSV_FIELDNAMES` before being written to
disk with :func:`rows_to_csv`. The module is also a CLI entry point (see
:func:`main`) so it can be invoked directly from ``run_profile.sh`` on the
profiling instance without any extra Python dependencies.

Notes
-----
Stdlib-only by design: this script runs on a freshly provisioned EC2
instance where installing a virtualenv is unwanted overhead. Only ``csv``,
``argparse``, and ``pathlib`` (plus ``sys`` for the CLI entry point) are
used, so it works under any plain ``python3`` >= 3.8.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Union

# Design: the unified schema is a module-level constant (rather than being
# implicit in whatever keys happen to be in the first row dict) so that
# rows_to_csv() always emits a stable, predictable column order regardless
# of which source produced the rows, and so the notebook can rely on the
# header never silently reordering itself as this module evolves.
UNIFIED_CSV_FIELDNAMES: list[str] = [
    "source",
    "topology",
    "dims",
    "collective",
    "size_bytes",
    "count",
    "dtype",
    "time_us",
    "algbw_GBps",
    "busbw_GBps",
    "wrong",
]

# Minimum number of whitespace-separated tokens a nccl-tests data row must
# have before it is considered parseable. A row always carries at least
# size, count, type, redop, root (5 leading columns) plus the 8 trailing
# out-of-place/in-place metric columns = 13 tokens; 10 is used as a looser
# lower bound per the spec so that unexpected/future nccl-tests column
# layouts with slightly fewer leading columns are still accepted as long as
# the trailing-8 structure holds.
_MIN_NCCL_TESTS_TOKENS = 10

# Number of trailing tokens on a nccl-tests data row that carry the
# out-of-place/in-place timing results. This is the anchor of the parsing
# strategy; see parse_nccl_tests() docstring for the rationale.
_TRAILING_METRIC_TOKENS = 8

# Sentinel line prefix emitted by torus_bench for each data point. Chosen
# by the sibling work package specifically so it is trivial to grep/parse
# out of interleaved '#'-prefixed human-readable log noise.
_TORUS_SENTINEL_PREFIX = "TORUSBENCH,"


def parse_nccl_tests(text: str) -> list[dict]:
    """Parse the stdout of an nccl-tests collective benchmark binary.

    nccl-tests binaries (``all_reduce_perf``, ``all_gather_perf``,
    ``reduce_scatter_perf``, ``alltoall_perf``, ``broadcast_perf``,
    ``sendrecv_perf``) share a common output shape: a block of ``#``-prefixed
    header/comment lines, followed by one data row per message size, followed
    by ``#``-prefixed summary lines. The *leading* columns of a data row vary
    per collective (e.g. ``redop``/``root`` are meaningless for
    ``alltoall_perf`` and print as ``none``/``-1``), but the *trailing* eight
    columns are always, in order: out-of-place ``time``, ``algbw``, ``busbw``,
    ``#wrong``, then in-place ``time``, ``algbw``, ``busbw``, ``#wrong``.

    Parameters
    ----------
    text : str
        Raw stdout captured from an nccl-tests binary invocation. May
        contain blank lines and ``#``-prefixed comment/header/summary lines
        interleaved with data rows.

    Returns
    -------
    list of dict
        One dict per parsed data row, in file order, with keys:
        ``size_bytes`` (int), ``count`` (int), ``dtype`` (str),
        ``time_us`` (float), ``algbw_GBps`` (float), ``busbw_GBps`` (float),
        ``wrong`` (str; ``"0"``, another digit string, or ``"N/A"`` when
        validation was disabled for the run). Only the *out-of-place*
        metrics are kept, matching the spec's trailing-token convention;
        the in-place metrics are intentionally discarded since the
        correlation study only needs one consistent number per size.

    Notes
    -----
    Design: rather than hand-writing a distinct column layout per collective
    (which would need to track every nccl-tests release), this uses a single
    robust rule anchored on the *trailing* 8 tokens, which nccl-tests has
    kept stable across collectives and versions even as leading columns
    (redop, root) have been added/repurposed. A line is treated as a data
    row only if tokens[0] and tokens[1] both parse as int -- this
    distinguishes real data rows (which always start with two integers:
    size in bytes, element count) from stray non-'#' lines (blank-ish
    whitespace, malformed output, or future header formats) without needing
    to hard-code the '#' comment convention as the *only* skip signal.

    This function does not raise on malformed input; unparseable lines are
    silently skipped so that a partially-corrupt log (e.g. truncated by a
    crashed run) still yields whatever valid rows it contains.

    Examples
    --------
    >>> text = (
    ...     "# nThread 1 nGpus 8\\n"
    ...     "     1048576        262144     float     sum      -1    "
    ...     "98.52   10.64   18.62      0    97.11   10.80   18.90      0\\n"
    ... )
    >>> rows = parse_nccl_tests(text)
    >>> rows[0]["size_bytes"], rows[0]["time_us"], rows[0]["wrong"]
    (1048576, 98.52, '0')
    """
    rows: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip blank lines and '#'-prefixed header/comment/summary lines.
        if not stripped or stripped.startswith("#"):
            continue

        tokens = stripped.split()
        if len(tokens) < _MIN_NCCL_TESTS_TOKENS:
            continue

        try:
            size_bytes = int(tokens[0])
            count = int(tokens[1])
        except ValueError:
            # Not a data row (e.g. stray non-'#' text); skip rather than
            # raise so one bad line doesn't abort parsing of an otherwise
            # good log.
            continue

        dtype = tokens[2]
        try:
            time_us = float(tokens[-_TRAILING_METRIC_TOKENS])
            algbw_gbps = float(tokens[-_TRAILING_METRIC_TOKENS + 1])
            busbw_gbps = float(tokens[-_TRAILING_METRIC_TOKENS + 2])
        except ValueError:
            # The trailing columns didn't parse as floats -- not a real
            # data row (defensive; shouldn't happen given the int checks
            # above already filtered most non-data lines).
            continue
        wrong = tokens[-_TRAILING_METRIC_TOKENS + 3]

        rows.append(
            {
                "size_bytes": size_bytes,
                "count": count,
                "dtype": dtype,
                "time_us": time_us,
                "algbw_GBps": algbw_gbps,
                "busbw_GBps": busbw_gbps,
                "wrong": wrong,
            }
        )
    return rows


def parse_torus_bench(text: str) -> list[dict]:
    """Parse the stdout of the custom ``torus_bench`` binary.

    ``torus_bench`` prints ``#``-prefixed human-readable header lines plus
    machine-parseable sentinel lines of the exact form::

        TORUSBENCH,<collective>,<dims e.g. 2x2x2>,<total_size_bytes>,<time_us_avg>,<check 0|1|->

    Parameters
    ----------
    text : str
        Raw stdout captured from a ``torus_bench`` invocation.

    Returns
    -------
    list of dict
        One dict per ``TORUSBENCH,`` sentinel line, in file order, with
        keys: ``collective`` (str), ``dims`` (str, e.g. ``"2x2x2"``),
        ``size_bytes`` (int), ``time_us`` (float), ``wrong`` (str; one of
        ``"0"`` (check passed), ``"1"`` (check failed), or ``"N/A"``
        (validation was not run for this data point)).

    Notes
    -----
    Design: the sentinel line is comma-delimited (unlike nccl-tests'
    whitespace-delimited columns) specifically so the sibling work package
    could emit it without worrying about column-alignment padding; this
    parser simply looks for the fixed ``TORUSBENCH,`` prefix and splits on
    commas, ignoring every other line (including the human-readable ``#``
    header). This makes the parser forward-compatible with additional
    ``#``-prefixed diagnostic lines torus_bench might add later.

    The ``check`` field's three-way encoding (``1``/``0``/``-``) is
    remapped onto the same ``wrong`` vocabulary nccl-tests uses (a per-row
    "wrongness" indicator string) so downstream consumers (rows_to_csv,
    the notebook) can treat the ``wrong`` column uniformly across sources:
    ``check == "1"`` (validation ran and passed) maps to ``wrong = "0"``
    (zero wrong elements); ``check == "-"`` (validation was skipped for
    this run) maps to ``wrong = "N/A"``, mirroring nccl-tests' own "N/A"
    convention for validation-disabled runs; anything else (i.e.
    ``check == "0"``, validation ran and failed) maps to ``wrong = "1"``.

    Malformed sentinel lines (wrong field count, non-numeric size/time) are
    silently skipped rather than raising, for the same reasons as
    :func:`parse_nccl_tests`.
    """
    rows: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_TORUS_SENTINEL_PREFIX):
            continue

        fields = stripped.split(",")
        # TORUSBENCH,<collective>,<dims>,<size_bytes>,<time_us>,<check> = 6 fields.
        if len(fields) != 6:
            continue

        _, collective, dims, size_bytes_str, time_us_str, check = fields
        try:
            size_bytes = int(size_bytes_str)
            time_us = float(time_us_str)
        except ValueError:
            continue

        if check == "1":
            wrong = "0"
        elif check == "-":
            wrong = "N/A"
        else:
            wrong = "1"

        rows.append(
            {
                "collective": collective,
                "dims": dims,
                "size_bytes": size_bytes,
                "time_us": time_us,
                "wrong": wrong,
            }
        )
    return rows


def rows_to_csv(rows: list[dict], out_path: Union[str, Path]) -> None:
    """Write unified-schema rows to a CSV file.

    Parameters
    ----------
    rows : list of dict
        Rows already reshaped into the unified schema (see
        :data:`UNIFIED_CSV_FIELDNAMES` for the exact column set and order).
        Each dict must contain every key in ``UNIFIED_CSV_FIELDNAMES``;
        missing keys are written as empty cells by :class:`csv.DictWriter`
        default behavior is NOT relied upon here -- callers are expected to
        supply complete rows (see :func:`main` for how CLI callers build
        them). Extra keys beyond the unified schema are rejected by
        :class:`csv.DictWriter` (``extrasaction="raise"``, the default) so
        schema drift is caught early rather than silently dropped.
    out_path : str or pathlib.Path
        Destination file path. Parent directories are NOT created by this
        function; callers must ensure the directory exists.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a row dict contains a key not present in
        :data:`UNIFIED_CSV_FIELDNAMES` (raised by the underlying
        :class:`csv.DictWriter`).
    OSError
        If ``out_path`` cannot be opened for writing (e.g. parent directory
        does not exist, permission denied).

    Notes
    -----
    Opens the file with ``newline=""`` as recommended by the :mod:`csv`
    module docs, so that the csv module's own line-ending handling is used
    verbatim rather than being double-translated by Python's text-mode
    newline translation.
    """
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for this module.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting the raw log path plus labeling/validation flags
        described in the module CLI usage (see :func:`main`).
    """
    parser = argparse.ArgumentParser(
        prog="parse_nccl.py",
        description=(
            "Parse a raw nccl-tests or torus_bench profiling log into the "
            "unified CSV schema consumed by the correlation notebook."
        ),
    )
    parser.add_argument(
        "raw_log",
        type=Path,
        help="Path to the raw stdout log captured from the profiling binary.",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["nccl-tests", "torus_bench"],
        help="Which tool produced raw_log.",
    )
    parser.add_argument(
        "--collective",
        required=True,
        help=(
            "Collective name. For --source nccl-tests this labels every "
            "output row directly (the tool's own stdout does not name the "
            "collective). For --source torus_bench this is instead "
            "cross-checked against the collective embedded in each "
            "TORUSBENCH sentinel line; a mismatch is an error."
        ),
    )
    parser.add_argument(
        "--topology",
        required=True,
        choices=["fc", "torus"],
        help="Fabric topology label to stamp onto every output row.",
    )
    parser.add_argument(
        "--dims",
        default=None,
        help=(
            "Dimension string (e.g. '2x2x2'). For --source nccl-tests this "
            "overrides the default dims label of '8' (the fixed GPU count "
            "of a single NVSwitch-connected node). For --source "
            "torus_bench, if given, it is cross-checked against the dims "
            "embedded in each TORUSBENCH sentinel line; a mismatch is an "
            "error. If omitted for torus_bench, the sentinel's own dims "
            "value is used unchecked."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Destination CSV path.",
    )
    return parser


# Default dims label for nccl-tests rows when --dims is not supplied on the
# CLI. FC-leg runs are always against a single 8x-H100 NVSwitch node, so "8"
# (the GPU count) is the natural default; --dims exists mainly to let the
# CLI stay uniform with the torus_bench invocation and to support future
# multi-node FC runs without changing this module.
_DEFAULT_NCCL_TESTS_DIMS = "8"


def main(argv: Union[list, None] = None) -> None:
    """CLI entry point: parse a raw log and write the unified CSV.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector to parse in place of ``sys.argv[1:]``. Primarily
        useful for testing; production invocations (from ``run_profile.sh``)
        pass ``None`` and rely on ``sys.argv``.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Raised by :mod:`argparse` on invalid/missing arguments (exit code
        2), or explicitly via ``parser.error()`` when a ``--collective``/
        ``--dims`` value supplied on the CLI disagrees with the value
        embedded in a torus_bench sentinel line (exit code 2). Also raised
        implicitly if ``raw_log`` cannot be read (propagates as an
        unhandled :class:`OSError`, not caught here -- a missing/unreadable
        input log is a hard setup error the caller (``run_profile.sh``)
        should see immediately rather than have masked).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    text = args.raw_log.read_text(encoding="utf-8")

    if args.source == "torus_bench":
        parsed = parse_torus_bench(text)
        # Design: validate CLI-supplied --collective/--dims against what
        # the sentinel lines actually say, rather than trusting the CLI
        # blindly. This catches operator error in run_profile.sh (e.g. a
        # copy-paste mistake wiring the wrong collective's log into the
        # wrong parse invocation) at parse time instead of silently
        # mislabeling data that later gets combined into the notebook.
        for parsed_row in parsed:
            if parsed_row["collective"] != args.collective:
                parser.error(
                    f"--collective {args.collective!r} does not match "
                    f"collective {parsed_row['collective']!r} found in "
                    f"{args.raw_log}"
                )
            if args.dims is not None and parsed_row["dims"] != args.dims:
                parser.error(
                    f"--dims {args.dims!r} does not match dims "
                    f"{parsed_row['dims']!r} found in {args.raw_log}"
                )
        rows = [
            {
                "source": "torus_bench",
                "topology": args.topology,
                "dims": parsed_row["dims"],
                "collective": parsed_row["collective"],
                "size_bytes": parsed_row["size_bytes"],
                # Design: count/dtype/algbw/busbw are left empty for torus
                # rows per spec -- bandwidth conventions for the torus
                # topology (e.g. what counts as "algorithm bandwidth" when
                # hops differ per link) are derived in the notebook from
                # size_bytes/time_us/dims, not computed here, to keep this
                # parser topology-agnostic.
                "count": "",
                "dtype": "",
                "time_us": parsed_row["time_us"],
                "algbw_GBps": "",
                "busbw_GBps": "",
                "wrong": parsed_row["wrong"],
            }
            for parsed_row in parsed
        ]
    else:
        parsed = parse_nccl_tests(text)
        dims = args.dims if args.dims is not None else _DEFAULT_NCCL_TESTS_DIMS
        rows = [
            {
                "source": "nccl-tests",
                "topology": args.topology,
                "dims": dims,
                "collective": args.collective,
                "size_bytes": parsed_row["size_bytes"],
                "count": parsed_row["count"],
                "dtype": parsed_row["dtype"],
                "time_us": parsed_row["time_us"],
                "algbw_GBps": parsed_row["algbw_GBps"],
                "busbw_GBps": parsed_row["busbw_GBps"],
                "wrong": parsed_row["wrong"],
            }
            for parsed_row in parsed
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_to_csv(rows, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
