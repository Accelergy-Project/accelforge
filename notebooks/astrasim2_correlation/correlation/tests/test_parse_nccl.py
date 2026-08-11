"""Tests for parse_nccl.py: the nccl-tests / torus_bench log parser.

Fixtures under ``tests/fixtures/`` hold representative raw stdout captured
from the two profiling tools (see the ``correlation`` package's WP2 spec
for the exact log formats). These tests exercise the parsing functions
directly (:func:`parse_nccl.parse_nccl_tests`,
:func:`parse_nccl.parse_torus_bench`), the CSV writer
(:func:`parse_nccl.rows_to_csv`), and the CLI entry point end-to-end via
:mod:`subprocess`, using ``sys.executable`` so the tests run under whatever
interpreter is running pytest itself (matching how ``run_profile.sh``
invokes this script with a plain ``python3``).
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from parse_nccl import (
    UNIFIED_CSV_FIELDNAMES,
    parse_nccl_tests,
    parse_torus_bench,
    rows_to_csv,
)

# Design: resolve fixture/script paths relative to this test file (not the
# CWD) so the suite passes regardless of where pytest is invoked from, per
# the spec's instruction to load fixtures with pathlib relative to the test
# file.
TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
PARSE_NCCL_SCRIPT = TESTS_DIR.parent / "parse_nccl.py"

FC_ALL_REDUCE_LOG = FIXTURES_DIR / "fc_all_reduce.log"
FC_ALLTOALL_LOG = FIXTURES_DIR / "fc_alltoall.log"
TORUS_ALL_REDUCE_LOG = FIXTURES_DIR / "torus_all_reduce.log"


def test_parse_nccl_tests_all_reduce():
    """fc_all_reduce.log parses to 2 rows with the expected first-row values.

    Exercises the common case: a well-formed all_reduce_perf log with a
    standard 13-token data row (size, count, type, redop, root, then the
    trailing-8 out-of-place/in-place metrics).
    """
    text = FC_ALL_REDUCE_LOG.read_text(encoding="utf-8")
    rows = parse_nccl_tests(text)

    assert len(rows) == 2
    first = rows[0]
    assert first["size_bytes"] == 1048576
    assert first["count"] == 262144
    assert first["dtype"] == "float"
    assert first["time_us"] == pytest.approx(98.52)
    assert first["algbw_GBps"] == pytest.approx(10.64)
    assert first["busbw_GBps"] == pytest.approx(18.62)
    assert first["wrong"] == "0"


def test_parse_nccl_tests_alltoall_na_wrong():
    """fc_alltoall.log parses to 2 rows and exercises the 'N/A' #wrong path.

    alltoall_perf prints redop="none" and root="-1" instead of a real
    reduction op/root -- this test confirms the trailing-8-token rule
    parses those rows correctly regardless, and that an "N/A" out-of-place
    #wrong value (validation disabled for that data point) is preserved
    as the literal string "N/A" rather than raising or being coerced to a
    number.
    """
    text = FC_ALLTOALL_LOG.read_text(encoding="utf-8")
    rows = parse_nccl_tests(text)

    assert len(rows) == 2
    assert rows[0]["wrong"] == "N/A"
    assert rows[0]["size_bytes"] == 1048576
    assert rows[0]["time_us"] == pytest.approx(120.44)
    # Second row is a normal (non-N/A) row, confirming N/A handling on row 0
    # didn't leak into subsequent parsing.
    assert rows[1]["wrong"] == "0"
    assert rows[1]["size_bytes"] == 2097152


def test_parse_torus_bench_all_reduce():
    """torus_all_reduce.log parses to 3 rows with the expected first row.

    Exercises the comma-delimited TORUSBENCH sentinel format and the
    check-field-to-wrong-string remapping (check "1" -> wrong "0").
    """
    text = TORUS_ALL_REDUCE_LOG.read_text(encoding="utf-8")
    rows = parse_torus_bench(text)

    assert len(rows) == 3
    first = rows[0]
    assert first["collective"] == "all_reduce"
    assert first["dims"] == "2x2x2"
    assert first["size_bytes"] == 1048576
    assert first["time_us"] == pytest.approx(142.11)
    assert first["wrong"] == "0"


def test_rows_to_csv_round_trip(tmp_path):
    """rows_to_csv() writes the exact unified header, in order, and empty
    algbw/busbw cells for torus rows round-trip as empty strings.

    Builds one nccl-tests-shaped row and one torus_bench-shaped row (as
    main() would produce them) and confirms the on-disk CSV, when read back
    with csv.DictReader, has fieldnames matching UNIFIED_CSV_FIELDNAMES
    exactly (order included) and that the torus row's bandwidth columns are
    empty rather than "None" or some other stand-in.
    """
    rows = [
        {
            "source": "nccl-tests",
            "topology": "fc",
            "dims": "8",
            "collective": "all_reduce",
            "size_bytes": 1048576,
            "count": 262144,
            "dtype": "float",
            "time_us": 98.52,
            "algbw_GBps": 10.64,
            "busbw_GBps": 18.62,
            "wrong": "0",
        },
        {
            "source": "torus_bench",
            "topology": "torus",
            "dims": "2x2x2",
            "collective": "all_reduce",
            "size_bytes": 1048576,
            "count": "",
            "dtype": "",
            "time_us": 142.11,
            "algbw_GBps": "",
            "busbw_GBps": "",
            "wrong": "0",
        },
    ]
    out_path = tmp_path / "unified.csv"

    rows_to_csv(rows, out_path)

    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == UNIFIED_CSV_FIELDNAMES
        read_rows = list(reader)

    assert len(read_rows) == 2
    assert read_rows[1]["source"] == "torus_bench"
    assert read_rows[1]["algbw_GBps"] == ""
    assert read_rows[1]["busbw_GBps"] == ""


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke parse_nccl.py's CLI as a subprocess.

    Parameters
    ----------
    *args : str
        Arguments to pass after the script path, e.g. the raw log path and
        ``--source``/``--out``/etc. flags.

    Returns
    -------
    subprocess.CompletedProcess
        Result of the invocation, with stdout/stderr captured as text.

    Notes
    -----
    Design: uses ``sys.executable`` (not a hard-coded ``python3``) so the
    subprocess runs under the exact interpreter executing the test suite,
    matching pytest's own environment rather than risking a PATH mismatch.
    """
    return subprocess.run(
        [sys.executable, str(PARSE_NCCL_SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_cli_nccl_tests_end_to_end(tmp_path):
    """CLI parses an nccl-tests log to a CSV with the expected row count."""
    out_path = tmp_path / "fc_all_reduce.csv"
    result = _run_cli(
        str(FC_ALL_REDUCE_LOG),
        "--source",
        "nccl-tests",
        "--collective",
        "all_reduce",
        "--topology",
        "fc",
        "--out",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    with out_path.open(newline="", encoding="utf-8") as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == 2
    assert all(row["source"] == "nccl-tests" for row in read_rows)
    assert all(row["dims"] == "8" for row in read_rows)


def test_cli_torus_bench_end_to_end(tmp_path):
    """CLI parses a torus_bench log to a CSV with the expected row count."""
    out_path = tmp_path / "torus_all_reduce.csv"
    result = _run_cli(
        str(TORUS_ALL_REDUCE_LOG),
        "--source",
        "torus_bench",
        "--collective",
        "all_reduce",
        "--topology",
        "torus",
        "--dims",
        "2x2x2",
        "--out",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    with out_path.open(newline="", encoding="utf-8") as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == 3
    assert all(row["source"] == "torus_bench" for row in read_rows)
    assert all(row["algbw_GBps"] == "" for row in read_rows)


def test_cli_torus_bench_collective_mismatch_errors(tmp_path):
    """CLI exits non-zero when --collective disagrees with the log content.

    Bonus coverage beyond the spec's 6 mandated cases: confirms the
    validate-CLI-against-sentinel behavior documented in parse_nccl.main()
    actually triggers a hard failure rather than silently mislabeling data,
    since a silent mismatch here would corrupt the correlation notebook's
    inputs without any visible signal.
    """
    out_path = tmp_path / "should_not_be_created.csv"
    result = _run_cli(
        str(TORUS_ALL_REDUCE_LOG),
        "--source",
        "torus_bench",
        "--collective",
        "all_gather",  # deliberately wrong; fixture is all_reduce
        "--topology",
        "torus",
        "--out",
        str(out_path),
    )

    assert result.returncode != 0
    assert not out_path.exists()


@pytest.mark.parametrize(
    "log_text",
    [
        "# just a comment\n\n# another comment\n",
        "hello world\n",
        "# nThread 1 nGpus 8\nhello world\n# trailing comment\n",
    ],
)
def test_parse_nccl_tests_skips_malformed_lines(log_text):
    """Comment-only, blank, and garbage lines are skipped without raising."""
    assert parse_nccl_tests(log_text) == []


@pytest.mark.parametrize(
    "log_text",
    [
        "# torus_bench collective=all_reduce dims=2x2x2\n",
        "hello world\n",
        "# header\nhello world\nTORUSBENCH,incomplete,field\n",
    ],
)
def test_parse_torus_bench_skips_malformed_lines(log_text):
    """Comment-only, blank, garbage, and malformed sentinel lines are
    skipped without raising (including a TORUSBENCH, line with too few
    comma-separated fields)."""
    assert parse_torus_bench(log_text) == []
