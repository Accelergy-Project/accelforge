"""Tests for orchestrate.py.

Like ``test_provision_teardown.py``, these tests never touch real AWS,
never touch the network, and never spawn a real ``ssh``/``scp`` process.
Where ``test_provision_teardown.py`` achieves that with
``botocore.stub.Stubber`` (because it exercises ``provision.py``'s and
``teardown.py``'s own boto3-request-building code directly), this module
takes a different, coarser-grained approach: ``orchestrate.py`` mostly
*sequences* those already-tested functions rather than building its own
boto3 requests, so the provisioning functions themselves
(``resolve_ami``, ``ensure_key_pair``, ``ensure_security_group``,
``launch_instance``, ``wait_for_instance``), ``teardown.teardown_run``,
``orchestrate.caller_ip``, ``orchestrate.boto3``, and
``orchestrate.subprocess.run`` are all monkeypatched with lightweight
fakes that record what they were called with. This exercises
``orchestrate.py``'s own sequencing/argv-building logic -- the part this
work package is actually responsible for -- without needing Stubber
responses for calls this module never makes itself.

Import strategy
-----------------
Mirrors ``test_provision_teardown.py`` exactly: ``correlation/`` has no
``__init__.py`` (deliberately not a package), so it is inserted onto
``sys.path`` explicitly before importing ``config``/``orchestrate``,
rather than relying on pytest's own rootdir-insertion behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# boto3 is not (and must not become) an accelforge package dependency; skip
# this whole module rather than error if it is not installed in the
# environment running the tests. orchestrate.py's own import of boto3 is
# guarded the same way provision.py's/teardown.py's is, but the tests below
# exercise real code paths that assume boto3 (and its stub-friendly
# botocore internals) are present, matching test_provision_teardown.py's
# identical importorskip.
boto3 = pytest.importorskip("boto3")

_CORRELATION_DIR = Path(__file__).resolve().parent.parent
if str(_CORRELATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CORRELATION_DIR))

import config  # noqa: E402
import orchestrate  # noqa: E402

Config = config.Config


# ---------------------------------------------------------------------------
# legs_for
# ---------------------------------------------------------------------------


def test_legs_for_both_returns_fc_then_torus():
    """"both" expands to the FC leg followed by the torus leg, in that order."""
    assert orchestrate.legs_for("both") == ["fc", "torus"]


def test_legs_for_fc_returns_single_leg():
    assert orchestrate.legs_for("fc") == ["fc"]


def test_legs_for_torus_returns_single_leg():
    assert orchestrate.legs_for("torus") == ["torus"]


# ---------------------------------------------------------------------------
# build_ssh_cmd / build_scp_cmd
# ---------------------------------------------------------------------------


def test_build_ssh_cmd_exact_argv(tmp_path):
    """build_ssh_cmd's argv matches the exact option set/order the spec requires."""
    key_path = tmp_path / "keys" / "run.pem"
    known_hosts_path = tmp_path / ".state" / "known_hosts"

    cmd = orchestrate.build_ssh_cmd(key_path, "ubuntu", "203.0.113.9", "echo hi", known_hosts_path)

    assert cmd == [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        "-o",
        "ConnectTimeout=30",
        "ubuntu@203.0.113.9",
        "echo hi",
    ]


def test_build_scp_cmd_non_recursive_push_argv(tmp_path):
    """A multi-source, non-recursive push builds argv with sources then dest, no -r."""
    key_path = tmp_path / "keys" / "run.pem"
    known_hosts_path = tmp_path / ".state" / "known_hosts"
    sources = ["setup_node.sh", "run_profile.sh", "parse_nccl.py"]
    dest = "ubuntu@203.0.113.9:~/"

    cmd = orchestrate.build_scp_cmd(key_path, sources, dest, known_hosts_path)

    assert cmd == [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        "-o",
        "ConnectTimeout=30",
        "setup_node.sh",
        "run_profile.sh",
        "parse_nccl.py",
        "ubuntu@203.0.113.9:~/",
    ]
    assert "-r" not in cmd


def test_build_scp_cmd_recursive_flag_is_first_positional_after_scp(tmp_path):
    """recursive=True inserts -r immediately after the program name, before -i."""
    key_path = tmp_path / "keys" / "run.pem"
    known_hosts_path = tmp_path / ".state" / "known_hosts"

    cmd = orchestrate.build_scp_cmd(
        key_path,
        ["ubuntu@203.0.113.9:~/results_fc"],
        "/local/data/run/fc",
        known_hosts_path,
        recursive=True,
    )

    assert cmd[0] == "scp"
    assert cmd[1] == "-r"
    assert cmd[2] == "-i"
    assert cmd[-2] == "ubuntu@203.0.113.9:~/results_fc"
    assert cmd[-1] == "/local/data/run/fc"


def test_build_scp_cmd_rejects_empty_sources(tmp_path):
    with pytest.raises(ValueError):
        orchestrate.build_scp_cmd(tmp_path / "k.pem", [], "dest", tmp_path / "known_hosts")


# ---------------------------------------------------------------------------
# End-to-end orchestration (main()), everything monkeypatched
# ---------------------------------------------------------------------------


class _FakeBoto3:
    """Stand-in for the `boto3` module, only supplying `.client(...)`.

    Design: monkeypatched onto `orchestrate.boto3` specifically (not the
    real, globally-shared `boto3` module) so this fake never leaks into
    any other module's view of boto3. Returns a plain sentinel string
    rather than a real client, since every function orchestrate.py passes
    that "client" to (resolve_ami, ensure_key_pair, ...) is itself
    monkeypatched below and never actually calls a botocore method on it.
    """

    @staticmethod
    def client(service_name: str, region_name: str = None):
        return f"fake-{service_name}-client[{region_name}]"


@pytest.fixture
def orchestrate_fakes(tmp_path, monkeypatch):
    """Monkeypatch every AWS/network/subprocess seam orchestrate.py has.

    Returns
    -------
    dict
        ``{"calls": list of (name, args) tuples recording every fake
        invocation in order, "run_cmds": list of argv lists recorded by
        the fake subprocess.run, "teardown_calls": list of state dicts
        teardown_run was called with}``.

    Notes
    -----
    Also points ``cfg``'s ``--key-dir``/``--state-dir`` and
    ``orchestrate._DATA_DIR`` at ``tmp_path`` subdirectories (see
    ``orchestrate._DATA_DIR``'s module docstring comment for why that
    constant exists) so a full ``orchestrate.main(...)`` run in these
    tests writes real state/CSV files only under pytest's ephemeral
    ``tmp_path``, never into the real repository tree.
    """
    calls: List[tuple] = []
    run_cmds: List[List[str]] = []
    teardown_calls: List[Dict[str, Any]] = []

    monkeypatch.setattr(orchestrate, "boto3", _FakeBoto3)
    monkeypatch.setattr(orchestrate, "caller_ip", lambda: (_ for _ in ()).throw(
        AssertionError("caller_ip() should never be called when --ssh-cidr is passed")
    ))

    def fake_resolve_ami(ssm_client, parameter):
        calls.append(("resolve_ami", parameter))
        return "ami-fake0123456789"

    def fake_ensure_key_pair(ec2_client, key_name, key_dir):
        calls.append(("ensure_key_pair", key_name))
        key_dir.mkdir(parents=True, exist_ok=True)
        return key_dir / f"{key_name}.pem"

    def fake_ensure_security_group(ec2_client, group_name, ssh_cidr, tag_project, run_id):
        calls.append(("ensure_security_group", group_name, ssh_cidr))
        return "sg-fake0123456789"

    def fake_launch_instance(ec2_client, cfg, ami_id, sg_id, key_name, dry_run=False):
        calls.append(("launch_instance", dry_run))
        if dry_run:
            return {"instance_id": None, "purchasing_used": "ondemand"}
        return {"instance_id": "i-fake0123456789", "purchasing_used": "spot"}

    def fake_wait_for_instance(ec2_client, instance_id):
        # Ordering assertion baked into the fake itself (rather than only
        # checked after main() returns): the state file must already
        # exist, with this instance_id recorded, by the time
        # wait_for_instance is called -- this is the exact "state written
        # before the SSH wait" ordering the work-package spec requires.
        # See orchestrate.main's docstring "Design/WHY" note.
        state_path = Path(_last_state_dir[0]) / f"{_last_run_id[0]}.json"
        assert state_path.exists(), "state file must be written before wait_for_instance is called"
        written = json.loads(state_path.read_text())
        assert written["instance_id"] == instance_id
        assert written["public_ip"] is None
        calls.append(("wait_for_instance", instance_id))
        return "203.0.113.9"

    def fake_teardown_run(ec2_client, state, delete_key):
        teardown_calls.append(dict(state))
        calls.append(("teardown_run", state.get("instance_id"), delete_key))

    def fake_run(cmd, check=True, **kwargs):
        run_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    # _last_state_dir / _last_run_id let fake_wait_for_instance locate the
    # state file without needing main()'s local `cfg` in scope; populated
    # by the test itself right before calling orchestrate.main(...).
    _last_state_dir: List[Path] = [None]
    _last_run_id: List[str] = [None]

    monkeypatch.setattr(orchestrate, "resolve_ami", fake_resolve_ami)
    monkeypatch.setattr(orchestrate, "ensure_key_pair", fake_ensure_key_pair)
    monkeypatch.setattr(orchestrate, "ensure_security_group", fake_ensure_security_group)
    monkeypatch.setattr(orchestrate, "launch_instance", fake_launch_instance)
    monkeypatch.setattr(orchestrate, "wait_for_instance", fake_wait_for_instance)
    monkeypatch.setattr(orchestrate, "teardown_run", fake_teardown_run)
    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)

    data_dir = tmp_path / "data"
    monkeypatch.setattr(orchestrate, "_DATA_DIR", data_dir)

    return {
        "calls": calls,
        "run_cmds": run_cmds,
        "teardown_calls": teardown_calls,
        "state_dir_holder": _last_state_dir,
        "run_id_holder": _last_run_id,
        "data_dir": data_dir,
    }


def _base_argv(tmp_path, run_id: str, extra: List[str] = None) -> List[str]:
    """Shared CLI args for the end-to-end tests below.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's per-test temp directory; key-dir/state-dir are pointed
        here so no test ever touches the real correlation/keys or
        correlation/.state directories.
    run_id : str
        Deterministic run id so tests can locate the state file/data
        directory by name instead of discovering a generated one.
    extra : list[str] or None
        Additional argv to append (e.g. ``["--keep-alive"]``).

    Returns
    -------
    list[str]
        argv suitable for ``orchestrate.main(...)``. Always includes
        ``--ssh-cidr`` explicitly so ``caller_ip()`` (a real network call)
        is never reached, and ``--yes`` so no interactive prompt blocks
        the test.
    """
    argv = [
        "--yes",
        "--topology",
        "both",
        "--run-id",
        run_id,
        "--key-dir",
        str(tmp_path / "keys"),
        "--state-dir",
        str(tmp_path / "state"),
        "--ssh-cidr",
        "203.0.113.5/32",
        "--collectives",
        "all_reduce,alltoall",
        "--min-mib",
        "1",
        "--max-mib",
        "2",
        "--torus-dims",
        "2x2x2",
    ]
    if extra:
        argv += extra
    return argv


def test_main_end_to_end_happy_path(tmp_path, orchestrate_fakes, monkeypatch):
    """Full main() run: state ordering, per-leg profiling, fetch, and teardown.

    Asserts, per the work-package spec's test #3:
    - the state file is written after launch (before wait_for_instance is
      called -- enforced inside the fake_wait_for_instance itself) and
      contains instance_id,
    - the setup ssh command is executed with DEADMAN_MINUTES set,
    - exactly one run_profile.sh invocation per leg, with the correct
      dims ("8" then "2x2x2") and byte bounds (1 MiB / 2 MiB here),
    - one scp fetch per leg,
    - teardown_run is called exactly once, at the end,
    - the state file is removed afterward.
    """
    run_id = "test-run-e2e"
    state_dir = tmp_path / "state"
    orchestrate_fakes["state_dir_holder"][0] = state_dir
    orchestrate_fakes["run_id_holder"][0] = run_id

    exit_code = orchestrate.main(_base_argv(tmp_path, run_id))

    assert exit_code == 0

    # --- state file lifecycle -------------------------------------------------
    state_path = state_dir / f"{run_id}.json"
    assert not state_path.exists(), "state file must be removed after a successful teardown"

    # --- setup command ----------------------------------------------------------
    run_cmds = orchestrate_fakes["run_cmds"]
    setup_cmds = [c for c in run_cmds if "setup_node.sh" in c[-1]]
    assert len(setup_cmds) == 1
    assert "DEADMAN_MINUTES=120" in setup_cmds[0][-1]
    assert setup_cmds[0][0] == "ssh"

    # --- one run_profile.sh invocation per leg, correct dims/bytes -----------
    profile_cmds = [c for c in run_cmds if "run_profile.sh" in c[-1]]
    assert len(profile_cmds) == 2
    fc_cmd, torus_cmd = profile_cmds[0][-1], profile_cmds[1][-1]
    assert "results_fc fc 1048576 2097152 8 all_reduce alltoall" in fc_cmd
    assert "results_torus torus 1048576 2097152 2x2x2 all_reduce alltoall" in torus_cmd

    # --- fetch per leg ------------------------------------------------------
    # Recursive scp fetch argv shape (per build_scp_cmd): [..., source, dest],
    # so the source (a "user@ip:~/results_<leg>" string) is always the
    # second-to-last element.
    scp_fetch_cmds = [c for c in run_cmds if c[0] == "scp" and "-r" in c and "results_" in c[-2]]
    fetch_sources = {c[-2] for c in scp_fetch_cmds}
    assert any("results_fc" in s for s in fetch_sources)
    assert any("results_torus" in s for s in fetch_sources)

    # --- push commands happened before setup/profiling -----------------------
    # A push argv is a flat list of local file-path elements followed by a
    # remote dest string, so membership needs a substring scan across
    # elements rather than an exact-element containment check (the pushed
    # sources are full absolute paths, not the bare "setup_node.sh").
    def _any_elem_contains(cmd: List[str], needle: str) -> bool:
        return any(needle in elem for elem in cmd)

    push_cmds = [c for c in run_cmds if c[0] == "scp" and _any_elem_contains(c, "setup_node.sh")]
    assert len(push_cmds) == 1
    torus_push_cmds = [c for c in run_cmds if c[0] == "scp" and "-r" in c and "torus_bench" in c[-1]]
    assert len(torus_push_cmds) == 1

    # --- ordering: push -> setup -> (profile -> fetch) x legs ----------------
    def _first_index(predicate):
        return next(i for i, c in enumerate(run_cmds) if predicate(c))

    push_idx = _first_index(lambda c: c[0] == "scp" and _any_elem_contains(c, "setup_node.sh"))
    setup_idx = _first_index(lambda c: c[0] == "ssh" and "setup_node.sh" in c[-1])
    fc_profile_idx = _first_index(lambda c: c[0] == "ssh" and "run_profile.sh" in c[-1] and " fc " in c[-1])
    assert push_idx < setup_idx < fc_profile_idx

    # --- teardown called exactly once, at the very end -----------------------
    assert len(orchestrate_fakes["teardown_calls"]) == 1
    assert orchestrate_fakes["teardown_calls"][0]["instance_id"] == "i-fake0123456789"
    call_names = [c[0] for c in orchestrate_fakes["calls"]]
    assert call_names[-1] == "teardown_run"


# ---------------------------------------------------------------------------
# Fix 1 (BLOCKER): _run_leg must create the fetch's PARENT directory (not the
# leaf) before scp runs, and must refuse to fetch into an already-existing
# leaf directory.
# ---------------------------------------------------------------------------


def test_main_creates_run_data_dir_before_fetch_scp(tmp_path, orchestrate_fakes, monkeypatch):
    """run_data_dir (the fetch's parent) exists by the time each leg's fetch scp runs.

    Regression test for Fix 1: without ``run_data_dir.mkdir(...)`` before the
    fetch, `scp -r user@host:~/results_fc <run_data_dir>/fc` would fail
    outright the first time a run_id is used, since its parent directory
    would not exist yet. This test captures os-path existence from *inside*
    the fake ``subprocess.run`` at the exact moment a fetch command is
    recorded, per the work-package spec's test #1.
    """
    run_id = "test-run-mkdir"
    state_dir = tmp_path / "state"
    orchestrate_fakes["state_dir_holder"][0] = state_dir
    orchestrate_fakes["run_id_holder"][0] = run_id
    data_dir = orchestrate_fakes["data_dir"]

    run_data_dir_existed_at_fetch: List[bool] = []

    def fake_run(cmd, check=True, **kwargs):
        orchestrate_fakes["run_cmds"].append(list(cmd))
        # A fetch is a recursive scp whose source (second-to-last argv
        # element, per build_scp_cmd's argv shape) names a remote
        # results_<leg> path -- distinguishes it from the (also recursive)
        # torus_bench/ push, whose source is a local path instead.
        if cmd[0] == "scp" and "-r" in cmd and "results_" in cmd[-2]:
            run_data_dir_existed_at_fetch.append((data_dir / run_id).exists())
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)

    exit_code = orchestrate.main(_base_argv(tmp_path, run_id))

    assert exit_code == 0
    # One fetch per leg (fc, torus); run_data_dir must already exist at both.
    assert len(run_data_dir_existed_at_fetch) == 2
    assert all(run_data_dir_existed_at_fetch)


def test_run_leg_raises_if_leaf_dir_already_exists(tmp_path, monkeypatch):
    """_run_leg refuses to re-fetch into an already-existing run_data_dir/leg.

    Direct unit test of :func:`orchestrate._run_leg` (rather than a full
    ``main()`` run) per the work-package spec's test #2, exercising the
    RuntimeError in isolation. ``_run_streaming`` is monkeypatched to a
    recording no-op so no real ssh/scp subprocess is ever attempted; the
    remote profiling ssh command runs (it happens before the leaf-dir check
    in ``_run_leg``'s body), but the fetch scp must never be reached.
    """
    run_cmds: List[List[str]] = []
    monkeypatch.setattr(orchestrate, "_run_streaming", lambda cmd: run_cmds.append(cmd))

    cfg = Config(run_id="test-run-leaf-exists", torus_dims=(2, 2, 2), collectives="all_reduce")
    run_data_dir = tmp_path / "data" / cfg.run_id
    local_leg_dir = run_data_dir / "fc"
    local_leg_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="already exists"):
        orchestrate._run_leg(
            cfg,
            "fc",
            tmp_path / "key.pem",
            "203.0.113.9",
            tmp_path / "known_hosts",
            run_data_dir,
        )

    # Only the remote profiling ssh command (which precedes the leaf-dir
    # check in _run_leg's body) ran; the fetch scp was never attempted.
    assert len(run_cmds) == 1
    assert run_cmds[0][0] == "ssh"


def test_main_dry_run_stops_before_launch_and_ssh(tmp_path, orchestrate_fakes):
    """--dry-run creates key+SG+a dry launch, but never waits/pushes/profiles/tears down."""
    run_id = "test-run-dry"
    state_dir = tmp_path / "state"
    orchestrate_fakes["state_dir_holder"][0] = state_dir
    orchestrate_fakes["run_id_holder"][0] = run_id

    exit_code = orchestrate.main(_base_argv(tmp_path, run_id, extra=["--dry-run"]))

    assert exit_code == 0
    call_names = [c[0] for c in orchestrate_fakes["calls"]]
    assert "ensure_key_pair" in call_names
    assert "ensure_security_group" in call_names
    assert "launch_instance" in call_names
    assert "wait_for_instance" not in call_names
    assert "teardown_run" not in call_names
    assert orchestrate_fakes["run_cmds"] == []
    assert not (state_dir / f"{run_id}.json").exists()


def test_main_profiling_failure_still_tears_down_and_propagates(tmp_path, orchestrate_fakes, monkeypatch):
    """A failed profiling subprocess still triggers teardown, and the error propagates."""
    run_id = "test-run-fail"
    state_dir = tmp_path / "state"
    orchestrate_fakes["state_dir_holder"][0] = state_dir
    orchestrate_fakes["run_id_holder"][0] = run_id

    def failing_run(cmd, check=True, **kwargs):
        orchestrate_fakes["run_cmds"].append(list(cmd))
        if "run_profile.sh" in cmd[-1]:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(orchestrate.subprocess, "run", failing_run)

    with pytest.raises(subprocess.CalledProcessError):
        orchestrate.main(_base_argv(tmp_path, run_id))

    # Teardown must still have run despite the propagating exception.
    assert len(orchestrate_fakes["teardown_calls"]) == 1
    assert orchestrate_fakes["teardown_calls"][0]["instance_id"] == "i-fake0123456789"
    # And the state file was still cleaned up by that successful teardown.
    assert not (state_dir / f"{run_id}.json").exists()


def test_main_teardown_failure_does_not_mask_original_exception(tmp_path, orchestrate_fakes, monkeypatch):
    """If teardown_run ALSO raises, the original profiling exception still propagates."""
    run_id = "test-run-double-fail"
    state_dir = tmp_path / "state"
    orchestrate_fakes["state_dir_holder"][0] = state_dir
    orchestrate_fakes["run_id_holder"][0] = run_id

    def failing_run(cmd, check=True, **kwargs):
        orchestrate_fakes["run_cmds"].append(list(cmd))
        if "run_profile.sh" in cmd[-1]:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0)

    def failing_teardown(ec2_client, state, delete_key):
        orchestrate_fakes["teardown_calls"].append(dict(state))
        raise RuntimeError("simulated teardown failure (e.g. DependencyViolation exhausted)")

    monkeypatch.setattr(orchestrate.subprocess, "run", failing_run)
    monkeypatch.setattr(orchestrate, "teardown_run", failing_teardown)

    # The ORIGINAL exception (CalledProcessError from profiling) must be
    # what propagates, not the teardown's RuntimeError -- this is the
    # "don't mask the original exception" behavior _teardown_and_cleanup
    # documents.
    with pytest.raises(subprocess.CalledProcessError):
        orchestrate.main(_base_argv(tmp_path, run_id))

    assert len(orchestrate_fakes["teardown_calls"]) == 1
    # Teardown failed, so the state file must NOT have been removed --
    # teardown.py --run-id needs it to find leftover resources later.
    assert (state_dir / f"{run_id}.json").exists()


def test_main_keep_alive_skips_teardown_and_keeps_state_file(tmp_path, orchestrate_fakes, capsys):
    """--keep-alive leaves the instance up: no teardown_run call, state file remains."""
    run_id = "test-run-keep-alive"
    state_dir = tmp_path / "state"
    orchestrate_fakes["state_dir_holder"][0] = state_dir
    orchestrate_fakes["run_id_holder"][0] = run_id

    exit_code = orchestrate.main(_base_argv(tmp_path, run_id, extra=["--keep-alive"]))

    assert exit_code == 0
    assert orchestrate_fakes["teardown_calls"] == []
    call_names = [c[0] for c in orchestrate_fakes["calls"]]
    assert "teardown_run" not in call_names

    state_path = state_dir / f"{run_id}.json"
    assert state_path.exists()
    written = json.loads(state_path.read_text())
    assert written["public_ip"] == "203.0.113.9"

    out = capsys.readouterr().out
    assert "STILL BEING BILLED" in out
    assert "ssh -i" in out
