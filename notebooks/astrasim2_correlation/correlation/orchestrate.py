"""End-to-end CLI for the NCCL correlation study's empirical leg.

This is the "run everything" work package: it ties together the
provisioning infrastructure (``config.py``, ``provision.py``,
``teardown.py``) and the profiling infrastructure (``setup_node.sh``,
``run_profile.sh``, ``parse_nccl.py``, ``torus_bench/``) -- all owned by
sibling work packages and imported/invoked here, never modified -- into one
command an operator runs to go from "nothing provisioned" to "CSVs sitting
in ``data/<run_id>/<leg>/csv/`` and the instance torn down".

Pipeline
--------
1. Parse CLI args into a :class:`config.Config` plus this script's own
   ``--yes``/``--keep-alive``/``--dry-run``/``--ssh-cidr`` flags.
2. Print the run header, cost warning, a wall-time estimate, and the leg
   plan (via :func:`legs_for`); ask for interactive confirmation unless
   ``--yes`` (skipped entirely for ``--dry-run``, which spends no money).
3. Resolve the AMI, create the SSH key pair and security group, and call
   ``provision.launch_instance`` -- exactly the sequence ``provision.py``
   itself runs, reusing its functions directly rather than reimplementing
   any of them. If ``--dry-run``, stop here (the key pair and security
   group above were still created for real; see the ``--dry-run`` flag's
   help text).
4. Write the provisioning state file immediately, before waiting for SSH
   (see the design comment at that call site for why).
5. Wait for the instance to become SSH-reachable, then update the state
   file with the now-known public IP.
6. scp the profiling scripts and ``torus_bench/`` onto the instance.
7. ssh in to run ``setup_node.sh`` (arms the dead-man timer, builds
   nccl-tests/torus_bench).
8. For each leg selected by ``--topology`` (see :func:`legs_for`), ssh in
   to run ``run_profile.sh`` for the full collective sweep, then scp the
   results back to ``data/<run_id>/<leg>/``.
9. In a ``finally`` block around steps 5-8: tear the instance down (unless
   ``--keep-alive``), so a crash or a failed profiling leg never leaves an
   (expensive, 8x H100) instance running unattended. See the design
   comment on :func:`_teardown_and_cleanup` for how a teardown failure
   itself is handled without masking whatever exception was already
   propagating.

Design: no new AWS/SSH logic here
-----------------------------------
Every AWS API call in this module goes through a ``provision.py`` or
``teardown.py`` function that already exists, is already tested, and is
already documented as part of this work package's contract (see those
modules' docstrings). This module's own responsibility is narrower:
sequencing those calls correctly, building ``ssh``/``scp`` argv lists
(:func:`build_ssh_cmd`, :func:`build_scp_cmd`), and running them as
subprocesses. Keeping that boundary sharp is also what makes this module
testable without any real AWS/SSH/SCP access -- every seam it introduces
(the two ``build_*_cmd`` functions, plus the imported provisioning
functions) is a plain function that a test can monkeypatch or inspect the
return value of, per this work package's "no AWS, no network, no ssh in
tests" constraint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from provision import (
    _COST_WARNING,
    _prompt_yes_no,
    _require_boto3,
    caller_ip,
    ensure_key_pair,
    ensure_security_group,
    launch_instance,
    resolve_ami,
    wait_for_instance,
    write_state,
)
from teardown import teardown_run

# Design: guarded import, matching provision.py/teardown.py's own
# convention (see provision.py's module docstring for the full rationale)
# -- so `python orchestrate.py --help` keeps working even in a Python
# environment that lacks boto3, since argparse's own --help handling exits
# before main() ever reaches _require_boto3(). Importing `boto3` here as a
# module-level name of orchestrate.py's own (rather than reaching into
# `provision.boto3`) keeps this module's boto3.client(...) calls readable
# without poking at another module's internals; _require_boto3() (reused
# from provision.py, not redefined) is still what actually validates
# availability before any client is constructed, since both imports
# resolve to the same cached sys.modules entry (or both to None) in any
# given interpreter.
try:
    import boto3
except ImportError:  # pragma: no cover - exercised only when boto3 truly absent
    boto3 = None

# ---------------------------------------------------------------------------
# Module-level path constants
# ---------------------------------------------------------------------------

# Anchor every sibling-file path to this file's own directory (not the
# process cwd), matching config.py's identical _THIS_DIR convention -- so
# `python orchestrate.py` behaves the same regardless of the caller's shell
# cwd.
_THIS_DIR = Path(__file__).resolve().parent

_SETUP_NODE_SH = _THIS_DIR / "setup_node.sh"
_RUN_PROFILE_SH = _THIS_DIR / "run_profile.sh"
_PARSE_NCCL_PY = _THIS_DIR / "parse_nccl.py"
_TORUS_BENCH_DIR = _THIS_DIR / "torus_bench"

# Design: expose the fetched-results root as its own module-level constant
# (rather than inlining `_THIS_DIR / "data"` at the one call site) purely
# so tests can monkeypatch `orchestrate._DATA_DIR` to a tmp_path and
# guarantee the end-to-end test never writes into the real repository's
# data/ directory -- every other AWS/ssh/scp side effect in a test is
# already monkeypatched away, and this is the one remaining plain
# filesystem write main() would otherwise perform unconditionally.
_DATA_DIR = _THIS_DIR / "data"

# run_profile.sh's <dims> argument for the FC leg: the profiling scripts
# treat "8" as "all 8 GPUs, fully connected" -- there is no logical torus
# shape to describe for that leg (see run_profile.sh's `-g 8` on the FC
# path), unlike the torus leg where <dims> is a "DxDx..." shape string.
_FC_DIMS = "8"

_BYTES_PER_MIB = 2**20


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, no AWS, no subprocess) -- kept separate from main()
# specifically so they are trivially unit-testable per this work package's
# "no AWS, no network, no ssh in tests" constraint.
# ---------------------------------------------------------------------------


def legs_for(topology: str) -> List[str]:
    """Expand a ``Config.topology`` value into the ordered list of legs to run.

    Parameters
    ----------
    topology : str
        Typically ``cfg.topology``, one of ``"fc"``, ``"torus"``, or
        ``"both"`` (``Config.__post_init__`` already validates this, so
        this function does not re-validate it -- see Notes).

    Returns
    -------
    list[str]
        ``["fc", "torus"]`` if ``topology == "both"``; otherwise the
        single-element list ``[topology]``.

    Notes
    -----
    Deliberately permissive for any value other than ``"both"``: it simply
    echoes that value back as a one-element list rather than checking it
    against the allowed set. Re-validating here would duplicate
    ``Config.__post_init__``'s already-authoritative check for no benefit,
    since every caller in this module only ever passes an already-validated
    ``cfg.topology``.

    Examples
    --------
    >>> legs_for("both")
    ['fc', 'torus']
    >>> legs_for("fc")
    ['fc']
    >>> legs_for("torus")
    ['torus']
    """
    if topology == "both":
        return ["fc", "torus"]
    return [topology]


def _dims_for_leg(leg: str, torus_dims: Tuple[int, ...]) -> str:
    """Compute run_profile.sh's ``<dims>`` argument for one leg.

    Parameters
    ----------
    leg : str
        ``"fc"`` or ``"torus"``.
    torus_dims : tuple[int, ...]
        Per-axis torus dimensions, e.g. ``(2, 2, 2)`` (only consulted when
        ``leg == "torus"``).

    Returns
    -------
    str
        :data:`_FC_DIMS` (``"8"``) for the FC leg; otherwise
        ``torus_dims`` joined with ``"x"``, e.g. ``"2x2x2"``.

    Examples
    --------
    >>> _dims_for_leg("fc", (2, 2, 2))
    '8'
    >>> _dims_for_leg("torus", (2, 2, 2))
    '2x2x2'
    """
    if leg == "fc":
        return _FC_DIMS
    return "x".join(str(d) for d in torus_dims)


def _estimate_wall_time_message(legs: List[str]) -> str:
    """Build the human-readable wall-time estimate printed before confirmation.

    Parameters
    ----------
    legs : list[str]
        The leg plan, as returned by :func:`legs_for`.

    Returns
    -------
    str
        A one-line estimate: ~30-60 minutes when both legs are selected
        (the work-package spec's own figure for a full FC+torus sweep),
        or roughly half that for a single leg.

    Notes
    -----
    This is a coarse, documented *estimate* for setting operator
    expectations before a real-money confirmation prompt, not a measured
    or SLA'd figure -- actual time depends on instance boot time, spot
    availability, and the exact collective/message-size sweep configured.
    """
    if len(legs) >= 2:
        return (
            "Estimated wall time: ~30-60 minutes for both legs "
            "(excludes instance boot/provisioning time)."
        )
    return (
        "Estimated wall time: ~15-30 minutes for a single leg "
        "(excludes instance boot/provisioning time)."
    )


def build_ssh_cmd(
    key_path: Path,
    user: str,
    ip: str,
    remote_cmd: str,
    known_hosts_path: Path,
) -> List[str]:
    """Build an ``ssh`` argv list to run one command on the provisioned instance.

    Parameters
    ----------
    key_path : pathlib.Path
        Path to the local PEM private key (as returned by
        ``provision.ensure_key_pair``).
    user : str
        SSH login user, typically ``cfg.ssh_user``.
    ip : str
        Target host's public IP address.
    remote_cmd : str
        The full remote command line to execute, e.g.
        ``"DEADMAN_MINUTES=120 bash ~/setup_node.sh"``. Passed to ``ssh``
        as a single trailing argv element; ``ssh`` hands it to the remote
        login shell for interpretation, so ordinary shell syntax (env var
        prefixes, multiple space-separated arguments) works as expected
        without any extra quoting from this function.
    known_hosts_path : pathlib.Path
        Path to a per-run known-hosts file (typically
        ``cfg.state_dir / "known_hosts"``). Kept as an explicit parameter
        (rather than a hardcoded/global path) so this function stays pure
        and independently testable -- see the module docstring's "Design:
        no new AWS/SSH logic here" section.

        .. note::
           NOTE ON SPEC DEVIATION: the work-package spec lists this
           function's signature as ``build_ssh_cmd(key_path, user, ip,
           remote_cmd)`` -- four parameters -- but also requires the
           ``-o UserKnownHostsFile=<state_dir>/known_hosts`` option, which
           cannot be constructed without knowing ``state_dir`` from
           *somewhere*. Reaching into a module-global ``Config`` from
           inside this function would break the "pure helper, easy to
           unit-test" property the spec explicitly asks for. Adding
           ``known_hosts_path`` as an explicit fifth parameter is the
           minimal change that preserves purity/testability; every
           call site in this module passes ``cfg.state_dir / "known_hosts"``
           for it, matching the spec's intent.

    Returns
    -------
    list[str]
        argv list of the form ``["ssh", "-i", <key>, "-o",
        "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=<...>",
        "-o", "ConnectTimeout=30", "<user>@<ip>", <remote_cmd>]``, ready to
        pass to ``subprocess.run``.

    Examples
    --------
    >>> build_ssh_cmd(Path("/k/id.pem"), "ubuntu", "1.2.3.4", "echo hi", Path("/s/known_hosts"))
    ... # doctest: +NORMALIZE_WHITESPACE
    ['ssh', '-i', '/k/id.pem', '-o', 'StrictHostKeyChecking=accept-new',
     '-o', 'UserKnownHostsFile=/s/known_hosts', '-o', 'ConnectTimeout=30',
     'ubuntu@1.2.3.4', 'echo hi']
    """
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        "-o",
        "ConnectTimeout=30",
        f"{user}@{ip}",
        remote_cmd,
    ]


def build_scp_cmd(
    key_path: Path,
    sources: List[str],
    dest: str,
    known_hosts_path: Path,
    recursive: bool = False,
) -> List[str]:
    """Build an ``scp`` argv list to copy one or more paths to/from the instance.

    Parameters
    ----------
    key_path : pathlib.Path
        Path to the local PEM private key.
    sources : list[str]
        Source path(s) to copy, in whatever form ``scp`` accepts: plain
        local paths for a push, or a single ``"user@ip:remote/path"``
        string for a fetch. Must be non-empty.
    dest : str
        Destination, again in whatever form ``scp`` accepts (a local
        directory for a fetch, or ``"user@ip:remote/path"`` for a push).
    known_hosts_path : pathlib.Path
        Path to a per-run known-hosts file. See :func:`build_ssh_cmd`'s
        docstring for why this is an explicit parameter rather than an
        implicit global (the same rationale applies here).
    recursive : bool, default False
        If ``True``, prepend ``-r`` (needed for copying a directory, e.g.
        ``torus_bench/`` or a remote ``results_<leg>/`` directory).

    Returns
    -------
    list[str]
        argv list of the form ``["scp", ["-r"], "-i", <key>, "-o",
        "StrictHostKeyChecking=accept-new", "-o",
        "UserKnownHostsFile=<...>", "-o", "ConnectTimeout=30", *sources,
        dest]``, ready to pass to ``subprocess.run``. The ``-r`` flag (when
        present) is placed immediately after ``"scp"``, before every other
        option, so its position is fixed and independently assertable in
        tests regardless of how many sources are given.

    Raises
    ------
    ValueError
        If ``sources`` is empty -- an ``scp`` invocation with no source
        path is never meaningful and would otherwise fail cryptically at
        the OS level instead of at this argv-building step.

    Examples
    --------
    >>> build_scp_cmd(Path("/k/id.pem"), ["a.sh", "b.sh"], "ubuntu@1.2.3.4:~/", Path("/s/known_hosts"))
    ... # doctest: +NORMALIZE_WHITESPACE
    ['scp', '-i', '/k/id.pem', '-o', 'StrictHostKeyChecking=accept-new',
     '-o', 'UserKnownHostsFile=/s/known_hosts', '-o', 'ConnectTimeout=30',
     'a.sh', 'b.sh', 'ubuntu@1.2.3.4:~/']
    >>> build_scp_cmd(Path("/k/id.pem"), ["dir"], "ubuntu@1.2.3.4:~/dir", Path("/s/known_hosts"), recursive=True)[:2]
    ['scp', '-r']
    """
    if not sources:
        raise ValueError("build_scp_cmd requires at least one source path")

    cmd: List[str] = ["scp"]
    if recursive:
        cmd.append("-r")
    cmd += [
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        "-o",
        "ConnectTimeout=30",
    ]
    cmd += list(sources)
    cmd.append(dest)
    return cmd


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _run_streaming(cmd: List[str]) -> None:
    """Run an ``ssh``/``scp`` argv list, streaming its output live.

    Parameters
    ----------
    cmd : list[str]
        argv list, typically from :func:`build_ssh_cmd` or
        :func:`build_scp_cmd`.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess exits with a non-zero status (``check=True``).

    Notes
    -----
    Deliberately does not pass ``capture_output``/``stdout``/``stderr`` --
    the child process's output streams straight through to this process's
    own stdout/stderr, so an operator watching a multi-minute profiling
    sweep sees live progress rather than a silent hang followed by a wall
    of buffered text at the end. Prints the command line first (to stdout)
    so the corresponding output block is identifiable in a long combined
    log.
    """
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Pipeline stages -- each a thin, single-responsibility wrapper around a
# handful of _run_streaming calls, factored out of main() so that function
# stays a readable top-to-bottom sequence rather than one long body.
# ---------------------------------------------------------------------------


def _push_files(cfg: Config, key_path: Path, public_ip: str, known_hosts_path: Path) -> None:
    """scp the profiling scripts and torus_bench/ onto the instance.

    Parameters
    ----------
    cfg : Config
        Run configuration; only ``cfg.ssh_user`` is consulted directly
        (the rest flows through ``key_path``/``public_ip``/
        ``known_hosts_path``).
    key_path : pathlib.Path
        Local PEM private key path.
    public_ip : str
        Instance's public IP, as returned by ``provision.wait_for_instance``.
    known_hosts_path : pathlib.Path
        Per-run known-hosts file path.

    Notes
    -----
    Two separate ``scp`` invocations, matching the two different transfer
    shapes: (1) three individual files pushed non-recursively to the
    remote home directory, and (2) the ``torus_bench/`` directory pushed
    recursively to ``~/torus_bench`` so ``setup_node.sh`` can build it.
    Side effect: two subprocess invocations over the network to the
    instance.
    """
    remote_home = f"{cfg.ssh_user}@{public_ip}:~/"
    print("=== Pushing profiling scripts (setup_node.sh, run_profile.sh, parse_nccl.py) ===")
    _run_streaming(
        build_scp_cmd(
            key_path,
            [str(_SETUP_NODE_SH), str(_RUN_PROFILE_SH), str(_PARSE_NCCL_PY)],
            remote_home,
            known_hosts_path,
        )
    )

    remote_torus_dir = f"{cfg.ssh_user}@{public_ip}:~/torus_bench"
    print("=== Pushing torus_bench/ ===")
    _run_streaming(
        build_scp_cmd(
            key_path,
            [str(_TORUS_BENCH_DIR)],
            remote_torus_dir,
            known_hosts_path,
            recursive=True,
        )
    )


def _run_setup(cfg: Config, key_path: Path, public_ip: str, known_hosts_path: Path) -> None:
    """ssh in and run ``setup_node.sh`` (arms dead-man timer, builds binaries).

    Parameters
    ----------
    cfg : Config
        Run configuration; ``cfg.dead_man_minutes`` is forwarded to
        ``setup_node.sh`` as the ``DEADMAN_MINUTES`` environment variable.
    key_path : pathlib.Path
        Local PEM private key path.
    public_ip : str
        Instance's public IP.
    known_hosts_path : pathlib.Path
        Per-run known-hosts file path.

    Notes
    -----
    Must run after :func:`_push_files` (``setup_node.sh`` and
    ``torus_bench/`` must already be on the instance) and before any
    profiling leg (``setup_node.sh`` is what builds the nccl-tests and
    torus_bench binaries those legs depend on, and arms the on-instance
    dead-man safety timer). Side effect: one subprocess invocation over
    the network to the instance.
    """
    remote_cmd = f"DEADMAN_MINUTES={cfg.dead_man_minutes} bash ~/setup_node.sh"
    print("=== Running setup_node.sh ===")
    _run_streaming(build_ssh_cmd(key_path, cfg.ssh_user, public_ip, remote_cmd, known_hosts_path))


def _run_leg(
    cfg: Config,
    leg: str,
    key_path: Path,
    public_ip: str,
    known_hosts_path: Path,
    run_data_dir: Path,
) -> None:
    """Run one profiling leg on the instance and fetch its results locally.

    Parameters
    ----------
    cfg : Config
        Run configuration (``torus_dims``, ``min_mib``/``max_mib``,
        ``collectives``, ``ssh_user``).
    leg : str
        ``"fc"`` or ``"torus"``.
    key_path : pathlib.Path
        Local PEM private key path.
    public_ip : str
        Instance's public IP.
    known_hosts_path : pathlib.Path
        Per-run known-hosts file path.
    run_data_dir : Path
        Local directory this run's results should land under (typically
        ``_DATA_DIR / cfg.run_id``); this leg's results land specifically
        at ``run_data_dir / leg``.

    Notes
    -----
    Side effects: one ``ssh`` subprocess invocation that runs the full
    collective sweep for this leg on the instance (this is the
    long-running step -- see :func:`_estimate_wall_time_message`), a
    ``mkdir`` of ``run_data_dir`` (the PARENT of this leg's fetch
    destination -- see the inline comment above that call for why the leaf
    directory itself is deliberately left uncreated), and one ``scp -r``
    subprocess invocation that fetches the results back, creating the new
    local directory tree at ``run_data_dir / leg`` itself.

    Prints the list of fetched ``*.csv`` files at the end, so an operator
    watching the run can immediately confirm data landed without a
    separate ``ls``.

    Raises
    ------
    RuntimeError
        If ``run_data_dir / leg`` already exists before the fetch -- see
        the inline comment above the check for why this is treated as an
        error rather than silently proceeding.
    """
    dims = _dims_for_leg(leg, cfg.torus_dims)
    min_bytes = cfg.min_mib * _BYTES_PER_MIB
    max_bytes = cfg.max_mib * _BYTES_PER_MIB
    remote_results_dir = f"~/results_{leg}"
    remote_cmd = (
        f"bash ~/run_profile.sh {remote_results_dir} {leg} {min_bytes} {max_bytes} {dims} "
        f"{' '.join(cfg.collectives)}"
    )

    print(f"=== Profiling leg: {leg} (dims={dims}, {cfg.min_mib}-{cfg.max_mib} MiB) ===")
    _run_streaming(build_ssh_cmd(key_path, cfg.ssh_user, public_ip, remote_cmd, known_hosts_path))

    # Design: fetch into a leaf directory (run_data_dir/leg) that must NOT
    # already exist when scp runs. `scp -r user@host:~/results_fc
    # <local_dest>` renames the copied directory to <local_dest> when
    # <local_dest> doesn't yet exist, landing its contents (csv/, raw/,
    # metadata.txt) directly inside it -- exactly the data/<run_id>/<leg>/
    # layout this work package's spec requires. Pre-creating that leaf
    # directory first would instead make scp nest an extra results_fc/
    # level inside it, since scp's "copy into vs. rename to" behavior
    # depends on whether the destination path already exists.
    local_leg_dir = run_data_dir / leg
    if local_leg_dir.exists():
        # A pre-existing leaf directory means a previous run already fetched
        # results for this exact run_id+leg combination (or something else
        # created the path). Silently proceeding would make scp nest an
        # extra results_<leg>/ level inside the existing directory instead
        # of landing csv/raw/metadata.txt directly in it (see the comment
        # above), corrupting the data/<run_id>/<leg>/ layout without any
        # error -- raise instead so a re-run with a colliding run_id+leg
        # fails loudly here rather than silently mis-nesting fetched data.
        raise RuntimeError(
            f"{local_leg_dir} already exists; a previous fetch for run_id="
            f"{cfg.run_id!r} leg={leg!r} already landed results there. "
            "Refusing to scp into it again (that would nest an extra "
            f"results_{leg}/ level inside the existing directory instead of "
            "csv/raw/metadata.txt landing directly in it). Remove or rename "
            "the existing directory, or re-run with a different --run-id."
        )
    # Only the PARENT (run_data_dir) is created here -- local_leg_dir itself
    # must be left absent (see the comment above) for scp's rename-vs-nest
    # behavior to produce the right layout. Without this mkdir, the very
    # first leg fetched under a fresh run_id would fail outright: scp cannot
    # write to <run_data_dir>/<leg> if <run_data_dir> itself doesn't exist
    # yet (parents=True/exist_ok=True mirrors provision.write_state's own
    # "create the directory, then write into it" idiom).
    run_data_dir.mkdir(parents=True, exist_ok=True)
    fetch_source = f"{cfg.ssh_user}@{public_ip}:{remote_results_dir}"
    print(f"=== Fetching {leg} leg results ===")
    _run_streaming(
        build_scp_cmd(
            key_path,
            [fetch_source],
            str(local_leg_dir),
            known_hosts_path,
            recursive=True,
        )
    )

    csv_files = sorted((local_leg_dir / "csv").glob("*.csv"))
    print(f"Fetched {leg} leg results to {local_leg_dir}:")
    for csv_path in csv_files:
        print(f"  {csv_path}")


def _print_keep_alive_notice(cfg: Config, key_path: Path, state: Dict[str, Any]) -> None:
    """Print the SSH command and a loud cost reminder for a ``--keep-alive`` run.

    Parameters
    ----------
    cfg : Config
        Run configuration (``ssh_user``, ``run_id``, ``dead_man_minutes``).
    key_path : pathlib.Path
        Local PEM private key path, included in the printed SSH command.
    state : dict
        This run's state dict; ``state["public_ip"]`` is used if present.

    Notes
    -----
    Called from :func:`main`'s ``finally`` block in place of
    :func:`_teardown_and_cleanup` when ``--keep-alive`` was passed. Does
    not delete the state file (unlike the teardown path), since
    ``teardown.py`` needs it later to find and tear down this
    still-running instance.
    """
    ip_display = state.get("public_ip") or "<unknown -- see state file>"
    print("=" * 70)
    print("--keep-alive set: instance left RUNNING. YOU ARE STILL BEING BILLED.")
    print(f"SSH command: ssh -i {key_path} {cfg.ssh_user}@{ip_display}")
    print(
        f"Dead-man shutdown deadline: ~{cfg.dead_man_minutes} minutes after "
        "setup_node.sh armed it (this is a backstop independent of teardown.py)."
    )
    print(f"Tear down manually with: python teardown.py --run-id {cfg.run_id}")
    print("=" * 70)


def _teardown_and_cleanup(
    ec2_client, state: Dict[str, Any], state_path: Path, cfg: Config
) -> None:
    """Tear down this run's AWS resources and remove its local state file.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    state : dict
        This run's state dict, as passed to ``teardown.teardown_run``.
    state_path : pathlib.Path
        Path to this run's ``.state/<run_id>.json`` file, removed only if
        teardown succeeds.
    cfg : Config
        Run configuration, used only to print ``cfg.run_id`` into the
        follow-up messages below.

    Notes
    -----
    Called from :func:`main`'s ``finally`` block, which may itself be
    running while an exception from the profiling steps (e.g. a failed
    ``run_profile.sh`` invocation) is already propagating out of the
    surrounding ``try``. This is exactly the scenario the work-package
    spec calls out: a ``finally`` block that itself raises would, under
    Python's ordinary exception semantics, cause *that new* exception to
    be what the caller sees instead of the original one -- effectively
    masking the original failure behind a teardown failure, even though
    both are independently worth surfacing. This function therefore
    catches any exception ``teardown_run`` raises, prints it (and does not
    delete the state file, since a failed teardown may have left resources
    behind that ``teardown.py --run-id <run_id>`` will still need it to
    find), and simply returns -- letting whatever exception was already
    active in the caller's ``try`` block continue propagating undisturbed.
    """
    print(f"Tearing down run {cfg.run_id}...")
    try:
        teardown_run(ec2_client, state, delete_key=False)
    except Exception as teardown_exc:  # noqa: BLE001
        # Design/WHY deliberately broad and deliberately NOT re-raised:
        # see this function's docstring Notes above. Printing to stderr
        # (rather than raising) is what prevents this teardown failure
        # from masking an in-flight exception from the try block this
        # finally belongs to.
        print(f"ERROR: teardown failed: {teardown_exc}", file=sys.stderr)
        print(
            "Manual cleanup required -- the instance may still be running. "
            f"Try: python teardown.py --run-id {cfg.run_id}",
            file=sys.stderr,
        )
        return

    if state_path.exists():
        state_path.unlink()
        print(f"Removed state file {state_path}.")
    print("Recommended audit: python teardown.py --verify")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI args and run the full provision -> profile -> teardown pipeline.

    Parameters
    ----------
    argv : list[str] or None, default None
        Argument list, as passed to ``argparse``'s ``parse_args``. ``None``
        reads from ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success (including a completed ``--dry-run``), ``1`` if
        the user declined the interactive confirmation prompt.

    Raises
    ------
    Exception
        Any exception raised by provisioning, SSH/SCP subprocess failures
        (``subprocess.CalledProcessError``), or waiting for the instance
        propagates out of this function uncaught -- per the work-package
        spec, a profiling/provisioning failure is a real failure and
        should surface as one (a nonzero process exit code via
        ``raise SystemExit(main())`` in ``__main__``), not be silently
        downgraded to a return code. The ``finally`` block described below
        still runs before that propagation completes.

    Notes
    -----
    Side effects (skipped when ``--dry-run`` is passed, past the point
    where the SSH key pair and security group are created -- see the
    ``--dry-run`` flag's help text): creates an SSH key pair and security
    group, launches an instance, writes/updates a state JSON file, scp's
    profiling scripts and results to/from the instance, ssh's in to run
    setup and profiling commands, and (unless ``--keep-alive``) tears the
    instance down again.

    Design/WHY the state file is written immediately after
    ``launch_instance`` returns, before ``wait_for_instance`` is even
    called: ``wait_for_instance`` can block for several minutes (instance
    boot, then polling for SSH) and can itself raise (``TimeoutError``,
    ``WaiterError``). If this controller process crashes or is killed
    during that wait, an instance is still running and being billed with
    no local record of it unless the state file was already written
    beforehand. Writing state right after launch -- with ``public_ip`` as
    an explicit placeholder, filled in by a second ``write_state`` call
    once ``wait_for_instance`` returns -- means ``teardown.py`` (which
    also cross-checks live AWS tags, not just local state, per its own
    docstring) can find and tear down this run from local state alone even
    in that crash scenario, without waiting on SSH reachability first.
    """
    parser = argparse.ArgumentParser(
        prog="orchestrate.py",
        description=(
            "End-to-end NCCL correlation-study orchestration: provision one "
            "p5.48xlarge instance, push profiling scripts, run the FC "
            "and/or torus profiling sweep, fetch results into data/<run_id>/, "
            "and tear the instance down."
        ),
    )
    Config.add_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Perform DryRun=True authorization checks only; launch no "
            "instance and run no profiling. NOTE: a real SSH key pair and "
            "security group ARE still created in AWS even with --dry-run, "
            "mirroring provision.py's documented --dry-run semantics -- "
            "only launch_instance's actual launch and everything after it "
            "(waiting for SSH, pushing files, profiling, teardown) is "
            "skipped."
        ),
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help=(
            "Skip automatic teardown after profiling completes (or fails); "
            "leave the instance running and print its SSH command instead "
            "of tearing it down. The on-instance dead-man timer "
            "(--dead-man-minutes) still applies regardless of this flag -- "
            "it only disables orchestrate.py's own teardown call, not that "
            "backstop."
        ),
    )
    parser.add_argument(
        "--ssh-cidr",
        type=str,
        default=None,
        help=(
            "CIDR block allowed SSH access, e.g. 1.2.3.4/32. Overrides "
            "auto-detected caller IP (see provision.caller_ip())."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive cost-confirmation prompt (not required for --dry-run).",
    )
    args = parser.parse_args(argv)
    cfg = Config.from_parsed(args)

    # Fail fast on a missing boto3 before printing anything else, matching
    # provision.py's own main() -- no point asking the operator to confirm
    # a cost warning for a run that cannot possibly proceed.
    _require_boto3()

    legs = legs_for(cfg.topology)
    print(f"=== accelforge correlation-study orchestration: run_id={cfg.run_id} ===")
    print(_COST_WARNING)
    print(_estimate_wall_time_message(legs))
    print(f"Legs to profile: {', '.join(legs)} (topology={cfg.topology!r})")
    print(
        f"Purchasing mode: {cfg.purchasing}. Instance type: {cfg.instance_type}. "
        f"Region: {cfg.region}."
    )

    if not args.dry_run and not args.yes:
        if not _prompt_yes_no("Proceed with provisioning and profiling? [yes/N]: "):
            print("Aborted by user.")
            return 1

    ec2_client = boto3.client("ec2", region_name=cfg.region)
    ssm_client = boto3.client("ssm", region_name=cfg.region)

    ssh_cidr = args.ssh_cidr
    if not ssh_cidr:
        ip = caller_ip()
        ssh_cidr = f"{ip}/32"
    print(f"SSH will be allowed from: {ssh_cidr}")

    ami_id = resolve_ami(ssm_client, cfg.ami_ssm_parameter)
    print(f"Resolved AMI: {ami_id}")

    key_name = f"{cfg.tag_project}-{cfg.run_id}"
    key_path = ensure_key_pair(ec2_client, key_name, cfg.key_dir)
    print(f"Key pair ready: {key_name} -> {key_path}")

    sg_name = f"{cfg.tag_project}-{cfg.run_id}-sg"
    sg_id = ensure_security_group(ec2_client, sg_name, ssh_cidr, cfg.tag_project, cfg.run_id)
    print(f"Security group ready: {sg_id}")

    # Mirrors provision.py's main(): the dry_run flag flows all the way
    # into launch_instance (which makes a real, DryRun=True API call) so a
    # dry run exercises the exact same request-building code path a real
    # launch would, rather than short-circuiting before this call.
    launch_result = launch_instance(
        ec2_client, cfg, ami_id, sg_id, key_name, dry_run=args.dry_run
    )

    if args.dry_run:
        print(
            f"Dry run complete (purchasing checked: {launch_result['purchasing_used']}); "
            "no instance was launched, no SSH was attempted, and no profiling ran. "
            "The key pair and security group above WERE created for real -- "
            f"clean them up with: python teardown.py --run-id {cfg.run_id} --delete-key"
        )
        return 0

    instance_id = launch_result["instance_id"]
    purchasing_used = launch_result["purchasing_used"]
    print(f"Launched instance {instance_id} ({purchasing_used}).")

    # See this function's "Design/WHY" docstring Notes above for the full
    # rationale: write state now, with public_ip left as an explicit
    # placeholder, rather than waiting until wait_for_instance (which can
    # block for minutes and can itself fail) returns.
    state: Dict[str, Any] = {
        "run_id": cfg.run_id,
        "region": cfg.region,
        "instance_id": instance_id,
        "sg_id": sg_id,
        "key_name": key_name,
        "key_path": str(key_path),
        "public_ip": None,
        "purchasing_used": purchasing_used,
        "ami_id": ami_id,
    }
    state_path = write_state(cfg.state_dir, cfg.run_id, state)
    print(f"State written to: {state_path} (public_ip pending SSH reachability).")

    known_hosts_path = cfg.state_dir / "known_hosts"
    run_data_dir = _DATA_DIR / cfg.run_id

    try:
        public_ip = wait_for_instance(ec2_client, instance_id)
        state["public_ip"] = public_ip
        write_state(cfg.state_dir, cfg.run_id, state)
        print(f"Instance is running and SSH-reachable at {public_ip}")
        print(f"SSH command: ssh -i {key_path} {cfg.ssh_user}@{public_ip}")

        _push_files(cfg, key_path, public_ip, known_hosts_path)
        _run_setup(cfg, key_path, public_ip, known_hosts_path)

        for leg in legs:
            _run_leg(cfg, leg, key_path, public_ip, known_hosts_path, run_data_dir)
    finally:
        # Teardown-in-finally: this block runs whether the try body above
        # succeeded, raised (e.g. a failed run_profile.sh -> a propagating
        # subprocess.CalledProcessError), or was interrupted -- so an
        # (expensive, 8x H100) instance is never left running just because
        # one profiling leg failed partway through. See
        # _teardown_and_cleanup's docstring for how a *second* failure
        # (teardown itself failing) is handled without masking whichever
        # exception was already propagating out of the try body.
        if args.keep_alive:
            _print_keep_alive_notice(cfg, key_path, state)
        else:
            _teardown_and_cleanup(ec2_client, state, state_path, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
