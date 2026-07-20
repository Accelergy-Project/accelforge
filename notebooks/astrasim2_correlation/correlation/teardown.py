"""Tear down AWS resources created by ``provision.py``.

This script is the "down" half of the correlation study's empirical leg
(see ``provision.py`` for the "up" half and ``README.md`` for the full
runbook). It is designed to be safe to run more than once, and to work
even when its own local state is missing or stale, because it discovers
targets two ways and reconciles them:

1. Local state files under ``.state/*.json``, written by
   ``provision.py``'s ``write_state`` -- the fast, detailed path, since a
   state file already has the security group id and key pair name without
   any extra API calls.
2. A live ``describe_instances`` search filtered on the ``Project`` tag
   (and ``RunId`` tag, when ``--run-id`` is given) -- the authoritative
   path, since it reflects what AWS actually has running right now even if
   a state file was deleted, never written (a crash mid-provision), or the
   run was started from a different machine/checkout.

Every function below is written to be independently importable and
testable, matching ``provision.py``'s convention; see that module's
docstring for why boto3 is imported guarded rather than as a hard
dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - exercised only when boto3 truly absent
    boto3 = None

    # See provision.py's identical placeholder for why this exists: keeps
    # `except ClientError:` clauses valid Python even without boto3
    # installed, without ever actually being reachable (main() always
    # calls _require_boto3() first).
    class ClientError(Exception):  # type: ignore[no-redef]
        pass


def _require_boto3() -> None:
    """Raise a clear, actionable error if boto3 is not installed.

    Raises
    ------
    SystemExit
        Always, if ``boto3`` failed to import. See ``provision._require_boto3``
        for the identical rationale; kept as a separate copy here (rather
        than importing it from ``provision``) so this module has no
        import-time dependency on ``provision.py`` at all.
    """
    if boto3 is None:
        raise SystemExit(
            "boto3 is required for AWS teardown but is not installed in this "
            "Python environment.\n"
            "Install it with: pip install boto3"
        )


# Instance states worth discovering/tearing down. Deliberately excludes
# "shutting-down" and "terminated": those instances are already on their
# way out or gone and do not need (and, for "terminated", cannot receive)
# a terminate_instances call.
_ACTIVE_STATES = ("pending", "running", "stopping", "stopped")

# EC2 does not release the ENI-to-security-group association the instant
# terminate_instances returns (or the instance_terminated waiter is
# satisfied); delete_security_group can fail with DependencyViolation for a
# short window afterward while that teardown finishes propagating. Retrying
# with a fixed backoff is expected to succeed within a few attempts rather
# than being a genuine, permanent conflict.
_SG_DELETE_MAX_RETRIES = 5
_SG_DELETE_RETRY_SLEEP_S = 5.0


def find_tagged_instances(
    ec2_client, tag_project: str, run_id: Optional[str] = None
) -> List[dict]:
    """Find EC2 instances tagged for this study, optionally scoped to one run.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    tag_project : str
        Value the ``Project`` tag must match.
    run_id : str or None, default None
        If given, additionally require the ``RunId`` tag to match this
        value. If ``None``, instances from every run under ``tag_project``
        are returned.

    Returns
    -------
    list[dict]
        Raw ``Instance`` dicts (the ``Reservations[].Instances[]`` shape
        returned by ``describe_instances``), for instances currently in
        one of :data:`_ACTIVE_STATES`. Empty list if none match.

    Notes
    -----
    Paginates via ``NextToken`` manually (rather than
    ``ec2_client.get_paginator(...)``) so this function works identically
    against a plain client and a ``botocore.stub.Stubber``-wrapped one used
    in tests, without needing the Stubber to understand paginator internals.
    """
    filters = [
        {"Name": "tag:Project", "Values": [tag_project]},
        {"Name": "instance-state-name", "Values": list(_ACTIVE_STATES)},
    ]
    if run_id:
        filters.append({"Name": "tag:RunId", "Values": [run_id]})

    instances: List[dict] = []
    kwargs: Dict[str, Any] = {"Filters": filters}
    while True:
        response = ec2_client.describe_instances(**kwargs)
        for reservation in response.get("Reservations", []):
            instances.extend(reservation.get("Instances", []))
        next_token = response.get("NextToken")
        if not next_token:
            break
        kwargs["NextToken"] = next_token
    return instances


def _delete_security_group_with_retry(
    ec2_client,
    sg_id: str,
    max_retries: int = _SG_DELETE_MAX_RETRIES,
    retry_sleep_s: float = _SG_DELETE_RETRY_SLEEP_S,
) -> None:
    """Delete a security group, retrying on ``DependencyViolation``.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    sg_id : str
        Security group id to delete.
    max_retries : int, default 5
        Maximum number of ``delete_security_group`` attempts.
    retry_sleep_s : float, default 5.0
        Seconds to sleep between retries.

    Raises
    ------
    RuntimeError
        If every attempt fails with ``DependencyViolation`` (the ENI
        association never cleared in time).
    botocore.exceptions.ClientError
        For any ``ClientError`` code other than ``DependencyViolation`` or
        ``InvalidGroup.NotFound``, propagated immediately without retry.

    Notes
    -----
    See the module-level comment on :data:`_SG_DELETE_MAX_RETRIES` for why
    ``DependencyViolation`` specifically is retried rather than treated as
    fatal on the first failure.
    """
    last_exc: Optional[ClientError] = None
    for attempt in range(1, max_retries + 1):
        try:
            ec2_client.delete_security_group(GroupId=sg_id)
            print(f"Deleted security group {sg_id}.")
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "InvalidGroup.NotFound":
                print(f"Security group {sg_id} already gone.")
                return
            if code != "DependencyViolation":
                raise
            last_exc = exc
            if attempt < max_retries:
                print(
                    f"delete_security_group({sg_id}) hit DependencyViolation "
                    f"(attempt {attempt}/{max_retries}); retrying in "
                    f"{retry_sleep_s}s..."
                )
                time.sleep(retry_sleep_s)
    raise RuntimeError(
        f"Failed to delete security group {sg_id} after {max_retries} attempts "
        "due to a persistent DependencyViolation."
    ) from last_exc


def teardown_run(ec2_client, state: dict, delete_key: bool) -> None:
    """Terminate one run's instance and clean up its security group and key.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    state : dict
        Per-run state, either loaded from a ``.state/*.json`` file (see
        ``provision.write_state``) or synthesized from a live
        ``describe_instances`` result (see ``_derive_state_from_instance``).
        Recognized keys: ``instance_id``, ``sg_id``, ``key_name``,
        ``key_path``. All are optional -- a missing key simply skips that
        cleanup step, so a partially-populated state (e.g. derived from
        AWS alone, with no known ``key_path``) still tears down whatever it
        can.
    delete_key : bool
        If ``True``, also delete the AWS-side key pair (and the local PEM,
        if ``state["key_path"]`` is known and exists on disk).

    Raises
    ------
    RuntimeError
        If security group deletion exhausts its retries (see
        :func:`_delete_security_group_with_retry`).
    botocore.exceptions.ClientError
        For any AWS failure other than the specific "already gone" codes
        this function is written to tolerate (``InvalidInstanceID.NotFound``
        for the instance, ``InvalidGroup.NotFound`` for the security
        group), propagated unmodified.
    botocore.exceptions.WaiterError
        If the ``instance_terminated`` waiter times out or the instance
        reaches an unexpected terminal state.

    Notes
    -----
    Order matters: instance termination is started and waited on *before*
    security group deletion is attempted, because the security group
    cannot be deleted while an instance's network interface still
    references it (see :data:`_SG_DELETE_MAX_RETRIES`'s docstring).
    """
    instance_id = state.get("instance_id")
    if instance_id:
        try:
            ec2_client.terminate_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "InvalidInstanceID.NotFound":
                raise
            print(
                f"Instance {instance_id} already gone "
                "(InvalidInstanceID.NotFound); continuing teardown."
            )
        else:
            print(
                f"Termination requested for {instance_id}; waiting for it to "
                "fully terminate..."
            )
            waiter = ec2_client.get_waiter("instance_terminated")
            waiter.wait(InstanceIds=[instance_id])
            print(f"Instance {instance_id} terminated.")

    sg_id = state.get("sg_id")
    if sg_id:
        _delete_security_group_with_retry(ec2_client, sg_id)

    if delete_key:
        key_name = state.get("key_name")
        if key_name:
            try:
                ec2_client.delete_key_pair(KeyName=key_name)
                print(f"Deleted AWS key pair {key_name}.")
            except ClientError as exc:
                # Best-effort: a stale/already-deleted key pair should not
                # block the rest of teardown, but it is still reported, not
                # silently dropped.
                print(f"WARNING: failed to delete AWS key pair {key_name}: {exc}")

        key_path = state.get("key_path")
        if key_path:
            local_path = Path(key_path)
            if local_path.exists():
                local_path.unlink()
                print(f"Deleted local PEM {local_path}.")
        elif key_name:
            print(
                f"No local PEM path known for key pair {key_name!r} (this run's "
                "state was derived from AWS alone, not a local state file); "
                "only the AWS-side key pair was deleted. If a PEM for it exists "
                "on this or another machine, remove it manually."
            )


def _derive_state_from_instance(instance: dict) -> dict:
    """Reconstruct a minimal teardown ``state`` dict from a live instance.

    Used when a tagged instance is discovered via :func:`find_tagged_instances`
    but has no matching local ``.state/*.json`` file (deleted, never
    written, or written on a different machine). EC2 instance descriptions
    already carry everything ``teardown_run`` needs except the local PEM
    path, which cannot be recovered this way.

    Parameters
    ----------
    instance : dict
        One ``Instance`` dict as returned by ``describe_instances``.

    Returns
    -------
    dict
        ``{"run_id", "instance_id", "sg_id", "key_name", "key_path"}``,
        with ``key_path`` always ``None`` (see above) and ``run_id`` taken
        from the instance's ``RunId`` tag, falling back to the instance id
        itself if that tag is somehow missing.
    """
    tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
    security_groups = instance.get("SecurityGroups", [])
    return {
        "run_id": tags.get("RunId", instance["InstanceId"]),
        "instance_id": instance["InstanceId"],
        "sg_id": security_groups[0]["GroupId"] if security_groups else None,
        "key_name": instance.get("KeyName"),
        "key_path": None,
    }


def _state_dir_files(state_dir: Path, run_id: Optional[str]) -> List[Path]:
    """List local state files relevant to this teardown invocation.

    Parameters
    ----------
    state_dir : pathlib.Path
        Directory containing ``<run_id>.json`` state files.
    run_id : str or None
        If given, look only for ``<state_dir>/<run_id>.json``. If
        ``None``, return every ``*.json`` file in ``state_dir``.

    Returns
    -------
    list[pathlib.Path]
        Matching, existing file paths, sorted for deterministic output.
        Empty list if ``state_dir`` does not exist or nothing matches.
    """
    if not state_dir.exists():
        return []
    if run_id:
        candidate = state_dir / f"{run_id}.json"
        return [candidate] if candidate.exists() else []
    return sorted(state_dir.glob("*.json"))


def _load_state_file(path: Path) -> dict:
    """Load one state JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.state/<run_id>.json`` file.

    Returns
    -------
    dict
        The parsed JSON content.

    Raises
    ------
    OSError
        If ``path`` cannot be opened.
    json.JSONDecodeError
        If ``path`` does not contain valid JSON.
    """
    with open(path, "r") as fh:
        return json.load(fh)


def _prompt_yes_no(prompt: str) -> bool:
    """Ask an interactive yes/no question, returning ``True`` only for "yes".

    Parameters
    ----------
    prompt : str
        Text to show before the input cursor.

    Returns
    -------
    bool
        ``True`` only if the user typed exactly ``"yes"``
        (case-insensitive); ``False`` otherwise, including on EOF.
    """
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() == "yes"


def _verify(ec2_client, tag_project: str, run_id: Optional[str]) -> int:
    """Audit for any still-running tagged instances, without tearing anything down.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    tag_project : str
        Value the ``Project`` tag must match.
    run_id : str or None
        If given, scope the audit to just this run.

    Returns
    -------
    int
        ``0`` (and prints ``"no running instances"``) if nothing tagged
        remains in :data:`_ACTIVE_STATES`; ``1`` (and prints a table) if
        anything does. Intended for use as a CI/cron safety check after a
        teardown, so a stuck resource is caught rather than silently
        left running and accruing cost.
    """
    instances = find_tagged_instances(ec2_client, tag_project, run_id)
    if not instances:
        print("no running instances")
        return 0

    print("Tagged instances still present:")
    print(f"{'InstanceId':<21} {'State':<12} {'RunId'}")
    for instance in instances:
        tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
        state_name = instance.get("State", {}).get("Name", "unknown")
        print(f"{instance['InstanceId']:<21} {state_name:<12} {tags.get('RunId', '?')}")
    return 1


# ---------------------------------------------------------------------------
# Region resolution (Fix 4a)
# ---------------------------------------------------------------------------
#
# Design/WHY: before this fix, teardown.py always defaulted --region to
# "us-east-1" and used exactly that one region for every discovery/teardown
# call, regardless of what region a run's own state file recorded. A run
# provisioned in any other region (e.g. because capacity/quota forced a
# different --region at provision time) was therefore invisible to
# `teardown.py --all` and to a bare `teardown.py --run-id <id>` unless the
# operator remembered to pass --region explicitly every time -- silently
# leaving that run's instance running and billing. The fix: --region now
# defaults to None (so this module can tell "the user explicitly asked for
# us-east-1" apart from "the user said nothing"), and region resolution
# follows this priority, per run:
#   1. an explicit --region flag always wins (it is an explicit override of
#      whatever a state file might say, e.g. for recovering a run whose
#      state file was hand-edited or lost a region field);
#   2. otherwise, a matching local state file's own recorded "region" field
#      is authoritative (this is what makes --all correctly span multiple
#      regions in one invocation);
#   3. otherwise (no --region, no matching/region-bearing state file --
#      e.g. an instance discovered via live AWS tags alone, with no local
#      state at all), fall back to Config's own default region
#      ("us-east-1"), matching this module's pre-fix behavior for that one
#      case.
# resolve_regions() is a pure function (no AWS calls, no I/O) precisely so
# this priority logic is unit-testable on its own, per the work-package
# spec's explicit ask.


def resolve_regions(args, states: Dict[str, dict]) -> Dict[str, List[str]]:
    """Determine which AWS region(s) to operate in, and which run_ids live in each.

    See the "Region resolution" design comment immediately above this
    function for the full priority rationale (explicit ``--region`` flag,
    then a state file's own recorded region, then Config's default).

    Parameters
    ----------
    args : argparse.Namespace or any object with ``.region`` and ``.run_id``
        Only ``args.region`` (str or None) and ``args.run_id`` (str or
        None) are consulted; duck-typed so a test can pass a minimal stand-in
        without building a full parsed CLI namespace.
    states : dict[str, dict]
        Every locally known state dict, keyed by run_id, as loaded from
        ``.state/*.json`` files -- NOT pre-filtered to ``args.run_id``; this
        function does that scoping itself.

    Returns
    -------
    dict[str, list[str]]
        Mapping of region -> list of run_ids (drawn from `states`) resolved
        to that region.

        - If ``args.run_id`` is set: exactly one key (the resolved region
          for that one run), mapping to ``[args.run_id]``.
        - Otherwise (``--all`` or a bare ``--verify``): one key per distinct
          region recorded across every entry in `states` (grouped), PLUS
          the default/fallback region (``args.region`` if given, else
          Config's default) as its own key even if no state file happens to
          record it -- so a caller iterating this mapping's keys always
          still searches that region too, for tag-only discovery of
          instances with no matching local state file at all (e.g. a crash
          before ``provision.write_state`` ever ran).

    Notes
    -----
    Pure function: makes no AWS calls and performs no I/O, so it is
    directly unit-testable without a Stubber or any boto3 client.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(region=None, run_id=None)
    >>> resolve_regions(args, {"run-a": {"region": "us-west-2"}})
    {'us-east-1': [], 'us-west-2': ['run-a']}
    """
    default_region = args.region or Config().region

    if args.run_id:
        state = states.get(args.run_id)
        region = args.region or (state.get("region") if state else None) or default_region
        return {region: [args.run_id]}

    regions: Dict[str, List[str]] = {default_region: []}
    for run_id, state in states.items():
        region = args.region or state.get("region") or default_region
        regions.setdefault(region, []).append(run_id)
    return regions


def _client_for_region(clients: Dict[str, Any], region: str):
    """Return a cached boto3 ``ec2`` client for `region`, creating it on first use.

    Parameters
    ----------
    clients : dict[str, botocore.client.BaseClient]
        Mutable cache, keyed by region name; mutated in place on a cache
        miss. Callers own the dict's lifetime (typically one per
        :func:`main` invocation).
    region : str
        AWS region name to build (or reuse) a client for.

    Returns
    -------
    botocore.client.BaseClient
        A boto3 ``ec2`` client bound to `region`, reused across calls that
        pass the same `region` and the same `clients` dict.

    Notes
    -----
    Design: ``--all`` may need to talk to several regions in one
    invocation (one per distinct region recorded across this study's state
    files -- see :func:`resolve_regions`). Building a client lazily, keyed
    by region, keeps the number of client objects (and any connection-pool
    overhead) proportional to the number of *distinct regions* actually
    involved in one run, rather than the number of runs or the number of
    times a region happens to be revisited.
    """
    if region not in clients:
        clients[region] = boto3.client("ec2", region_name=region)
    return clients[region]


def _teardown_one_region(
    ec2_client,
    tag_project: str,
    region: str,
    run_id_filter: Optional[str],
    file_states_by_run: Dict[str, dict],
    file_paths_by_run: Dict[str, Path],
    delete_key: bool,
    yes: bool,
) -> int:
    """Discover, stale-clean, confirm, and tear down every matching resource in one region.

    Factored out of :func:`main` so :func:`main` itself only needs to
    resolve regions (:func:`resolve_regions`) and loop over them; this is
    also what makes the stale-state cleanup path (Fix 4b) directly
    unit-testable with a single ``botocore.stub.Stubber``-wrapped client,
    without needing to drive the CLI/argparse layer or monkeypatch
    ``boto3.client`` at all.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof) already bound to
        `region`.
    tag_project : str
        Value the ``Project`` tag must match (``Config().tag_project``).
    region : str
        The AWS region `ec2_client` is bound to; used only in printed
        messages (this function performs no region validation of its own).
    run_id_filter : str or None
        If given, :func:`find_tagged_instances` is scoped to just this one
        run (mirrors the original single-``--run-id`` behavior); ``None``
        discovers every tagged run in `region`.
    file_states_by_run : dict[str, dict]
        Locally known state dicts for the run_ids :func:`resolve_regions`
        assigned to this region, keyed by run_id.
    file_paths_by_run : dict[str, pathlib.Path]
        The corresponding ``.state/<run_id>.json`` paths, same keys as
        `file_states_by_run`.
    delete_key : bool
        Forwarded to every :func:`teardown_run` call in this region.
    yes : bool
        If ``False``, prompts for interactive confirmation before tearing
        down any LIVE target found in this region. Stale-state cleanup
        (see Notes) is never gated on this prompt: it only reclaims
        resources this function has already determined are orphaned (no
        matching live instance), not anything newly discovered as
        still-in-use.

    Returns
    -------
    int
        ``0`` if every teardown/cleanup in this region succeeded (including
        "nothing to do here"); ``1`` if any individual teardown/cleanup
        raised, or if the user declined confirmation for this region's live
        targets.

    Notes
    -----
    Order of operations, and why (Fix 4b): (1) live AWS discovery via
    :func:`find_tagged_instances`, reconciled against `file_states_by_run`
    exactly as the pre-fix module-level code did; (2) any state file in
    `file_states_by_run` with NO matching live instance is "stale" --
    rather than just unlinking its state file (the pre-fix behavior), it is
    now routed through :func:`teardown_run` FIRST. ``teardown_run``'s
    terminate step already tolerates ``InvalidInstanceID.NotFound``, so a
    truly-dead instance is a safe no-op there -- but its security group and
    (optionally) key pair do NOT disappear just because the instance is
    gone, e.g. via the on-instance dead-man timer terminating it out from
    under a local state file that was never cleaned up. Routing through
    ``teardown_run`` first closes that SG/key-pair leak; the state file is
    removed only if that call did NOT raise, so a failed stale cleanup
    leaves the state file in place for a future retry instead of silently
    losing track of the leak. (3) only THEN are any remaining LIVE targets
    confirmed and torn down, matching the pre-fix confirmation UX.
    """
    file_states_by_instance: Dict[str, dict] = {
        s["instance_id"]: s for s in file_states_by_run.values() if s.get("instance_id")
    }
    aws_instances = find_tagged_instances(ec2_client, tag_project, run_id_filter)

    targets: List[dict] = []
    for instance in aws_instances:
        instance_id = instance["InstanceId"]
        if instance_id in file_states_by_instance:
            targets.append(file_states_by_instance[instance_id])
        else:
            targets.append(_derive_state_from_instance(instance))

    exit_code = 0

    # --- Stale-state cleanup: see the "Order of operations" note above. ---
    live_run_ids = {t.get("run_id") for t in targets if t.get("run_id")}
    for run_id_key in set(file_states_by_run) - live_run_ids:
        print(
            f"State file for run {run_id_key!r} (region {region}) has no matching live AWS "
            "instance; routing it through teardown to reclaim any leftover security group/key "
            "pair before removing the state file."
        )
        try:
            teardown_run(ec2_client, file_states_by_run[run_id_key], delete_key=delete_key)
        except Exception as exc:  # noqa: BLE001
            # Design: deliberately broad, matching the identical rationale
            # on the live-target teardown loop below -- one stale run's
            # cleanup failure must not abort the rest of a --all batch.
            print(f"ERROR cleaning up stale run {run_id_key!r}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        state_path = file_paths_by_run.get(run_id_key)
        if state_path and state_path.exists():
            state_path.unlink()
            print(f"Removed state file {state_path}.")

    if not targets:
        if not file_states_by_run:
            print(f"No matching resources found in {region}.")
        return exit_code

    print(f"The following will be torn down in {region}:")
    for t in targets:
        print(
            f"  run_id={t.get('run_id', '?')} instance_id={t.get('instance_id', '?')} "
            f"sg_id={t.get('sg_id', '?')}"
        )

    if not yes and not _prompt_yes_no("Proceed with teardown? [yes/N]: "):
        print("Aborted by user.")
        return 1

    for t in targets:
        try:
            teardown_run(ec2_client, t, delete_key=delete_key)
        except Exception as exc:  # noqa: BLE001
            # Design: deliberately broad. One run's teardown failure should
            # not abort the rest of a --all batch; report it and keep
            # going, then reflect the failure in the process exit code
            # rather than swallowing it silently.
            print(f"ERROR tearing down run {t.get('run_id', '?')}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        run_id_key = t.get("run_id")
        state_path = file_paths_by_run.get(run_id_key)
        if state_path and state_path.exists():
            state_path.unlink()
            print(f"Removed state file {state_path}.")

    return exit_code


def main(argv: Optional[list] = None) -> int:
    """Parse CLI args and tear down matching resources.

    Parameters
    ----------
    argv : list[str] or None, default None
        Argument list, as passed to ``argparse``'s ``parse_args``. ``None``
        reads from ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code. For ``--verify``: ``0`` if every audited region
        is clean, ``1`` if tagged instances remain in any of them.
        Otherwise: ``0`` on a fully successful teardown across every region
        involved (or nothing to do anywhere), ``1`` if the user declined
        confirmation in any region or if any individual run's
        teardown/stale-cleanup raised.

    Notes
    -----
    ``--tag-project`` is deliberately not a flag here (this CLI's flag set
    is fixed by the work-package spec to
    ``[--region] [--run-id | --all] [--verify] [--delete-key] [--yes]``):
    the ``Project`` tag value used for discovery is always
    ``Config().tag_project`` (i.e. ``Config``'s dataclass default,
    ``"accelforge-correlation"``). A run provisioned with a *custom*
    ``--tag-project`` cannot be found by this CLI and must instead be torn
    down by calling :func:`find_tagged_instances`/:func:`teardown_run`
    directly with that project value -- both are plain importable
    functions for exactly this reason.

    ``--region`` now defaults to ``None`` (Fix 4a), NOT ``"us-east-1"``: see
    the "Region resolution" design comment above :func:`resolve_regions`
    for the full rationale and priority order. This function may therefore
    talk to *more than one* region in a single invocation (e.g. ``--all``
    spanning every region this study's local state knows about, or
    ``--verify`` auditing all of them) -- :func:`_client_for_region` keeps
    one cached boto3 client per distinct region actually needed.
    """
    parser = argparse.ArgumentParser(
        prog="teardown.py",
        description="Tear down accelforge correlation-study AWS resources.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help=(
            "AWS region. If omitted, each targeted run's own state-file-recorded "
            "region is used when known, else Config's default region (us-east-1); "
            "an explicit --region always overrides both. See resolve_regions()."
        ),
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-id", type=str, default=None, help="Tear down only this run.")
    target.add_argument(
        "--all",
        action="store_true",
        help=(
            "Tear down every accelforge-correlation-tagged run this study's local "
            "state knows about, across every region those runs were provisioned in."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Audit only: report any still-running tagged instances and exit non-zero if any remain.",
    )
    parser.add_argument(
        "--delete-key",
        action="store_true",
        help="Also delete the AWS key pair and local PEM.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    args = parser.parse_args(argv)

    if not args.verify and not args.run_id and not args.all:
        parser.error("one of --run-id, --all, or --verify is required")

    _require_boto3()

    # Design: instantiate a plain Config() rather than duplicating its
    # tag_project/state_dir default logic here. See the "Notes" above on
    # why --tag-project is not a teardown.py flag.
    defaults = Config()
    tag_project = defaults.tag_project
    state_dir = defaults.state_dir

    # Load EVERY locally known state file up front, regardless of scope --
    # resolve_regions() needs the full set to group by region (Fix 4a);
    # scope filtering (--run-id vs --all/--verify) happens inside
    # resolve_regions() and the per-region loop below, not here.
    all_states_by_run: Dict[str, dict] = {}
    all_paths_by_run: Dict[str, Path] = {}
    for path in _state_dir_files(state_dir, None):
        try:
            loaded = _load_state_file(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: could not read state file {path}: {exc}; skipping.")
            continue
        run_id_key = loaded.get("run_id", path.stem)
        all_states_by_run[run_id_key] = loaded
        all_paths_by_run[run_id_key] = path

    region_map = resolve_regions(args, all_states_by_run)
    clients: Dict[str, Any] = {}

    if args.verify:
        # Fix 4c: audit every region resolve_regions() knows about (not
        # only the --region flag/fallback), so a run provisioned in a
        # different region than the operator happens to be thinking about
        # is not silently skipped by an audit meant to catch exactly that.
        exit_code = 0
        for region in sorted(region_map):
            print(f"--- Verifying region {region} ---")
            client = _client_for_region(clients, region)
            if _verify(client, tag_project, args.run_id) != 0:
                exit_code = 1
        return exit_code

    exit_code = 0
    for region in sorted(region_map):
        ec2_client = _client_for_region(clients, region)
        run_ids_here = region_map[region]
        file_states_by_run = {
            rid: all_states_by_run[rid] for rid in run_ids_here if rid in all_states_by_run
        }
        file_paths_by_run = {
            rid: all_paths_by_run[rid] for rid in run_ids_here if rid in all_paths_by_run
        }
        region_exit_code = _teardown_one_region(
            ec2_client,
            tag_project,
            region,
            args.run_id,
            file_states_by_run,
            file_paths_by_run,
            args.delete_key,
            args.yes,
        )
        if region_exit_code != 0:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
