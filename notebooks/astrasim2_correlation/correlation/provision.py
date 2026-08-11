"""Provision one p5.48xlarge (8x H100, NVSwitch) EC2 instance for NCCL profiling.

This script is the "up" half of the correlation study's empirical leg: it
launches exactly one spot-first, on-demand-fallback instance, waits for it
to be SSH-reachable, and records everything needed to find/tear it down
again in a small JSON state file. See ``teardown.py`` for the "down" half
and ``README.md`` for the full runbook.

Every function below is written to be independently importable and
testable: ``orchestrate.py`` (a sibling work package, written separately)
imports these functions directly rather than shelling out to this file, so
their signatures are part of this module's public contract and must not
change without updating that caller too.

Design: boto3 is not a repo dependency
----------------------------------------
accelforge's ``pyproject.toml`` does not (and per this work package's scope
must not) depend on ``boto3`` -- only this AWS-provisioning corner of one
notebook's correlation study needs it. The import below is therefore
guarded: importing this module never fails just because boto3 is missing,
so ``python provision.py --help`` keeps working in any environment. Actual
AWS calls fail fast with a clear "pip install boto3" message via
:func:`_require_boto3`, called once at the top of :func:`main` before any
client is constructed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from config import Config

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - exercised only when boto3 truly absent
    boto3 = None
    # Design: fall back to plain Exception as a placeholder so that
    # `except ClientError:` clauses elsewhere in this module remain valid
    # Python (no NameError at import time) even when boto3 is missing.
    # Those clauses are only ever reached after _require_boto3() has
    # already raised, so this placeholder is never actually matched in
    # practice -- it exists purely to keep module import side-effect-free.
    class ClientError(Exception):  # type: ignore[no-redef]
        pass


def _require_boto3() -> None:
    """Raise a clear, actionable error if boto3 is not installed.

    Raises
    ------
    SystemExit
        Always, if ``boto3`` failed to import. The message tells the user
        exactly how to fix it rather than surfacing a bare
        ``ModuleNotFoundError`` traceback.
    """
    if boto3 is None:
        raise SystemExit(
            "boto3 is required for AWS provisioning but is not installed in "
            "this Python environment.\n"
            "Install it with: pip install boto3"
        )


# ClientError codes under which "spot-then-ondemand" purchasing retries on
# demand instead of failing the whole run. All represent spot-market
# scarcity/limits rather than a request-shape problem, so retrying the same
# request as on-demand is expected to succeed. p5.48xlarge (8x H100) is a
# scarce, high-demand instance type, so hitting these in practice is not
# unusual and should not be treated as fatal when the operator has opted
# into a fallback.
_SPOT_FALLBACK_ERROR_CODES = frozenset(
    {
        "InsufficientInstanceCapacity",
        "SpotMaxPriceTooLow",
        "MaxSpotInstanceCountExceeded",
        "Unsupported",
        "InstanceLimitExceeded",
    }
)

_SSH_POLL_INTERVAL_S = 5.0
_SSH_POLL_TIMEOUT_S = 5 * 60.0

_COST_WARNING = (
    "COST WARNING: p5.48xlarge on-demand pricing is roughly $30-55/hr "
    "depending on region and current AWS pricing. VERIFY CURRENT PRICING "
    "before proceeding: https://aws.amazon.com/ec2/pricing/on-demand/"
)


def resolve_ami(ssm_client, parameter: str) -> str:
    """Resolve an AMI id from a public SSM parameter.

    Parameters
    ----------
    ssm_client : botocore.client.BaseClient
        A boto3 ``ssm`` client (or a stub thereof).
    parameter : str
        Fully-qualified SSM parameter name, e.g.
        ``"/aws/service/deeplearning/ami/x86_64/.../latest/ami-id"``.

    Returns
    -------
    str
        The AMI id stored at ``parameter``.

    Raises
    ------
    botocore.exceptions.ClientError
        If ``parameter`` does not exist or the caller lacks
        ``ssm:GetParameter`` permission; propagated unmodified so callers
        see AWS's own error code and message.
    """
    response = ssm_client.get_parameter(Name=parameter)
    return response["Parameter"]["Value"]


def caller_ip() -> str:
    """Discover the caller's public IPv4 address via checkip.amazonaws.com.

    Used by :func:`main` to scope the provisioned security group's SSH
    ingress rule to just this machine, when the operator has not supplied
    an explicit ``--ssh-cidr``.

    Returns
    -------
    str
        The caller's public IP as a dotted-quad string.

    Raises
    ------
    RuntimeError
        If the IP could not be discovered for any reason (network error,
        timeout, or an empty response body). See Notes for why this fails
        loudly rather than falling back to any sentinel value.

    Notes
    -----
    Design: fails *fast* (raises) rather than falling back to a sentinel.
    An earlier version of this function failed *open* to the sentinel
    ``"0.0.0.0"`` on discovery failure, reasoning that a transient DNS blip
    or checkip.amazonaws.com outage shouldn't hard-fail an
    otherwise-working run. That reasoning had a bug: every caller turns
    this return value into a CIDR via ``f"{ip}/32"``, so the sentinel
    actually produced ``"0.0.0.0/32"`` -- a CIDR matching no address at
    all -- which locks *everyone*, including the operator, out over SSH,
    the opposite of what the old warning text claimed ("falling back to
    0.0.0.0/0", i.e. open to the world). Worse, that silent misconfiguration
    was only discoverable *after* a real key pair, security group, and
    instance had already been created and billing had already started.
    Raising here instead is strictly better on both axes this function
    cares about -- SAFE (no accidental everyone-blocked security group) and
    SECURE (no accidental world-open one either) -- and it fires before any
    AWS resource exists or any money is spent: both call sites
    (``provision.main`` and ``orchestrate.main``) invoke this function only
    when ``--ssh-cidr`` was not supplied, and always before
    ``ensure_key_pair``/``ensure_security_group``/``launch_instance``. The
    error message tells the operator exactly how to proceed: re-run with
    ``--ssh-cidr <their-ip>/32``.
    """
    try:
        with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10) as resp:
            ip = resp.read().decode("utf-8").strip()
        if not ip:
            raise ValueError("empty response body from checkip.amazonaws.com")
        return ip
    except (urllib.error.URLError, ValueError, OSError) as exc:
        raise RuntimeError(
            "Could not determine your public IP via checkip.amazonaws.com "
            f"({exc!r}). No AWS resources have been created yet, so there is "
            "nothing to clean up -- re-run with --ssh-cidr <your-ip>/32 to "
            "supply your CIDR explicitly instead of relying on auto-detection."
        ) from exc


def ensure_key_pair(ec2_client, key_name: str, key_dir: Path) -> Path:
    """Create (or reuse) an EC2 key pair and its local PEM file.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    key_name : str
        Name to give the key pair in AWS.
    key_dir : pathlib.Path
        Local directory to write ``<key_name>.pem`` into. Created if it
        does not already exist.

    Returns
    -------
    pathlib.Path
        Path to the local PEM file (either freshly written, or the
        existing one being reused).

    Raises
    ------
    RuntimeError
        If AWS reports the key pair already exists (``ClientError`` code
        ``InvalidKeyPair.Duplicate``) but no local PEM file is present.
        AWS never returns private key material for a pre-existing key
        pair, so there is no way to recover the PEM in this situation --
        the caller must delete the AWS-side key pair or choose a
        different ``run_id``.
    botocore.exceptions.ClientError
        For any other ``create_key_pair`` failure, propagated unmodified.

    Notes
    -----
    Side effect: writes a file to disk at ``<key_dir>/<key_name>.pem`` with
    ``0o600`` permissions (owner read/write only, matching what ``ssh``
    requires of private key files).
    """
    key_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / f"{key_name}.pem"

    try:
        response = ec2_client.create_key_pair(
            KeyName=key_name, KeyType="rsa", KeyFormat="pem"
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code == "InvalidKeyPair.Duplicate":
            if key_path.exists():
                print(
                    f"Key pair {key_name!r} already exists in AWS and a local "
                    f"PEM was found at {key_path}; reusing it."
                )
                return key_path
            raise RuntimeError(
                f"Key pair {key_name!r} already exists in AWS, but no local "
                f"PEM file was found at {key_path}. AWS never returns private "
                "key material for a pre-existing key pair, so it cannot be "
                "recovered. Either delete the AWS-side key pair "
                f"(aws ec2 delete-key-pair --key-name {key_name}) and re-run, "
                "or pass a different --run-id so a fresh key pair name is used."
            ) from exc
        raise

    key_material = response["KeyMaterial"]
    # Design: use os.open with the 0o600 mode baked into file creation
    # (rather than write-then-chmod) so the PEM is never briefly readable
    # at default (often world-readable) permissions between those two
    # steps.
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key_material)
    return key_path


def ensure_security_group(
    ec2_client, group_name: str, ssh_cidr: str, tag_project: str, run_id: str
) -> str:
    """Create a security group in the default VPC allowing SSH from one CIDR.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    group_name : str
        Name to give the new security group.
    ssh_cidr : str
        CIDR block (e.g. ``"1.2.3.4/32"`` or ``"0.0.0.0/0"``) to allow
        inbound TCP/22 from.
    tag_project : str
        Value for the ``Project`` tag on the new group.
    run_id : str
        Value for the ``RunId`` tag on the new group, and included in its
        description.

    Returns
    -------
    str
        The new security group's id.

    Raises
    ------
    RuntimeError
        If the region has no default VPC (``describe_vpcs`` returns no
        results for ``isDefault=true``). NCCL profiling on a single node
        has no cross-VPC requirements, so this script deliberately does
        not attempt to create or select a non-default VPC -- that is out
        of scope for a short-lived profiling instance.
    botocore.exceptions.ClientError
        For any ``create_security_group``/``authorize_security_group_ingress``/
        ``create_tags`` failure, propagated unmodified.
    """
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_list = vpcs.get("Vpcs", [])
    if not vpc_list:
        raise RuntimeError(
            "No default VPC found in this region. Create one "
            "(aws ec2 create-default-vpc) or provision a VPC manually, then "
            "re-run."
        )
    vpc_id = vpc_list[0]["VpcId"]

    create_resp = ec2_client.create_security_group(
        GroupName=group_name,
        Description=f"accelforge correlation study SG for run {run_id}",
        VpcId=vpc_id,
    )
    sg_id = create_resp["GroupId"]

    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [
                    {
                        "CidrIp": ssh_cidr,
                        "Description": "SSH access for accelforge correlation study",
                    }
                ],
            }
        ],
    )

    ec2_client.create_tags(
        Resources=[sg_id],
        Tags=[
            {"Key": "Project", "Value": tag_project},
            {"Key": "RunId", "Value": run_id},
            {"Key": "Name", "Value": group_name},
        ],
    )
    return sg_id


def _build_run_instances_kwargs(
    cfg: Config, ami_id: str, sg_id: str, key_name: str, dry_run: bool, use_spot: bool
) -> dict:
    """Build the ``run_instances`` kwargs shared by the spot and on-demand paths.

    Parameters
    ----------
    cfg : Config
        Run configuration.
    ami_id : str
        AMI id resolved by :func:`resolve_ami`.
    sg_id : str
        Security group id from :func:`ensure_security_group`.
    key_name : str
        Key pair name from :func:`ensure_key_pair`.
    dry_run : bool
        Whether to set ``DryRun=True`` on the request.
    use_spot : bool
        Whether to request a spot instance (adds ``InstanceMarketOptions``)
        or an on-demand one.

    Returns
    -------
    dict
        Keyword arguments ready to pass to ``ec2_client.run_instances(**kwargs)``.

    Notes
    -----
    Factored out of :func:`launch_instance` so the spot attempt and the
    on-demand fallback attempt build their request the same way apart from
    the one ``InstanceMarketOptions`` difference -- avoids the two paths
    silently drifting apart (e.g. one attempt forgetting a tag) as this
    function evolves.
    """
    kwargs = {
        "ImageId": ami_id,
        "InstanceType": cfg.instance_type,
        "KeyName": key_name,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
        # "terminate" (not "stop") so an in-instance `shutdown` -- e.g. the
        # dead-man timer armed by setup scripts -- fully releases the
        # instance rather than leaving it (and its EBS billing) stopped
        # but still provisioned.
        "InstanceInitiatedShutdownBehavior": "terminate",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": cfg.root_volume_gb,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": cfg.tag_project},
                    {"Key": "RunId", "Value": cfg.run_id},
                    {"Key": "Name", "Value": f"{cfg.tag_project}-{cfg.run_id}"},
                ],
            },
            {
                "ResourceType": "volume",
                "Tags": [
                    {"Key": "Project", "Value": cfg.tag_project},
                    {"Key": "RunId", "Value": cfg.run_id},
                    {"Key": "Name", "Value": f"{cfg.tag_project}-{cfg.run_id}"},
                ],
            },
        ],
        "DryRun": dry_run,
    }
    if cfg.availability_zone:
        kwargs["Placement"] = {"AvailabilityZone": cfg.availability_zone}
    if use_spot:
        kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }
    return kwargs


def launch_instance(
    ec2_client,
    cfg: Config,
    ami_id: str,
    sg_id: str,
    key_name: str,
    dry_run: bool = False,
) -> dict:
    """Launch exactly one instance, honoring ``cfg.purchasing``.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    cfg : Config
        Run configuration; ``cfg.purchasing`` selects the strategy below.
    ami_id : str
        AMI id resolved by :func:`resolve_ami`.
    sg_id : str
        Security group id from :func:`ensure_security_group`.
    key_name : str
        Key pair name from :func:`ensure_key_pair`.
    dry_run : bool, default False
        If ``True``, sets ``DryRun=True`` on every ``run_instances`` call.
        AWS answers a dry run with an error either way: ``DryRunOperation``
        means the call would have succeeded, ``UnauthorizedOperation``
        means the caller lacks permission. This function treats those two
        codes accordingly rather than as generic failures.

    Returns
    -------
    dict
        ``{"instance_id": str or None, "purchasing_used": "spot" or "ondemand"}``.
        ``instance_id`` is ``None`` when ``dry_run=True`` and the
        authorization check succeeded, since no instance was actually
        created in that case.

    Raises
    ------
    RuntimeError
        If a dry run reports ``UnauthorizedOperation`` (the configured
        credentials cannot launch this instance type/configuration).
    botocore.exceptions.ClientError
        - If ``cfg.purchasing == "spot"`` and the spot request fails for
          any reason (no fallback is attempted in this mode).
        - If ``cfg.purchasing == "spot-then-ondemand"`` and the spot
          request fails with a code *not* in
          :data:`_SPOT_FALLBACK_ERROR_CODES` (that set is deliberately
          narrow -- e.g. a malformed request should fail loudly rather
          than silently retrying as on-demand and masking the bug).
        - If the (possibly-fallback) on-demand request itself fails.

    Notes
    -----
    Purchasing strategies:

    - ``"ondemand"``: on-demand only, no spot attempt.
    - ``"spot"``: spot only; any failure propagates without a fallback.
    - ``"spot-then-ondemand"`` (the default): attempts spot first. If that
      attempt fails with one of :data:`_SPOT_FALLBACK_ERROR_CODES` --
      capacity/limit/market conditions rather than a malformed request --
      it prints the failure and retries once as on-demand. Any other
      ``ClientError`` code (e.g. a parameter validation error) propagates
      immediately without a fallback attempt, since retrying on-demand
      would not fix a malformed request and would only obscure the real
      error.
    """

    def _run(use_spot: bool) -> dict:
        kwargs = _build_run_instances_kwargs(cfg, ami_id, sg_id, key_name, dry_run, use_spot)
        try:
            response = ec2_client.run_instances(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "DryRunOperation":
                # AWS's DryRun contract: this specific error code means
                # "you WOULD have been authorized to make this call" -- it
                # is deliberately raised as an error even on the success
                # path, so seeing it here IS the successful outcome of a
                # dry run, not a failure.
                print("dry-run OK: authorized")
                return {
                    "instance_id": None,
                    "purchasing_used": "spot" if use_spot else "ondemand",
                }
            if code == "UnauthorizedOperation":
                raise RuntimeError(
                    "AWS denied the run_instances permission check "
                    "(UnauthorizedOperation). The configured credentials lack "
                    f"ec2:RunInstances (or a related) permission for "
                    f"{cfg.instance_type}."
                ) from exc
            raise
        instance_id = response["Instances"][0]["InstanceId"]
        return {
            "instance_id": instance_id,
            "purchasing_used": "spot" if use_spot else "ondemand",
        }

    if cfg.purchasing == "ondemand":
        return _run(use_spot=False)

    if cfg.purchasing == "spot":
        return _run(use_spot=True)

    # cfg.purchasing == "spot-then-ondemand" (validated by Config.__post_init__
    # to be one of exactly these three values).
    try:
        return _run(use_spot=True)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in _SPOT_FALLBACK_ERROR_CODES:
            print(f"Spot request failed ({code}); falling back to on-demand.")
            return _run(use_spot=False)
        raise


def wait_for_instance(ec2_client, instance_id: str) -> str:
    """Block until an instance is running and accepting TCP connections on port 22.

    Parameters
    ----------
    ec2_client : botocore.client.BaseClient
        A boto3 ``ec2`` client (or a stub thereof).
    instance_id : str
        Id of the instance to wait for.

    Returns
    -------
    str
        The instance's public IPv4 address.

    Raises
    ------
    RuntimeError
        If ``describe_instances`` returns no matching instance, or the
        instance has no public IP address (e.g. it landed in a subnet that
        does not auto-assign one).
    TimeoutError
        If port 22 does not become reachable within
        :data:`_SSH_POLL_TIMEOUT_S` seconds of the instance reaching the
        ``running`` state.
    botocore.exceptions.WaiterError
        If the ``instance_running`` waiter itself times out or the
        instance transitions to a terminal failure state.

    Notes
    -----
    Two-stage wait, because "EC2 says running" and "sshd is accepting
    connections" are different events with a real gap between them (boot,
    cloud-init, driver/NCCL setup on the deep learning AMI): first the
    ``instance_running`` waiter (AWS-side state), then a plain TCP connect
    poll against port 22 (this script's own liveness check), every
    :data:`_SSH_POLL_INTERVAL_S` seconds for up to
    :data:`_SSH_POLL_TIMEOUT_S`.
    """
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    describe = ec2_client.describe_instances(InstanceIds=[instance_id])
    reservations = describe.get("Reservations", [])
    if not reservations or not reservations[0].get("Instances"):
        raise RuntimeError(
            f"describe_instances returned no data for instance {instance_id!r}"
        )
    instance = reservations[0]["Instances"][0]
    public_ip = instance.get("PublicIpAddress")
    if not public_ip:
        raise RuntimeError(
            f"Instance {instance_id} is running but has no public IP address. "
            "Check that its subnet auto-assigns public IPs."
        )

    _wait_for_ssh_port(public_ip)
    return public_ip


def _wait_for_ssh_port(
    host: str,
    port: int = 22,
    interval_s: float = _SSH_POLL_INTERVAL_S,
    timeout_s: float = _SSH_POLL_TIMEOUT_S,
) -> None:
    """Poll a TCP port until it accepts a connection or a timeout elapses.

    Parameters
    ----------
    host : str
        Hostname or IP address to connect to.
    port : int, default 22
        TCP port to poll.
    interval_s : float, default 5.0
        Seconds to sleep between connection attempts.
    timeout_s : float, default 300.0
        Total seconds to poll before giving up.

    Raises
    ------
    TimeoutError
        If no connection succeeds within ``timeout_s`` seconds.

    Notes
    -----
    Uses ``socket.create_connection`` (a plain TCP connect/close) rather
    than an actual SSH handshake -- sufficient to confirm sshd is up
    without adding a paramiko/fabric dependency for a single boolean
    liveness check.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=interval_s):
                return
        except OSError:
            pass
        time.sleep(interval_s)
    raise TimeoutError(
        f"Timed out after {timeout_s}s waiting for {host}:{port} to accept "
        "TCP connections (SSH not yet reachable)."
    )


def write_state(state_dir, run_id: str, state: dict) -> Path:
    """Write a run's provisioning state to ``<state_dir>/<run_id>.json``.

    Parameters
    ----------
    state_dir : str or pathlib.Path
        Directory to write the state file into. Created if it does not
        already exist.
    run_id : str
        Run identifier; also the state file's basename (without ``.json``).
    state : dict
        JSON-serializable state to write. Expected (by ``teardown.py``) to
        contain ``run_id``, ``region``, ``instance_id``, ``sg_id``,
        ``key_name``, ``key_path``, ``public_ip``, ``purchasing_used``, and
        ``ami_id``, but this function itself does not validate the shape
        of ``state`` -- it is a thin, schema-agnostic writer so callers
        (including future ones) are free to add fields.

    Returns
    -------
    pathlib.Path
        Path to the written state file.

    Notes
    -----
    Side effect: writes ``<state_dir>/<run_id>.json``, creating
    ``state_dir`` if needed. Uses ``json.dump(..., default=str)`` so a
    stray non-JSON-native value (e.g. if a caller forgets to stringify a
    ``pathlib.Path``) is coerced to its string form instead of raising a
    ``TypeError`` deep in a provisioning run.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{run_id}.json"
    with open(state_path, "w") as fh:
        json.dump(state, fh, indent=2, default=str)
    return state_path


def _prompt_yes_no(prompt: str) -> bool:
    """Ask an interactive yes/no question, returning ``True`` only for "yes".

    Parameters
    ----------
    prompt : str
        Text to show before the input cursor.

    Returns
    -------
    bool
        ``True`` if the user typed exactly ``"yes"`` (case-insensitive,
        surrounding whitespace ignored); ``False`` for anything else,
        including EOF/blank input. Requiring the full word "yes" (not just
        "y") is a deliberate speed bump before an action that costs real
        money.
    """
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() == "yes"


def main(argv: Optional[list] = None) -> int:
    """Parse CLI args and provision one instance end to end.

    Parameters
    ----------
    argv : list[str] or None, default None
        Argument list, as passed to ``argparse``'s ``parse_args``. ``None``
        reads from ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: ``0`` on success (or a completed dry run),
        ``1`` if the user declined the cost confirmation prompt.

    Notes
    -----
    Side effects (skipped entirely when ``--dry-run`` is passed, except
    for the ``DryRun=True`` API calls themselves): creates an SSH key pair
    and local PEM, creates a security group, launches an instance, waits
    for it to be SSH-reachable, and writes a state JSON file -- TWICE: once
    immediately after launch (``public_ip`` as an explicit placeholder) and
    again once ``wait_for_instance`` returns a real IP, so a crash or
    interruption during the (potentially multi-minute) SSH wait still
    leaves a local record of a running, billing instance. See the inline
    "Design/WHY" comment at that first ``write_state`` call for the full
    rationale. Prints a cost warning and requires interactive ``"yes"``
    confirmation before any of that happens, unless ``--yes`` is passed.

    This function deliberately never tears anything down itself, even on a
    ``wait_for_instance`` failure -- ``provision.py``'s contract is
    provision-and-leave-running (see ``README.md``'s "advanced / manual
    control" section); recovery in that failure case is a printed
    ``teardown.py`` command for the operator to run, not an automatic call.
    """
    parser = argparse.ArgumentParser(
        prog="provision.py",
        description=(
            "Provision one p5.48xlarge (8x H100, NVSwitch) EC2 instance, "
            "spot-first with on-demand fallback, for NCCL profiling."
        ),
    )
    Config.add_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform DryRun=True authorization checks only; launch nothing.",
    )
    parser.add_argument(
        "--ssh-cidr",
        type=str,
        default=None,
        help=(
            "CIDR block allowed SSH access, e.g. 1.2.3.4/32. Overrides "
            "auto-detected caller IP (see caller_ip())."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive cost-confirmation prompt.",
    )
    args = parser.parse_args(argv)
    cfg = Config.from_parsed(args)

    # Fail fast on a missing boto3 before printing anything else -- no
    # point asking the operator to confirm a cost warning for a run that
    # cannot possibly proceed.
    _require_boto3()

    print(f"=== accelforge correlation-study provisioning: run_id={cfg.run_id} ===")
    print(_COST_WARNING)
    print(
        f"Purchasing mode: {cfg.purchasing}. Instance type: {cfg.instance_type}. "
        f"Region: {cfg.region}."
    )

    if not args.dry_run and not args.yes:
        if not _prompt_yes_no("Proceed with provisioning? [yes/N]: "):
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

    launch_result = launch_instance(
        ec2_client, cfg, ami_id, sg_id, key_name, dry_run=args.dry_run
    )
    if args.dry_run:
        print(
            f"Dry run complete (purchasing checked: {launch_result['purchasing_used']}); "
            "no instance was launched."
        )
        return 0

    instance_id = launch_result["instance_id"]
    purchasing_used = launch_result["purchasing_used"]
    print(f"Launched instance {instance_id} ({purchasing_used}).")

    # Design/WHY (closes an orphan-instance window): write the state file --
    # with public_ip as an explicit placeholder -- IMMEDIATELY after
    # launch_instance returns, before wait_for_instance is even called.
    # wait_for_instance can block for several minutes (instance boot, then
    # polling for SSH) and can itself raise (TimeoutError, WaiterError, or
    # simply be interrupted by Ctrl-C/a killed process). Before this fix, a
    # crash in that window left an instance running and billing with NO
    # local record of it at all, since write_state was only ever called
    # once, after wait_for_instance returned successfully. Writing state
    # now -- and rewriting it once the real public_ip is known below --
    # means teardown.py can always find and tear down this run from local
    # state alone, even if this process never gets past wait_for_instance.
    state = {
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

    print("Waiting for it to become reachable...")
    try:
        public_ip = wait_for_instance(ec2_client, instance_id)
    except Exception:
        # The instance is still running (and billing) regardless of why
        # wait_for_instance failed -- print a loud, impossible-to-miss block
        # naming exactly how to find and recover it, then re-raise
        # unmodified so this failure still surfaces as a nonzero exit code
        # (never silently swallowed).
        print("=" * 70, file=sys.stderr)
        print("ERROR: wait_for_instance failed (see traceback below).", file=sys.stderr)
        print(f"Instance {instance_id} IS STILL RUNNING AND BILLING.", file=sys.stderr)
        print(f"State file: {state_path}", file=sys.stderr)
        print(
            f"Recover with: python teardown.py --run-id {cfg.run_id} --region {cfg.region}",
            file=sys.stderr,
        )
        print("=" * 70, file=sys.stderr)
        raise

    print(f"Instance is running and SSH-reachable at {public_ip}")

    state["public_ip"] = public_ip
    state_path = write_state(cfg.state_dir, cfg.run_id, state)

    print(f"State written to: {state_path}")
    print(f"Public IP: {public_ip}")
    print(f"SSH command: ssh -i {key_path} {cfg.ssh_user}@{public_ip}")
    print(
        "Remember: tear this down when done "
        f"(python teardown.py --run-id {cfg.run_id})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
