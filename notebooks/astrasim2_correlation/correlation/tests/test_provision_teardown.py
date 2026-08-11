"""Tests for config.py, provision.py, and teardown.py.

These tests never touch real AWS: every boto3 client call is intercepted
by ``botocore.stub.Stubber``, which asserts on the exact request
parameters and returns a canned response (or raises a canned
``ClientError``) instead of making a network call. This lets the tests
exercise real botocore request-building and error-handling code paths --
including exact parameter shape assertions -- with zero network access and
zero AWS credentials, matching this work package's hard constraint that
nothing may call AWS.

Import strategy
-----------------
``config.py``/``provision.py``/``teardown.py`` live directly under
``correlation/`` (one level above this ``tests/`` package), and
``correlation/`` is deliberately *not* a Python package (no
``__init__.py`` there -- see the work-package spec). ``provision.py`` and
``teardown.py`` both do a plain ``from config import Config``, which only
resolves if ``correlation/`` is on ``sys.path``. We therefore insert that
directory onto ``sys.path`` explicitly before importing any of the three
modules under test, rather than relying on pytest's own rootdir-insertion
behavior (which happens to also achieve this here, but only for the
specific "prepend" import mode and package-marker layout currently in
place -- an explicit insert is robust to either changing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# boto3 is not (and must not become) an accelforge package dependency; skip
# this whole module rather than error if it is not installed in the
# environment running the tests.
boto3 = pytest.importorskip("boto3")

from botocore.exceptions import ClientError  # noqa: E402  (after importorskip)
from botocore.stub import Stubber  # noqa: E402

_CORRELATION_DIR = Path(__file__).resolve().parent.parent
if str(_CORRELATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CORRELATION_DIR))

import config  # noqa: E402
import provision  # noqa: E402
import teardown  # noqa: E402

Config = config.Config


def _make_client(service_name: str):
    """Build a boto3 client with dummy credentials, safe for Stubber use.

    Parameters
    ----------
    service_name : str
        E.g. ``"ec2"`` or ``"ssm"``.

    Returns
    -------
    botocore.client.BaseClient
        A real boto3 client object, never used to make a real network
        call in these tests (every call site below is wrapped in a
        ``Stubber`` context).

    Notes
    -----
    Design: ``Stubber`` intercepts the HTTP send step, but botocore still
    runs its normal request-signing step first, which raises
    ``NoCredentialsError`` if no credentials are configured anywhere
    (env vars, profile, instance metadata, ...). This test environment
    intentionally has none, so every client is built with harmless dummy
    static credentials purely to satisfy the signer -- these are never
    sent anywhere, since Stubber never performs a real HTTP request.
    """
    return boto3.client(
        service_name,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


# ---------------------------------------------------------------------------
# resolve_ami
# ---------------------------------------------------------------------------


def test_resolve_ami_returns_stubbed_parameter_value():
    """resolve_ami extracts Parameter.Value from the SSM response."""
    ssm_client = _make_client("ssm")
    stubber = Stubber(ssm_client)
    parameter_name = (
        "/aws/service/deeplearning/ami/x86_64/"
        "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
    )
    stubber.add_response(
        "get_parameter",
        {
            "Parameter": {
                "Name": parameter_name,
                "Value": "ami-0123456789abcdef0",
                "Type": "String",
            }
        },
        {"Name": parameter_name},
    )

    with stubber:
        result = provision.resolve_ami(ssm_client, parameter_name)

    stubber.assert_no_pending_responses()
    assert result == "ami-0123456789abcdef0"


# ---------------------------------------------------------------------------
# launch_instance
# ---------------------------------------------------------------------------


def test_launch_instance_spot_then_ondemand_falls_back_on_capacity_error():
    """A spot InsufficientInstanceCapacity error triggers an on-demand retry.

    Also asserts (per the work-package spec) that the first, failing
    request was a spot request -- i.e. it carried InstanceMarketOptions --
    both directly (inspecting the kwargs dict) and indirectly (via
    Stubber's expected_params, which would fail the test if
    launch_instance's real spot request didn't match).

    Design (Fix 10): the FIRST call's expected_params is an independently
    hardcoded literal dict, not derived from
    ``provision._build_run_instances_kwargs`` -- deriving it from the same
    helper the code under test calls would make this test circular (a bug
    in that helper's request-shape would go undetected, since the test's
    expectation and the code's actual request would drift together). The
    literal below pins the exact request shape against the work-package
    spec instead of against the code's own helper. The second (on-demand
    fallback) call keeps using the helper-derived ``ondemand_kwargs`` for
    convenience, since its shape isn't the focus of this particular test.
    """
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)

    cfg = Config(purchasing="spot-then-ondemand", run_id="test-run")
    ami_id = "ami-0123456789abcdef0"
    sg_id = "sg-0123456789abcdef0"
    key_name = "accelforge-correlation-test-run"

    # Independently hardcoded, per the work-package spec's exact field list
    # -- see the docstring above for why this must NOT be derived from
    # provision._build_run_instances_kwargs.
    spot_kwargs_literal = {
        "ImageId": ami_id,
        "InstanceType": "p5.48xlarge",
        "KeyName": key_name,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
        "InstanceInitiatedShutdownBehavior": "terminate",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 200,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": "accelforge-correlation"},
                    {"Key": "RunId", "Value": "test-run"},
                    {"Key": "Name", "Value": "accelforge-correlation-test-run"},
                ],
            },
            {
                "ResourceType": "volume",
                "Tags": [
                    {"Key": "Project", "Value": "accelforge-correlation"},
                    {"Key": "RunId", "Value": "test-run"},
                    {"Key": "Name", "Value": "accelforge-correlation-test-run"},
                ],
            },
        ],
        "DryRun": False,
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        },
    }
    # Kept helper-derived for the fallback call, per the spec ("if convenient").
    ondemand_kwargs = provision._build_run_instances_kwargs(
        cfg, ami_id, sg_id, key_name, dry_run=False, use_spot=False
    )

    # The spec's explicit ask: the FIRST request must have carried
    # InstanceMarketOptions (spot), the fallback must not.
    assert "InstanceMarketOptions" in spot_kwargs_literal
    assert "InstanceMarketOptions" not in ondemand_kwargs

    stubber.add_client_error(
        "run_instances",
        service_error_code="InsufficientInstanceCapacity",
        service_message="There is no Spot capacity available.",
        expected_params=spot_kwargs_literal,
    )
    stubber.add_response(
        "run_instances",
        {"Instances": [{"InstanceId": "i-0123456789abcdef0"}]},
        expected_params=ondemand_kwargs,
    )

    with stubber:
        result = provision.launch_instance(
            ec2_client, cfg, ami_id, sg_id, key_name, dry_run=False
        )

    stubber.assert_no_pending_responses()
    assert result == {
        "instance_id": "i-0123456789abcdef0",
        "purchasing_used": "ondemand",
    }


def test_launch_instance_spot_only_does_not_fall_back():
    """purchasing="spot" propagates the ClientError instead of retrying on-demand."""
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)

    cfg = Config(purchasing="spot", run_id="test-run")
    ami_id = "ami-0123456789abcdef0"
    sg_id = "sg-0123456789abcdef0"
    key_name = "accelforge-correlation-test-run"

    spot_kwargs = provision._build_run_instances_kwargs(
        cfg, ami_id, sg_id, key_name, dry_run=False, use_spot=True
    )
    # Only one response is ever queued: if launch_instance incorrectly
    # attempted a second (fallback) call, Stubber itself would raise for
    # having no more queued responses, which is not a ClientError -- so
    # pytest.raises(ClientError) below would fail loudly in that case too.
    stubber.add_client_error(
        "run_instances",
        service_error_code="InsufficientInstanceCapacity",
        service_message="There is no Spot capacity available.",
        expected_params=spot_kwargs,
    )

    with stubber:
        with pytest.raises(ClientError):
            provision.launch_instance(
                ec2_client, cfg, ami_id, sg_id, key_name, dry_run=False
            )

    stubber.assert_no_pending_responses()


def test_launch_instance_dry_run_success_does_not_raise(capsys):
    """A DryRunOperation error is treated as a successful authorization check."""
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)

    cfg = Config(purchasing="ondemand", run_id="test-run")
    ami_id = "ami-0123456789abcdef0"
    sg_id = "sg-0123456789abcdef0"
    key_name = "accelforge-correlation-test-run"

    ondemand_kwargs = provision._build_run_instances_kwargs(
        cfg, ami_id, sg_id, key_name, dry_run=True, use_spot=False
    )
    stubber.add_client_error(
        "run_instances",
        service_error_code="DryRunOperation",
        service_message="Request would have succeeded, but DryRun flag is set.",
        expected_params=ondemand_kwargs,
    )

    with stubber:
        result = provision.launch_instance(
            ec2_client, cfg, ami_id, sg_id, key_name, dry_run=True
        )

    stubber.assert_no_pending_responses()
    assert result["purchasing_used"] == "ondemand"
    # No instance was actually created during a dry run.
    assert result["instance_id"] is None
    assert "dry-run OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ensure_security_group
# ---------------------------------------------------------------------------


def test_ensure_security_group_authorizes_requested_cidr():
    """ensure_security_group wires the given ssh_cidr into the ingress rule."""
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)

    vpc_id = "vpc-0123456789abcdef0"
    sg_id = "sg-0123456789abcdef0"
    ssh_cidr = "203.0.113.5/32"
    group_name = "accelforge-correlation-test-run-sg"
    run_id = "test-run"
    tag_project = "accelforge-correlation"

    stubber.add_response(
        "describe_vpcs",
        {"Vpcs": [{"VpcId": vpc_id, "IsDefault": True}]},
        {"Filters": [{"Name": "isDefault", "Values": ["true"]}]},
    )
    stubber.add_response(
        "create_security_group",
        {"GroupId": sg_id},
        {
            "GroupName": group_name,
            "Description": f"accelforge correlation study SG for run {run_id}",
            "VpcId": vpc_id,
        },
    )
    # The parameter that matters most here: expected_params pins the exact
    # CidrIp ensure_security_group must send, so the test fails loudly if
    # the wrong CIDR (or the wrong port) were ever authorized.
    stubber.add_response(
        "authorize_security_group_ingress",
        {},
        {
            "GroupId": sg_id,
            "IpPermissions": [
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
        },
    )
    stubber.add_response(
        "create_tags",
        {},
        {
            "Resources": [sg_id],
            "Tags": [
                {"Key": "Project", "Value": tag_project},
                {"Key": "RunId", "Value": run_id},
                {"Key": "Name", "Value": group_name},
            ],
        },
    )

    with stubber:
        result = provision.ensure_security_group(
            ec2_client, group_name, ssh_cidr, tag_project, run_id
        )

    stubber.assert_no_pending_responses()
    assert result == sg_id


# ---------------------------------------------------------------------------
# teardown.find_tagged_instances
# ---------------------------------------------------------------------------


def test_find_tagged_instances_returns_ids_from_both_reservations():
    """find_tagged_instances flattens across multiple Reservations entries."""
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)

    tag_project = "accelforge-correlation"
    expected_filters = {
        "Filters": [
            {"Name": "tag:Project", "Values": [tag_project]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped"],
            },
        ]
    }
    stubber.add_response(
        "describe_instances",
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-aaaa000000000001",
                            "State": {"Name": "running"},
                            "Tags": [{"Key": "RunId", "Value": "run-a"}],
                        }
                    ]
                },
                {
                    "Instances": [
                        {
                            "InstanceId": "i-bbbb000000000002",
                            "State": {"Name": "pending"},
                            "Tags": [{"Key": "RunId", "Value": "run-b"}],
                        }
                    ]
                },
            ]
        },
        expected_filters,
    )

    with stubber:
        result = teardown.find_tagged_instances(ec2_client, tag_project)

    stubber.assert_no_pending_responses()
    ids = {instance["InstanceId"] for instance in result}
    assert ids == {"i-aaaa000000000001", "i-bbbb000000000002"}


# ---------------------------------------------------------------------------
# teardown security-group delete retry
# ---------------------------------------------------------------------------


def test_delete_security_group_retries_past_dependency_violation(monkeypatch):
    """A DependencyViolation is retried (not fatal) and eventually succeeds."""
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)
    sg_id = "sg-0123456789abcdef0"

    stubber.add_client_error(
        "delete_security_group",
        service_error_code="DependencyViolation",
        service_message="resource sg-0123456789abcdef0 has a dependent object",
        expected_params={"GroupId": sg_id},
    )
    stubber.add_response("delete_security_group", {}, {"GroupId": sg_id})

    sleep_calls = []
    # Patch time.sleep as seen through teardown's own `import time`, so the
    # retry loop does not actually block the test suite for
    # _SG_DELETE_RETRY_SLEEP_S seconds.
    monkeypatch.setattr(teardown.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with stubber:
        teardown._delete_security_group_with_retry(ec2_client, sg_id)

    stubber.assert_no_pending_responses()
    assert len(sleep_calls) == 1


# ---------------------------------------------------------------------------
# Fix 3 (MAJOR): caller_ip() fails fast instead of fail-open
# ---------------------------------------------------------------------------


def test_caller_ip_raises_on_discovery_failure(monkeypatch):
    """caller_ip() raises RuntimeError (mentioning --ssh-cidr) on any discovery failure.

    Regression test for Fix 3: the old behavior returned the sentinel
    "0.0.0.0" on failure, which every caller turned into the CIDR
    "0.0.0.0/32" -- unreachable by anyone, including the operator -- only
    after real AWS resources already existed and were already billing.
    Failing fast here means the error surfaces before any of that happens.
    """

    def fake_urlopen(*args, **kwargs):
        raise provision.urllib.error.URLError("simulated DNS failure")

    monkeypatch.setattr(provision.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="--ssh-cidr"):
        provision.caller_ip()


# ---------------------------------------------------------------------------
# Fix 2 (MAJOR): provision.main() closes the orphan-instance window
# ---------------------------------------------------------------------------


def test_provision_main_writes_state_before_wait_and_reports_recovery_on_failure(
    tmp_path, monkeypatch, capsys
):
    """provision.main() writes state (public_ip=None) before wait_for_instance runs,
    and on a wait_for_instance failure prints a loud recovery block (instance
    id, STILL RUNNING AND BILLING, state file path, exact recovery command)
    and re-raises rather than silently losing track of a running instance.

    Every AWS-touching seam provision.main() has (resolve_ami,
    ensure_key_pair, ensure_security_group, launch_instance,
    wait_for_instance, and boto3.client itself) is monkeypatched with a
    lightweight fake, mirroring test_orchestrate.py's ``orchestrate_fakes``
    approach -- this exercises provision.main()'s own sequencing/state-file
    logic (what Fix 2 changed) without touching real AWS or needing a
    Stubber response sequence for calls this test never lets happen for
    real.
    """

    class _FakeBoto3:
        @staticmethod
        def client(service_name, region_name=None):
            return f"fake-{service_name}-client[{region_name}]"

    monkeypatch.setattr(provision, "boto3", _FakeBoto3)
    monkeypatch.setattr(provision, "resolve_ami", lambda ssm, param: "ami-fake0123456789")

    def fake_ensure_key_pair(ec2, key_name, key_dir):
        key_dir.mkdir(parents=True, exist_ok=True)
        return key_dir / f"{key_name}.pem"

    monkeypatch.setattr(provision, "ensure_key_pair", fake_ensure_key_pair)
    monkeypatch.setattr(
        provision,
        "ensure_security_group",
        lambda ec2, name, cidr, tag_project, run_id: "sg-fake0123456789",
    )
    monkeypatch.setattr(
        provision,
        "launch_instance",
        lambda ec2, cfg, ami_id, sg_id, key_name, dry_run=False: {
            "instance_id": "i-fake0123456789",
            "purchasing_used": "spot",
        },
    )

    state_dir = tmp_path / "state"
    run_id = "test-run-orphan-window"

    def failing_wait_for_instance(ec2, instance_id):
        # By the time wait_for_instance is called, state must already be on
        # disk with public_ip still the None placeholder -- exactly the
        # ordering Fix 2 requires (state written BEFORE the SSH wait, not
        # only after it succeeds).
        state_path = state_dir / f"{run_id}.json"
        assert state_path.exists(), "state file must exist before wait_for_instance is called"
        written = json.loads(state_path.read_text())
        assert written["instance_id"] == instance_id
        assert written["public_ip"] is None
        raise TimeoutError("simulated SSH-reachability timeout")

    monkeypatch.setattr(provision, "wait_for_instance", failing_wait_for_instance)

    argv = [
        "--yes",
        "--run-id",
        run_id,
        "--key-dir",
        str(tmp_path / "keys"),
        "--state-dir",
        str(state_dir),
        "--ssh-cidr",
        "203.0.113.5/32",
    ]

    with pytest.raises(TimeoutError):
        provision.main(argv)

    # The state file must survive the failure -- teardown.py needs it to
    # find and tear down the still-running instance later.
    state_path = state_dir / f"{run_id}.json"
    assert state_path.exists()

    err = capsys.readouterr().err
    assert "i-fake0123456789" in err
    assert "STILL RUNNING AND BILLING" in err
    assert str(state_path) in err
    assert f"python teardown.py --run-id {run_id} --region" in err


# ---------------------------------------------------------------------------
# Fix 4a (MAJOR): teardown.resolve_regions -- pure region-resolution logic
# ---------------------------------------------------------------------------
#
# These tests make no AWS calls and use no Stubber, per resolve_regions()'s
# own design (a pure function of `args` and a plain dict of loaded state
# files) -- `args` is a minimal argparse.Namespace stand-in, not a fully
# parsed CLI invocation, since resolve_regions() only ever consults
# `.region` and `.run_id`.


def test_resolve_regions_explicit_flag_wins_over_state_region():
    """An explicit --region always overrides a state file's own recorded region."""
    args = argparse.Namespace(region="us-west-2", run_id="run-a")
    states = {"run-a": {"region": "eu-central-1"}}

    assert teardown.resolve_regions(args, states) == {"us-west-2": ["run-a"]}


def test_resolve_regions_uses_state_file_region_when_no_flag():
    """With no --region, a single --run-id's own recorded region is used."""
    args = argparse.Namespace(region=None, run_id="run-a")
    states = {"run-a": {"region": "ap-southeast-2"}}

    assert teardown.resolve_regions(args, states) == {"ap-southeast-2": ["run-a"]}


def test_resolve_regions_falls_back_to_config_default_when_unknown():
    """With no --region and no matching state file, Config's default region is used."""
    args = argparse.Namespace(region=None, run_id="run-with-no-state-file")
    states: dict = {}

    assert teardown.resolve_regions(args, states) == {
        Config().region: ["run-with-no-state-file"]
    }


def test_resolve_regions_all_groups_by_recorded_region_and_keeps_default():
    """--all (run_id=None) groups every known run by its own region, plus the default."""
    args = argparse.Namespace(region=None, run_id=None)
    states = {
        "run-a": {"region": "us-west-2"},
        "run-b": {"region": "us-west-2"},
        "run-c": {"region": "eu-central-1"},
        "run-d": {},  # no recorded region at all -> falls back to the default
    }

    result = teardown.resolve_regions(args, states)

    assert set(result["us-west-2"]) == {"run-a", "run-b"}
    assert result["eu-central-1"] == ["run-c"]
    # The default/fallback region (Config's own default) is always present
    # as a key, even though only run-d actually landed there via fallback,
    # so a caller iterating this mapping's keys always still searches it
    # for tag-only discovery of state-less instances.
    assert "run-d" in result[Config().region]


def test_resolve_regions_all_explicit_flag_overrides_every_run():
    """An explicit --region with --all overrides every individual run's own region."""
    args = argparse.Namespace(region="us-east-2", run_id=None)
    states = {
        "run-a": {"region": "us-west-2"},
        "run-b": {"region": "eu-central-1"},
    }

    result = teardown.resolve_regions(args, states)

    assert result == {"us-east-2": ["run-a", "run-b"]}


# ---------------------------------------------------------------------------
# Fix 4b (MAJOR): a stale state file (no matching live instance) is routed
# through teardown_run -- SG/key-pair cleanup included -- BEFORE its local
# state file is removed, instead of just being unlinked.
# ---------------------------------------------------------------------------


def test_teardown_one_region_stale_state_calls_delete_security_group_before_removing_file(
    tmp_path,
):
    """A stale run's security group is deleted before its state file disappears.

    Regression test for Fix 4b: the pre-fix behavior just unlinked a stale
    state file without ever touching AWS, leaking the security group (and,
    with --delete-key, the key pair) of any run whose instance died out
    from under it -- e.g. via the on-instance dead-man timer firing -- before
    teardown.py was ever run. Stubber's queued ``delete_security_group``
    response is only consumed if ``_teardown_one_region``'s internal
    ``teardown_run`` call actually invokes it (``stubber.assert_no_pending_
    responses()`` below fails the test otherwise); combined with the state
    file only being removed from disk AFTER that call returns without
    raising, these two assertions together establish the required "SG
    deleted before state file disappears" ordering.
    """
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)

    run_id = "stale-run"
    instance_id = "i-0123456789abcdef0"
    sg_id = "sg-0123456789abcdef0"
    tag_project = "accelforge-correlation"

    state = {
        "run_id": run_id,
        "region": "us-east-1",
        "instance_id": instance_id,
        "sg_id": sg_id,
        "key_name": "accelforge-correlation-stale-run",
        "key_path": None,
    }
    state_path = tmp_path / f"{run_id}.json"
    state_path.write_text(json.dumps(state))

    # Discovery: no live instance matches this run -- makes it "stale".
    stubber.add_response(
        "describe_instances",
        {"Reservations": []},
        {
            "Filters": [
                {"Name": "tag:Project", "Values": [tag_project]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping", "stopped"],
                },
            ]
        },
    )
    # teardown_run's terminate step: the instance is already gone (e.g. the
    # dead-man timer fired) -- tolerated as InvalidInstanceID.NotFound, per
    # teardown_run's own existing contract.
    stubber.add_client_error(
        "terminate_instances",
        service_error_code="InvalidInstanceID.NotFound",
        service_message=f"The instance ID '{instance_id}' does not exist",
        expected_params={"InstanceIds": [instance_id]},
    )
    # The assertion this test exists for: delete_security_group MUST be
    # called (Stubber raises on assert_no_pending_responses() otherwise).
    stubber.add_response("delete_security_group", {}, {"GroupId": sg_id})

    with stubber:
        exit_code = teardown._teardown_one_region(
            ec2_client,
            tag_project,
            "us-east-1",
            None,
            {run_id: state},
            {run_id: state_path},
            delete_key=False,
            yes=True,
        )

    stubber.assert_no_pending_responses()
    assert exit_code == 0
    # The state file is gone only now that teardown_run (including
    # delete_security_group) has already completed successfully.
    assert not state_path.exists()


def test_teardown_one_region_stale_state_keeps_file_if_teardown_run_raises(tmp_path, monkeypatch):
    """If the stale-cleanup teardown_run call itself raises, the state file survives.

    Complements the happy-path stale-cleanup test above: a failed cleanup
    must not lose track of the leak by removing the state file anyway.
    """
    ec2_client = _make_client("ec2")
    stubber = Stubber(ec2_client)

    run_id = "stale-run-cleanup-fails"
    sg_id = "sg-0123456789abcdef0"
    tag_project = "accelforge-correlation"

    # No instance_id: teardown_run skips straight to security-group
    # deletion, which is the call we make fail here.
    state = {"run_id": run_id, "region": "us-east-1", "sg_id": sg_id, "key_name": None}
    state_path = tmp_path / f"{run_id}.json"
    state_path.write_text(json.dumps(state))

    # Avoid actually sleeping between retries (see
    # test_delete_security_group_retries_past_dependency_violation's
    # identical rationale for patching teardown's own `import time`).
    monkeypatch.setattr(teardown.time, "sleep", lambda seconds: None)

    stubber.add_response(
        "describe_instances",
        {"Reservations": []},
        {
            "Filters": [
                {"Name": "tag:Project", "Values": [tag_project]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping", "stopped"],
                },
            ]
        },
    )
    # A persistent DependencyViolation exhausts _delete_security_group_with_retry's
    # retries and surfaces as a RuntimeError from teardown_run.
    for _ in range(teardown._SG_DELETE_MAX_RETRIES):
        stubber.add_client_error(
            "delete_security_group",
            service_error_code="DependencyViolation",
            service_message="resource has a dependent object",
            expected_params={"GroupId": sg_id},
        )

    with stubber:
        exit_code = teardown._teardown_one_region(
            ec2_client,
            tag_project,
            "us-east-1",
            None,
            {run_id: state},
            {run_id: state_path},
            delete_key=False,
            yes=True,
        )

    stubber.assert_no_pending_responses()
    assert exit_code == 1
    assert state_path.exists()
