# correlation/ -- AWS provisioning for NCCL profiling

This directory is the **empirical leg** of the ISL-model correlation
study: it profiles real NCCL collective-communication performance on one
AWS `p5.48xlarge` instance (8x H100, NVSwitch), both in the GPUs' native
fully-connected (FC) topology and under a logical torus overlay, so those
measurements can be compared against the model's and ASTRA-sim's
predictions.

## Prerequisites

- **AWS credentials** available to boto3 via the standard mechanisms (env
  vars `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`, a
  named profile via `AWS_PROFILE`, or an attached IAM role). This
  directory never manages or stores credentials itself.
- **boto3** installed in whatever Python environment you run these
  scripts from: `pip install boto3`. It is deliberately *not* a dependency
  of the `accelforge` package -- only this notebook's provisioning corner
  needs it.
- **Service Quotas.** `p5.48xlarge` requires 192 vCPUs of quota. Before
  your first run, check the Service Quotas console for your target region
  and confirm both:
  - the on-demand P-instance quota ("Running On-Demand P instances" or
    equivalent, depending on current AWS naming), and
  - the corresponding Spot P-instance vCPU quota (if you intend to use
    `--purchasing spot` or the default `spot-then-ondemand`),

  are each **>= 192 vCPUs**. Quota codes and exact names change over time
  and vary by account/region presentation -- look them up in the console
  rather than trusting a hardcoded code here. A quota that's too low
  surfaces as an `InsufficientInstanceCapacity`-adjacent or limit-exceeded
  error at launch time.

## Cost warning

`p5.48xlarge` on-demand pricing is roughly **$30-55/hr**, depending on
region and current AWS pricing -- **verify current pricing** at
<https://aws.amazon.com/ec2/pricing/on-demand/> before running anything.
A full profiling sweep (FC + torus, all collectives, the full message-size
range) is expected to take well under an hour, but see the dead-man-timer
note below: the default 120-minute timer is comfortably above that
estimate, but a large custom `--min-mib`/`--max-mib` range or collective
list can push a full sweep close to or past it. Spot pricing is
substantially cheaper when capacity is available, which is why
`spot-then-ondemand` is the default purchasing mode -- see
`provision.launch_instance`'s docstring for the exact fallback conditions.
Both `orchestrate.py` and `provision.py` print this same warning and
require an interactive `yes` confirmation before touching AWS (skippable
with `--yes`; not required for `--dry-run`).

## Quickstart

**`orchestrate.py` is the end-to-end path.** It provisions its own
instance, pushes the profiling scripts, runs the full FC and/or torus
sweep, fetches the results locally, and tears the instance down again --
all in one command:

```bash
# 1. Sanity-check permissions and request shape without launching anything
#    (still creates a real SSH key pair + security group -- see the
#    --dry-run note below):
python orchestrate.py --topology both --dry-run

# 2. The real run: provision, profile both legs, fetch results, tear down.
python orchestrate.py --topology both
```

**Do not run `provision.py` before `orchestrate.py`.** `orchestrate.py`
provisions its *own* instance internally; running `provision.py` first
would launch a *second*, independent `p5.48xlarge` instance that nothing
in the `orchestrate.py` run above knows about or tears down, silently
doubling your bill. `provision.py`/`teardown.py` are for advanced, manual
control only -- see the section below.

Every `Config` field (region, purchasing mode, topology, message-size
range, collectives, ...) is a CLI flag; run `python orchestrate.py --help`
for the full list, or pass `--config some.yaml` to load a batch of
overrides from YAML (CLI flags still win over anything in the YAML file).
Useful flags:

- `--keep-alive`: skip `orchestrate.py`'s own teardown at the end, leaving
  the instance running (its SSH command is printed) for manual
  inspection. The on-instance dead-man timer still applies regardless.
- `--dead-man-minutes <n>`: extend the dead-man timer past its 120-minute
  default -- see the note below.
- `--ssh-cidr <ip>/32`: skip auto-detecting your IP for the SSH
  security-group rule; required if IP auto-detection fails (see
  `provision.caller_ip`'s docstring).

### Fetched data layout

Each leg's results land under `data/<run_id>/<leg>/`:

```
data/<run_id>/<leg>/
├── csv/         # parsed, unified-schema CSVs consumed by the correlation notebook
├── raw/         # raw nccl-tests/torus_bench stdout logs, one per collective
└── metadata.txt # machine/software provenance (nvidia-smi topo, driver/NCCL versions, git rev)
```

`<leg>` is `fc` or `torus`. `data/` is the one subdirectory of this
project *not* gitignored -- fetched results are committed intentionally.

## Advanced / manual control: `provision.py` + `teardown.py`

`provision.py` and `teardown.py` are the individual "up" and "down" halves
`orchestrate.py` composes internally. Use them directly only if you need
manual control between provisioning and profiling (e.g. debugging the
instance by hand, or running a custom workload instead of
`run_profile.sh`) -- most users should use `orchestrate.py` above instead.

```bash
# Provision one instance and leave it running (see the warning below):
python provision.py

# ... do whatever manual work you need on the instance ...

# Tear down that one run:
python teardown.py --run-id <run-id-provision.py-printed>

# Tear down everything this study has tagged, across every region this
# study's local state knows about:
python teardown.py --all

# Also delete the SSH key pair (AWS-side) and local PEM:
python teardown.py --run-id <run-id> --delete-key

# Audit: confirm nothing is left running, without tearing anything down.
# Exits 0 ("no running instances") when clean, 1 with a table otherwise --
# safe to use as a post-teardown check or a periodic cron/CI safety net.
python teardown.py --verify
```

**Warning: `provision.py` leaves the instance running with NO dead-man
timer until `setup_node.sh` is run on it.** The dead-man timer is armed
*by* `setup_node.sh` (a step `orchestrate.py` always runs for you, but
`provision.py` alone does not reach) -- so an instance provisioned via
`provision.py` and never followed up with `setup_node.sh` (or
`teardown.py`) will run, and bill, indefinitely with no automatic
backstop. If you provision manually, either run `setup_node.sh` on the
instance promptly or tear it down yourself as soon as you're done.

`--dry-run` on **both** `provision.py` and `orchestrate.py` performs
`DryRun=True` authorization checks only and launches no instance -- but a
**real** SSH key pair and security group ARE still created in AWS either
way (there is no dry-run equivalent for those two calls). Clean them up
with:

```bash
python teardown.py --run-id <run-id> --delete-key
```

`teardown.py` discovers resources two ways and reconciles them: local
`.state/*.json` files written by `provision.py`/`orchestrate.py`, and a
live `describe_instances` search by `Project`/`RunId` tags. The tag search
is authoritative, so teardown still works even if a state file was lost or
a run was started from a different machine. `--region` defaults to
*resolving per run* rather than to a fixed region: an explicit `--region`
flag always wins, otherwise each run's own state file (if any) supplies
its region, otherwise `us-east-1` (`Config`'s default) is used -- so
`--all`/`--verify` correctly span every region this study's local state
knows about in one invocation, not just one hardcoded region.

## Safety guardrails

- **Dead-man timer -- armed by `setup_node.sh`, not at instance launch.**
  Once `setup_node.sh` has run on the instance (always true for an
  `orchestrate.py` run; not automatic for a manual `provision.py` one --
  see the warning above), an on-instance timer force-shuts-down the box
  after `--dead-man-minutes` (default 120) regardless of whether
  `teardown.py` was ever run -- a backstop against a forgotten or failed
  teardown. Re-running `setup_node.sh` (e.g. via a second `orchestrate.py`
  leg) pushes the deadline back rather than erroring or stacking. Note
  that the 120-minute default can be tight for a long custom sweep --
  raise it with `--dead-man-minutes` if you expect to run past it.
- **`InstanceInitiatedShutdownBehavior=terminate`.** The instance is
  launched so that an in-instance `shutdown` (including the dead-man
  timer firing) *terminates* it rather than merely stopping it, so it
  cannot be left billing in a stopped state.
- **Teardown-in-`finally`.** `orchestrate.py` calls teardown from a
  `finally` block around the provisioning-through-profiling sequence, so a
  mid-sweep crash or a failed leg still tears the instance down (unless
  `--keep-alive` was passed).
- **PEM files are gitignored.** `keys/`, `.state/`, `*.pem`, and `logs/`
  are all excluded (see `.gitignore`) -- private key material and
  per-run state never get committed. `data/` (fetched profiling CSVs) is
  the one subdirectory *not* ignored; those results are committed
  intentionally.
