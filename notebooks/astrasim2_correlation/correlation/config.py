"""Shared configuration for the NCCL profiling correlation-study AWS scripts.

This module defines :class:`Config`, the single source of truth for every
tunable knob used by ``provision.py``, ``teardown.py``, and (per the plan)
the sibling ``orchestrate.py`` that a later work package will add. Keeping
the configuration in one frozen dataclass -- rather than threading loose
kwargs through each script -- means every script agrees on defaults,
validation, and CLI flag names without duplicating logic.

Construction paths
-------------------
``Config`` instances can be built three ways, all of which funnel through
the same validation in :meth:`Config.__post_init__`:

1. Directly, e.g. ``Config(region="us-west-2")`` -- handy for tests and for
   any future caller that wants a config without touching argparse at all.
2. Via :meth:`Config.from_args`, which owns its own
   :class:`argparse.ArgumentParser` end to end.
3. Via :meth:`Config.add_args` + :meth:`Config.from_parsed`, which lets a
   *caller* (``provision.py``, ``teardown.py``) build one shared parser,
   add its own script-specific flags (e.g. ``--dry-run``), and only then
   hand the resulting namespace back to ``Config`` to extract just the
   fields that belong to it. This is the path ``provision.py`` and
   ``teardown.py`` actually use.

Design: frozen dataclass
-------------------------
``Config`` is declared ``frozen=True`` so that once built it can be passed
into ``provision.py`` functions (``launch_instance``, ``write_state``, ...)
without any risk of one function's edits leaking into another's view of the
same run. Frozen dataclasses cannot assign to ``self.<field>`` in the usual
way, so any post-construction normalization (parsing "2x2x2" into a tuple,
generating a run id, coercing str paths to :class:`pathlib.Path`) goes
through ``object.__setattr__`` inside ``__post_init__`` -- this is the
standard, documented escape hatch for "derive a field after validation" on
a frozen dataclass.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Design: anchor key_dir/state_dir to this file's directory (not the
# process cwd) so that `python provision.py` behaves identically no matter
# where the user's shell happens to be sitting when they invoke it.
_THIS_DIR = Path(__file__).resolve().parent

_ALLOWED_PURCHASING = frozenset({"spot", "ondemand", "spot-then-ondemand"})
_ALLOWED_TOPOLOGY = frozenset({"fc", "torus", "both"})

# Default sweep of NCCL collectives profiled by the (sibling-work-package)
# profiling scripts. Kept here, not in profiling code, so a single
# `--collectives` override on the CLI is the one place a user needs to
# change to alter the sweep.
_DEFAULT_COLLECTIVES: Tuple[str, ...] = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "alltoall",
    "broadcast",
    "sendrecv",
)

# The study profiles one 8x H100 node; a torus overlay must therefore have
# axis dimensions whose product is exactly 8 (one "logical GPU slot" per
# axis position), regardless of how many axes are used.
_TORUS_GPU_COUNT = 8

_RUN_ID_TIME_FORMAT = "%Y%m%d-%H%M%S"


def _default_run_id() -> str:
    """Generate a fresh, sortable run identifier.

    Returns
    -------
    str
        ``"correl-" + UTC timestamp`` formatted as
        ``%Y%m%d-%H%M%S`` (e.g. ``"correl-20260716-161503"``). Lexicographic
        sort order matches chronological order, which is convenient when
        listing ``.state/*.json`` files or AMI/SG names in a shell.

    Notes
    -----
    Uses UTC (not local time) so run ids are unambiguous and comparable
    regardless of which machine or timezone invokes the script.
    """
    return "correl-" + datetime.datetime.now(datetime.timezone.utc).strftime(
        _RUN_ID_TIME_FORMAT
    )


def _is_power_of_two(n: int) -> bool:
    """Return whether a positive integer is an exact power of two.

    Parameters
    ----------
    n : int
        Value to test.

    Returns
    -------
    bool
        ``True`` if ``n > 0`` and ``n & (n - 1) == 0``, ``False`` otherwise
        (including for ``n <= 0``).

    Examples
    --------
    >>> _is_power_of_two(1024)
    True
    >>> _is_power_of_two(0)
    False
    >>> _is_power_of_two(3)
    False
    """
    return n > 0 and (n & (n - 1)) == 0


def _parse_torus_dims(value: Any) -> Tuple[int, ...]:
    """Normalize a torus-dimensions spec into a tuple of ints.

    Accepts the three shapes this value can arrive in depending on
    construction path: an already-correct ``tuple[int, ...]`` (direct
    ``Config(...)`` construction), a ``list``/``tuple`` of ints or numeric
    strings (from a parsed YAML ``--config`` file), or a delimited string
    like ``"2x2x2"`` (from the CLI, where argparse hands raw strings to
    ``type=`` callables).

    Parameters
    ----------
    value : Any
        Torus dimensions in any of the accepted shapes described above.

    Returns
    -------
    tuple[int, ...]
        The parsed per-axis dimensions, in the order given.

    Raises
    ------
    ValueError
        If ``value`` is an empty string, or any component cannot be parsed
        as an integer.

    Examples
    --------
    >>> _parse_torus_dims("2x2x2")
    (2, 2, 2)
    >>> _parse_torus_dims([2, 4])
    (2, 4)
    >>> _parse_torus_dims((8,))
    (8,)
    """
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"torus_dims entries must all be integers, got {value!r}"
            ) from exc

    text = str(value).strip().lower()
    if not text:
        raise ValueError("torus_dims must not be empty")
    try:
        return tuple(int(part) for part in text.split("x"))
    except ValueError as exc:
        raise ValueError(
            f"torus_dims={value!r} is not a valid dims string; "
            "expected a form like '2x2x2'"
        ) from exc


def _parse_collectives(value: Any) -> Tuple[str, ...]:
    """Normalize a collectives spec into a tuple of collective names.

    Parameters
    ----------
    value : Any
        Either a ``list``/``tuple`` of collective-name strings (from a
        parsed YAML ``--config`` file or direct construction), or a
        comma-separated string (from the CLI).

    Returns
    -------
    tuple[str, ...]
        The parsed collective names, in the order given, with surrounding
        whitespace stripped and empty entries dropped (so a trailing comma
        like ``"all_reduce,"`` does not produce a spurious ``""`` entry).

    Examples
    --------
    >>> _parse_collectives("all_reduce, broadcast")
    ('all_reduce', 'broadcast')
    >>> _parse_collectives(["all_reduce", "broadcast"])
    ('all_reduce', 'broadcast')
    """
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


@dataclasses.dataclass(frozen=True)
class Config:
    """Immutable configuration for one correlation-study AWS provisioning run.

    All fields have defaults, so ``Config()`` alone yields a fully valid
    configuration (one on-demand-or-spot ``p5.48xlarge`` in ``us-east-1``,
    fully-connected topology, a freshly generated ``run_id``). Every
    construction path (direct, :meth:`from_args`, :meth:`from_parsed`) runs
    the same validation in :meth:`__post_init__`, so an invalid ``Config``
    can never be observed by downstream code.

    Parameters
    ----------
    region : str, default "us-east-1"
        AWS region to provision in. Also used for the SSM AMI lookup and
        must have the requested ``instance_type`` available.
    availability_zone : str or None, default None
        Specific AZ within ``region`` to pin the instance to. ``None``
        lets AWS/the spot fleet choose.
    instance_type : str, default "p5.48xlarge"
        EC2 instance type. The 8x H100 NVSwitch topology this study
        profiles is specific to ``p5.48xlarge``; other types are accepted
        without validation but are not what the rest of this work package
        was designed against.
    purchasing : str, default "spot-then-ondemand"
        One of ``"spot"``, ``"ondemand"``, ``"spot-then-ondemand"``. See
        ``provision.launch_instance`` for the fallback semantics of the
        combined mode.
    topology : str, default "fc"
        One of ``"fc"`` (fully connected, the NVSwitch-native topology),
        ``"torus"`` (a logical torus overlay profiled on top of the same
        physical node), or ``"both"``.
    torus_dims : tuple[int, ...], default (2, 2, 2)
        Per-axis dimensions of the logical torus. Precondition: the
        product of all dimensions must equal 8 (one axis slot per GPU on
        the node). Accepts a ``"2x2x2"``-style string on construction and
        normalizes it to a tuple of ints.
    collectives : tuple[str, ...], default (all_reduce, all_gather,
        reduce_scatter, alltoall, broadcast, sendrecv)
        NCCL collectives the (sibling) profiling scripts should sweep.
        Accepts a comma-separated string on construction.
    min_mib : int, default 1
        Smallest message size, in MiB, in the profiling sweep.
        Precondition: must be a power of two and ``<= max_mib``.
    max_mib : int, default 1024
        Largest message size, in MiB, in the profiling sweep.
        Precondition: must be a power of two and ``>= min_mib``.
    run_id : str or None, default None
        Identifier used to name/tag every AWS resource created for this
        run (instance, security group, key pair, state file). If left as
        ``None``, a fresh id is generated in :meth:`__post_init__` as
        ``"correl-" + UTC timestamp``. See the NOTE in
        :meth:`__post_init__` for why generation happens there rather than
        only in :meth:`from_args`.
    tag_project : str, default "accelforge-correlation"
        Value written to the ``Project`` tag on every AWS resource this
        run creates; also what ``teardown.py`` filters
        ``describe_instances`` on to discover a run's resources.
    ssh_user : str, default "ubuntu"
        Login user baked into the deep-learning AMI, used when printing
        the ``ssh`` command at the end of provisioning.
    ami_ssm_parameter : str, default
        "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
        SSM public parameter name that resolves to the latest matching
        Deep Learning AMI id for ``region``.
    root_volume_gb : int, default 200
        Size, in GiB, of the root ``gp3`` EBS volume attached at
        ``/dev/sda1``.
    dead_man_minutes : int, default 120
        Minutes of wall-clock time after which the on-instance dead-man
        timer (armed by setup scripts owned by a sibling work package)
        force-shuts-down the instance. This is a safety net independent of
        ``teardown.py`` actually being run; see the README's guardrails
        section.
    key_dir : pathlib.Path, default "<this file's dir>/keys"
        Directory where generated SSH private keys are written.
    state_dir : pathlib.Path, default "<this file's dir>/.state"
        Directory where per-run provisioning state JSON is written.

    Raises
    ------
    ValueError
        Raised by :meth:`__post_init__` if ``purchasing`` or ``topology``
        is not one of the allowed values, if ``torus_dims`` does not
        multiply out to 8 or contains a non-positive entry, if
        ``min_mib > max_mib``, or if either ``min_mib`` or ``max_mib`` is
        not a power of two.

    Examples
    --------
    >>> cfg = Config(region="us-west-2", purchasing="ondemand")
    >>> cfg.region, cfg.purchasing
    ('us-west-2', 'ondemand')
    >>> cfg.run_id is not None
    True
    """

    region: str = "us-east-1"
    availability_zone: Optional[str] = None
    instance_type: str = "p5.48xlarge"
    purchasing: str = "spot-then-ondemand"
    topology: str = "fc"
    torus_dims: Tuple[int, ...] = (2, 2, 2)
    collectives: Tuple[str, ...] = _DEFAULT_COLLECTIVES
    min_mib: int = 1
    max_mib: int = 1024
    run_id: Optional[str] = None
    tag_project: str = "accelforge-correlation"
    ssh_user: str = "ubuntu"
    ami_ssm_parameter: str = (
        "/aws/service/deeplearning/ami/x86_64/"
        "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
    )
    root_volume_gb: int = 200
    dead_man_minutes: int = 120
    key_dir: Path = _THIS_DIR / "keys"
    state_dir: Path = _THIS_DIR / ".state"

    def __post_init__(self) -> None:
        """Normalize field representations and validate all preconditions.

        Raises
        ------
        ValueError
            See the class docstring's ``Raises`` section; this method is
            where every one of those checks is actually enforced.

        Notes
        -----
        Runs for *every* construction path (``Config(...)`` directly,
        :meth:`from_args`, :meth:`from_parsed`), because dataclasses always
        call ``__post_init__`` after ``__init__``. This is a deliberate
        choice over validating only inside :meth:`from_args`: it means a
        test (or a future caller) that builds ``Config(purchasing="bogus")``
        directly fails loudly at construction time instead of silently
        producing an invalid config that only misbehaves once it reaches
        AWS calls.
        """
        # Frozen dataclasses disallow `self.field = ...`; object.__setattr__
        # is the standard, documented way to set fields from __post_init__.
        object.__setattr__(self, "torus_dims", _parse_torus_dims(self.torus_dims))
        object.__setattr__(self, "collectives", _parse_collectives(self.collectives))
        object.__setattr__(self, "key_dir", Path(self.key_dir))
        object.__setattr__(self, "state_dir", Path(self.state_dir))

        if self.run_id is None:
            # NOTE: the work-package spec describes run_id's default as
            # "generated ... at parse time", which most literally refers to
            # Config.from_args. We instead generate it here, in
            # __post_init__, so every construction path gets a valid run_id
            # -- see the docstring Notes above for why. This is the more
            # conservative reading: it can never produce a Config with
            # run_id=None reaching AWS tag values, which the "at parse
            # time" phrasing on its own does not guarantee for direct
            # `Config(...)` construction.
            object.__setattr__(self, "run_id", _default_run_id())

        if self.purchasing not in _ALLOWED_PURCHASING:
            raise ValueError(
                f"purchasing={self.purchasing!r} is not one of "
                f"{sorted(_ALLOWED_PURCHASING)}"
            )
        if self.topology not in _ALLOWED_TOPOLOGY:
            raise ValueError(
                f"topology={self.topology!r} is not one of {sorted(_ALLOWED_TOPOLOGY)}"
            )

        # A negative-dimension axis (e.g. (-2, -2, 2)) could still multiply
        # out to 8, silently passing a bare product check; reject it
        # explicitly since it is never physically meaningful for a torus.
        if any(d <= 0 for d in self.torus_dims):
            raise ValueError(
                f"torus_dims={self.torus_dims!r} must all be positive integers"
            )
        product = math.prod(self.torus_dims)
        if product != _TORUS_GPU_COUNT:
            raise ValueError(
                f"torus_dims={self.torus_dims!r} has product {product}, "
                f"expected {_TORUS_GPU_COUNT} (one p5.48xlarge node = "
                f"{_TORUS_GPU_COUNT} GPUs)"
            )

        if self.min_mib > self.max_mib:
            raise ValueError(
                f"min_mib={self.min_mib} must be <= max_mib={self.max_mib}"
            )
        if not _is_power_of_two(self.min_mib):
            raise ValueError(f"min_mib={self.min_mib} must be a power of two")
        if not _is_power_of_two(self.max_mib):
            raise ValueError(f"max_mib={self.max_mib} must be a power of two")

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register every ``Config`` field as a kebab-case CLI flag.

        Intended to be called by a script's own parser setup (see
        ``provision.py``/``teardown.py``) *before* that script adds its
        own extra flags (e.g. ``--dry-run``), so the two flag sets share
        one ``argparse.ArgumentParser`` and one ``--help`` output.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to add arguments to, mutated in place.

        Notes
        -----
        Design: every flag added here uses ``default=argparse.SUPPRESS``
        instead of the field's real default. This means an unset flag is
        simply *absent* from the parsed namespace, which is exactly what
        :meth:`from_parsed` needs to implement "CLI flags override
        ``--config`` YAML values override dataclass defaults" -- if every
        flag instead defaulted to its real value, :meth:`from_parsed` could
        not distinguish "user explicitly passed the default value" from
        "user didn't pass this flag at all", and CLI flags could never be
        overridden by anything.
        """
        # Only used to render human-readable defaults into --help text;
        # never used for the actual default values (see Notes above).
        defaults = cls()

        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help=(
                "Path to a YAML file of Config field overrides, applied "
                "before CLI flags (CLI flags always win over --config)."
            ),
        )
        parser.add_argument(
            "--region",
            type=str,
            default=argparse.SUPPRESS,
            help=f"AWS region (default: {defaults.region!r}).",
        )
        parser.add_argument(
            "--availability-zone",
            type=str,
            default=argparse.SUPPRESS,
            help="AWS availability zone, e.g. us-east-1a (default: let AWS choose).",
        )
        parser.add_argument(
            "--instance-type",
            type=str,
            default=argparse.SUPPRESS,
            help=f"EC2 instance type (default: {defaults.instance_type!r}).",
        )
        parser.add_argument(
            "--purchasing",
            type=str,
            choices=sorted(_ALLOWED_PURCHASING),
            default=argparse.SUPPRESS,
            help=f"Purchasing strategy (default: {defaults.purchasing!r}).",
        )
        parser.add_argument(
            "--topology",
            type=str,
            choices=sorted(_ALLOWED_TOPOLOGY),
            default=argparse.SUPPRESS,
            help=f"NCCL topology to profile (default: {defaults.topology!r}).",
        )
        parser.add_argument(
            "--torus-dims",
            type=_parse_torus_dims,
            default=argparse.SUPPRESS,
            help=(
                "Logical torus dims as e.g. '2x2x2'; product must be 8 "
                f"(default: {'x'.join(str(d) for d in defaults.torus_dims)!r})."
            ),
        )
        parser.add_argument(
            "--collectives",
            type=_parse_collectives,
            default=argparse.SUPPRESS,
            help=(
                "Comma-separated NCCL collectives to sweep "
                f"(default: {','.join(defaults.collectives)!r})."
            ),
        )
        parser.add_argument(
            "--min-mib",
            type=int,
            default=argparse.SUPPRESS,
            help=(
                "Smallest message size in MiB, must be a power of two "
                f"(default: {defaults.min_mib})."
            ),
        )
        parser.add_argument(
            "--max-mib",
            type=int,
            default=argparse.SUPPRESS,
            help=(
                "Largest message size in MiB, must be a power of two "
                f"(default: {defaults.max_mib})."
            ),
        )
        parser.add_argument(
            "--run-id",
            type=str,
            default=argparse.SUPPRESS,
            help=(
                "Run identifier used to tag/name all resources "
                "(default: generated as 'correl-<UTC timestamp>')."
            ),
        )
        parser.add_argument(
            "--tag-project",
            type=str,
            default=argparse.SUPPRESS,
            help=f"Value for the 'Project' tag on all resources (default: {defaults.tag_project!r}).",
        )
        parser.add_argument(
            "--ssh-user",
            type=str,
            default=argparse.SUPPRESS,
            help=f"SSH login user for the AMI (default: {defaults.ssh_user!r}).",
        )
        parser.add_argument(
            "--ami-ssm-parameter",
            type=str,
            default=argparse.SUPPRESS,
            help="SSM parameter name to resolve the AMI id from.",
        )
        parser.add_argument(
            "--root-volume-gb",
            type=int,
            default=argparse.SUPPRESS,
            help=f"Root EBS volume size in GiB (default: {defaults.root_volume_gb}).",
        )
        parser.add_argument(
            "--dead-man-minutes",
            type=int,
            default=argparse.SUPPRESS,
            help=(
                "Minutes before the on-instance dead-man timer force-shuts-down "
                f"the instance (default: {defaults.dead_man_minutes})."
            ),
        )
        parser.add_argument(
            "--key-dir",
            type=Path,
            default=argparse.SUPPRESS,
            help=f"Directory to store the generated SSH private key (default: {defaults.key_dir}).",
        )
        parser.add_argument(
            "--state-dir",
            type=Path,
            default=argparse.SUPPRESS,
            help=f"Directory to store per-run provisioning state JSON (default: {defaults.state_dir}).",
        )

    @classmethod
    def from_parsed(cls, namespace: argparse.Namespace) -> "Config":
        """Build a :class:`Config` from an already-parsed argparse namespace.

        Meant to be used together with :meth:`add_args`: a caller builds
        one ``ArgumentParser``, calls ``Config.add_args(parser)``, adds its
        own extra flags, calls ``parser.parse_args(argv)``, and passes the
        resulting namespace here. Any namespace attributes that are not
        ``Config`` field names (e.g. a caller's own ``--dry-run``) are
        ignored, so the same namespace can safely be shared with
        script-specific flags.

        Parameters
        ----------
        namespace : argparse.Namespace
            Parsed CLI arguments, as produced by
            ``parser.parse_args(...)`` on a parser that included
            :meth:`add_args`'s flags. If it has a ``config`` attribute
            (from the ``--config`` flag) that is truthy, that path is
            loaded as a YAML mapping of field overrides.

        Returns
        -------
        Config
            A validated ``Config`` built from, in increasing priority:
            dataclass defaults, then ``--config`` YAML values, then
            explicitly-passed CLI flags.

        Raises
        ------
        ValueError
            If ``--config`` points at a YAML document whose top level is
            not a mapping, or if any field fails :meth:`__post_init__`
            validation.
        OSError
            If ``--config`` points at a path that cannot be opened.
        """
        field_names = {f.name for f in dataclasses.fields(cls)}
        values: Dict[str, Any] = {}

        config_path = getattr(namespace, "config", None)
        if config_path:
            # Design: import PyYAML lazily, inside this branch, rather than
            # at module top. Only the --config path needs it; a plain
            # `provision.py --help` (or any run that never passes
            # --config) must keep working even in an environment that
            # only has boto3 installed and not PyYAML.
            import yaml

            with open(config_path, "r") as fh:
                yaml_values = yaml.safe_load(fh)
            if yaml_values is None:
                yaml_values = {}
            if not isinstance(yaml_values, dict):
                raise ValueError(
                    f"--config file {config_path!r} must contain a top-level "
                    f"YAML mapping, got {type(yaml_values).__name__}"
                )
            # Silently drop unknown keys rather than raising: this lets a
            # single shared YAML file carry keys meant for other tools
            # (e.g. a future orchestrate.py section) without every
            # consumer needing to know about every other consumer's keys.
            values.update({k: v for k, v in yaml_values.items() if k in field_names})

        # CLI flags win over --config values. Because add_args() gives
        # every flag default=argparse.SUPPRESS, `namespace` only carries a
        # key for a field the user actually typed on the command line, so
        # this unconditional overwrite is exactly "CLI beats YAML beats
        # dataclass default".
        for key, value in vars(namespace).items():
            if key in field_names:
                values[key] = value

        return cls(**values)

    @classmethod
    def from_args(cls, argv: Optional[List[str]] = None) -> "Config":
        """Parse ``argv`` with a fresh, ``Config``-only parser.

        Convenience wrapper around :meth:`add_args` + :meth:`from_parsed`
        for callers that only need ``Config``'s own flags and do not have
        any script-specific flags of their own to add.

        Parameters
        ----------
        argv : list[str] or None, default None
            Argument list to parse, as passed to
            ``argparse.ArgumentParser.parse_args``. ``None`` means "read
            from ``sys.argv[1:]``", argparse's own default behavior.

        Returns
        -------
        Config
            The parsed, validated configuration.

        Examples
        --------
        >>> Config.from_args(["--region", "us-west-2", "--purchasing", "ondemand"]).region
        'us-west-2'
        """
        parser = argparse.ArgumentParser(
            description="accelforge NCCL correlation-study provisioning config."
        )
        cls.add_args(parser)
        namespace = parser.parse_args(argv)
        return cls.from_parsed(namespace)
