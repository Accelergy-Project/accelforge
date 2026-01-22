"""
TODO:   Is this file still necessary? It is referenced elsewhere but is no longer
        the format we are looking for.
"""

from dataclasses import dataclass, field

import islpy as isl


@dataclass
class IslReuseAnalysisOutput:
    ops: dict = field(default_factory=dict)
    fills: dict = field(default_factory=dict)
    occupancy: dict = field(default_factory=dict)
    op_occupancy: dict = field(default_factory=dict)
    reads_to_peer: dict = field(default_factory=dict)
    reads_to_parent: dict = field(default_factory=dict)
    temporal_steps: dict = field(default_factory=dict)
    fanout: dict = field(default_factory=dict)
    op_intensity: dict = field(default_factory=dict)


def deserialize_looptree_output(
    looptree_output, isl_ctx: isl.Context  #: bindings.looptree.LooptreeResult,
) -> IslReuseAnalysisOutput:
    output = IslReuseAnalysisOutput()

    output.ops = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.ops.items()
    }

    output.fills = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.fills.items()
    }

    output.occupancy = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.occupancy.items()
    }

    output.reads_to_peer = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.reads_to_peer.items()
    }

    output.reads_to_parent = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.reads_to_parent.items()
    }

    output.temporal_steps = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.temporal_steps.items()
    }

    return output
