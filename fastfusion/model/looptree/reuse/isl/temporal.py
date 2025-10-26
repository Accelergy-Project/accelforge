"""
Handles the ISL temporal reuse functions.
"""

from dataclasses import dataclass

import islpy as isl

from fastfusion.model.looptree.reuse.isl.isl_functions import map_to_shifted
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import (
    TEMPORAL_TAGS,
    Fill,
    Occupancy,
)


@dataclass(frozen=True)
class TemporalReuse:
    """Results for an temporal reuse analysis."""

    effective_occupancy: Occupancy
    fill: Fill


def analyze_temporal_reuse(
    occ: Occupancy, exploit_reuse: bool = True, multi_loop_reuse: bool = True
) -> TemporalReuse:
    """
    Computes the required fill to satisfy the buffer occupancy.
    If the buffer can `exploit_reuse`, then the fill will only consist
    of data not currently resident in buffer.

    Parameters
    ----------
    occ:
        The logical occupancy to be temporally analyzed.
    exploit_reuse:
        Temporally exploits reuse through persisting data currently in buffer
        to the next time step.
    multi_loop_reuse:
        Whether when this loop, or one above it in the memory hierarchy, loops,
        does the buffer flush.

    Returns
    -------
    A struct containing a `..types.Fill` which is how to load data into the buffer
    across time and a `..types.Occupancy` describing the effective_occupancy across
    time (i.e., what data needs to be persisted in the buffer per time step and what
    can be ignored/purged).

    TODO: Make sure spaces are named properly
    """
    if exploit_reuse:
        return fill_from_occupancy(occ, multi_loop_reuse)
    return TemporalReuse(occ, Fill(occ.tags, occ.map_))


def fill_from_occupancy(
    occupancy: Occupancy, multiple_loop_reuse: bool
) -> TemporalReuse:
    """
    Given an occupancy and if you're allowed to reuse across loops, calculate the
    `fill` and the `effective_occupancy` per time step.

    Parameters
    ----------
    occupancy:
        The logical occupancy of data in logical buffers.
    multi_loop_reuse:
        If you are allowed to use data between loop iterations.

    Returns
    -------
    A `TemporalReuse` object that contains the `fill` and `effective_occpancy`
    of the lowest buffer level.
    """
    # Iterates through each dimension in reverse order (i.e., deepest loop first)
    occ = occupancy.map_.copy()
    tags = occupancy.tags.copy()
    for dim_idx, tag in reversed(list(enumerate(occupancy.tags))):
        if not isinstance(tag, TEMPORAL_TAGS):
            continue
        # Check if temporal dimension is "trivial," i.e., equals a singular value
        proj_occ: isl.Map = occ.project_out(isl.dim_type.in_, dim_idx, 1)
        reinserted_occ: isl.Map = (
            proj_occ.insert_dims(isl.dim_type.in_, dim_idx, 1)
        ).intersect_domain(occ.domain())

        if occ.plain_is_equal(reinserted_occ) or occ.is_equal(reinserted_occ):
            occ = proj_occ
            tags.pop(dim_idx)
            continue

        # Nontrivial analysis
        time_shift: isl.Map
        if not multiple_loop_reuse:
            time_shift = map_to_shifted(occ.domain().get_space(), dim_idx, -1)
        # Calculates the
        else:
            # TODO: this is a better way of getting time_shift. Use method to
            # replace the other branch (!multi_loop_reuse)
            spacetime_domain: isl.Set = occ.domain()
            spacetime_domain_to_time_domain: isl.Map = isl.Map.identity(
                spacetime_domain.get_space().map_from_set()
            )
            spacetime_domain_to_space_domain: isl.Map = isl.Map.identity(
                spacetime_domain.get_space().map_from_set()
            )

            # Prunes out the output dimensions that do not correspond to the
            # correct mapping into a generic space-to-space relation.
            for idx, tag in reversed(list(enumerate(tags))):
                if not isinstance(tag, TEMPORAL_TAGS):
                    spacetime_domain_to_time_domain = (
                        spacetime_domain_to_time_domain.project_out(
                            isl.dim_type.out, idx, 1
                        )
                    )
                else:
                    spacetime_domain_to_space_domain = (
                        spacetime_domain_to_space_domain.project_out(
                            isl.dim_type.out, idx, 1
                        )
                    )

            # Properly constrains the spacetime_domain_to_time_domain's domain.
            spacetime_domain_to_time_domain = (
                spacetime_domain_to_time_domain.intersect_domain(spacetime_domain)
            )
            time_domain: isl.Set = spacetime_domain_to_time_domain.range()
            # Creates a map of the time_domain to previous regions of the time_domain.
            time_domain_to_past: isl.Map = (
                isl.Map.lex_gt(spacetime_domain_to_time_domain.range().get_space())
                .intersect_domain(time_domain)
                .intersect_range(time_domain)
            )
            # Restricts the relation to only the most recent previous region of time_domain.
            time_domain_to_most_recent_past = time_domain_to_past.lexmax()
            # Relates the current spacetime domain to its direct predecessor in time.
            time_shift: isl.Map = spacetime_domain_to_time_domain.apply_range(
                time_domain_to_most_recent_past.apply_range(
                    spacetime_domain_to_time_domain.reverse()
                )
            )

            # Prunes spatial relations to only ones that are valid.
            spacetime_domain_space_preserver: isl.Map = (
                spacetime_domain_to_space_domain.apply_range(
                    spacetime_domain_to_space_domain.reverse()
                )
            )
            time_shift = time_shift.intersect(spacetime_domain_space_preserver)

        # Gets the fill (i.e., feeds data not currently in buffer).
        occ_before: isl.Map = time_shift.apply_range(occ)
        fill: isl.Map = occ.subtract(occ_before)

        return TemporalReuse(Occupancy(tags, occ), Fill(tags, fill))

    return TemporalReuse(Occupancy(tags, occ), Fill(tags, occ))
