from collections import defaultdict
import itertools

from fastfusion.frontend import arch
from fastfusion.frontend.specification import Specification
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import SIM
from fastfusion.mapper.FFM._join_pmappings.compatibility import Loop, Compatibility
from fastfusion.util import fzs
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import (
    make_full_equivalent_rank_variables,
)


class MapspaceGlobals:
    def __init__(
        self,
        sims: dict[str, list[SIM]],
        spec: Specification,
        objective_function_cols: list[str] = None,
        flattened_architecture: list[arch.Leaf] = None,
    ):
        self.sims = sims
        self.einsum_names = spec.workload.einsum_names
        self.einsum2ranks = {
            einsum_name: spec.workload.einsums[einsum_name].rank_variables
            for einsum_name in self.einsum_names
        }
        self.einsum2tensors = {
            einsum_name: spec.workload.einsums[einsum_name].tensor_names
            for einsum_name in self.einsum_names
        }
        self.tensor_names = set().union(
            *(self.einsum2tensors[e] for e in self.einsum_names)
        )
        self.fusable_tensor_names = spec.workload.fusable_tensor_names
        self.pairwise_equivalent_ranks = (
            spec.workload.get_pairwise_equivalent_rank_variables()
        )
        self.full_equivalent_ranks = make_full_equivalent_rank_variables(
            self.pairwise_equivalent_ranks
        )

        self.resource2capacity = {}
        flattened_architecture = (
            flattened_architecture or spec.get_flattened_architecture()
        )
        for l in flattened_architecture:
            if isinstance(l, arch.Memory):
                self.resource2capacity[l.name] = l.attributes.size

        self.objective_function_cols = objective_function_cols
        self.rank_translations = self._create_rank_translations(self.einsum2ranks)

        for i, (left_id, left_sims) in enumerate(sims.items()):
            for j, (right_id, right_sims) in enumerate(sims.items()):
                if i >= j:
                    continue

                left_live = self.get_live_tensors(*self.einsum_names[: i + 1])
                right_live = self.get_live_tensors(*self.einsum_names[j:])
                left_tensors = self.get_tensors(self.einsum_names[i])
                right_tensors = self.get_tensors(self.einsum_names[j])

                if not (left_live & right_live):
                    continue
                print(f"Checking {left_id} {right_id}")

                right_tilings = {
                    s.compatibility.clear_dead_tensors(
                        live_tensors=left_live
                    ).clear_dead_tensors(left_tensors, keep_loops=True)
                    for s in right_sims
                }
                assert right_tilings, f"R {left_id} {right_id}"
                for s in list(left_sims):
                    for t in self.get_possible_translations(s.compatibility, right_id):
                        t = t.clear_dead_tensors(live_tensors=right_live)
                        t = t.clear_dead_tensors(
                            live_tensors=right_tensors, keep_loops=True
                        )
                        if t in right_tilings:
                            break
                    else:
                        left_sims.remove(s)
                assert (
                    left_sims
                ), f"Removed all of left {left_id} while checking right {right_id}"

                left_tilings = {
                    s.compatibility.clear_dead_tensors(
                        live_tensors=right_live
                    ).clear_dead_tensors(right_tensors, keep_loops=True)
                    for s in left_sims
                }
                assert left_tilings, f"L {left_id} {right_id}"
                for s in list(right_sims):
                    for t in self.get_possible_translations(s.compatibility, left_id):
                        t = t.clear_dead_tensors(live_tensors=left_live)
                        t = t.clear_dead_tensors(
                            live_tensors=left_tensors, keep_loops=True
                        )
                        if t in left_tilings:
                            break
                    else:
                        right_sims.remove(s)
                assert (
                    right_sims
                ), f"Removed all of right {right_id} while checking left {left_id}"

        self.tensor2possible_loops_above = self._create_tensor2possible_loops_above()
        self.tensor2possible_loops_above_set = {
            k: {k2: set(v2) for k2, v2 in v.items()}
            for k, v in self.tensor2possible_loops_above.items()
        }
        self.tensor2memories = self._create_tensor2memories()
        self.einsum_tiling_2_sim = self._create_einsum_tiling_2_sim()
        self.einsum_rank_index_to_loops = self._create_einsum_rank_index_to_loops()
        (
            self.compatibility2leftcompatibility,
            self.compatibility2rightcompatibility,
            self.leftcompatibility2tiling,
            self.rightcompatibility2tiling,
        ) = self._create_compatibility()
        self.size_scale = len(self.einsum2ranks)
        n_optimal = sum(
            len(s.mappings.data) for simlist in self.sims.values() for s in simlist
        )
        n_pmappings = sum(
            s.mappings.n_pmappings for simlist in self.sims.values() for s in simlist
        )
        self.find_pmapping_scale = n_pmappings / n_optimal
        self.aliased_tensors = spec.workload.get_tensor_copies()

    def get_live_tensors(self, *einsums: str):
        return set.union(*(self.einsum2tensors[e] for e in einsums))

    def _create_compatibility(self):
        tiling2leftcompatibility = {}
        tiling2rightcompatibility = {}

        def tilings2compatibility(tilings: list[Compatibility], live_tensors: set[str]):
            return {t: t.clear_dead_tensors(live_tensors=live_tensors) for t in tilings}

        for i, (einsum_name, sim_list) in enumerate(self.sims.items()):
            if i > 0:
                prev_live = self.get_live_tensors(*self.einsum_names[:i])
                tiling2leftcompatibility[einsum_name] = tilings2compatibility(
                    [s.compatibility for s in sim_list],
                    prev_live,
                )
            if i < len(self.sims) - 1:
                next_live = self.get_live_tensors(*self.einsum_names[i + 1 :])
                tiling2rightcompatibility[einsum_name] = tilings2compatibility(
                    [s.compatibility for s in sim_list],
                    next_live,
                )

        leftcompatibility2tiling = {}
        rightcompatibility2tiling = {}
        for einsum_name in self.einsum_names:
            for src, dst in (
                (tiling2leftcompatibility, leftcompatibility2tiling),
                (tiling2rightcompatibility, rightcompatibility2tiling),
            ):
                if einsum_name not in src:
                    continue
                dst = dst.setdefault(einsum_name, {})
                for k, v in src[einsum_name].items():
                    dst.setdefault(v, []).append(k)
        return (
            tiling2leftcompatibility,
            tiling2rightcompatibility,
            leftcompatibility2tiling,
            rightcompatibility2tiling,
        )

    def _create_einsum_tiling_2_sim(self):
        einsum_tiling_2_sim = {}
        for e, sim_list in self.sims.items():
            cur_sims = defaultdict(list)
            for sim in sim_list:
                cur_sims[sim.compatibility].append(sim)
            einsum_tiling_2_sim[e] = {}
            for t, s in cur_sims.items():
                s = SIM.concat(s)
                einsum_tiling_2_sim[e][t] = s
        return einsum_tiling_2_sim

    def _create_tensor2possible_loops_above(self):
        tensor2possible_loops_above = {}
        for einsum_name, sim_list in self.sims.items():
            tensor2possible_loops_above[einsum_name] = defaultdict(set)
            for sim in sim_list:
                for tensor in sim.compatibility.tensors:
                    tensor2possible_loops_above[einsum_name][tensor] |= set(
                        sim.compatibility.loops[: tensor.above_loop_index]
                    )
        return {
            e: {s: list(l) for s, l in d.items()}
            for e, d in tensor2possible_loops_above.items()
        }

    def _create_tensor2memories(self):
        tensor2memories = {}
        for t in self.fusable_tensor_names:
            possible_memories = []
            for einsum_name, sim_list in self.sims.items():
                cur_memories = set()
                if t not in sim_list[0].tensor_names:
                    continue
                for sim in sim_list:
                    tensor = sim.compatibility.get_tensor_by_name(t)
                    cur_memories.add(tensor)
                possible_memories.append(cur_memories)
            if possible_memories:
                tensor2memories[t] = list(set.intersection(*possible_memories))
            else:
                raise ValueError(f"No memories for {t}")
        return tensor2memories

    def _create_rank_translations(self, einsum2ranks: dict[str, set[str]]):
        rank_translations = {}
        for einsum_name, ranks in einsum2ranks.items():
            translations = {einsum_name2: {} for einsum_name2 in self.einsum_names}
            for einsum_name2, ranks2 in einsum2ranks.items():
                for rank in ranks:
                    equiv = self.full_equivalent_ranks[rank] & ranks2
                    translations[einsum_name2][rank] = equiv
            rank_translations[einsum_name] = {
                k: {k2: list(v2) for k2, v2 in v.items()}
                for k, v in translations.items()
            }
        return rank_translations

    def _create_full_equivalent_ranks(
        self, pairwise_equivalent_ranks: dict[str, set[str]]
    ):
        full_equivalent_ranks = {
            k: set(v) for k, v in pairwise_equivalent_ranks.items()
        }
        changed = True
        while changed:
            changed = False
            for r in full_equivalent_ranks:
                for r2 in list(full_equivalent_ranks[r]):
                    for r3 in list(full_equivalent_ranks[r2]):
                        if r3 in full_equivalent_ranks[r]:
                            continue
                        changed = True
                        full_equivalent_ranks[r].add(r3)
        return full_equivalent_ranks

    def _create_einsum_rank_index_to_loops(
        self,
    ) -> dict[str, dict[str, dict[int, list[Loop]]]]:
        einsum_rank_index_to_loops = {}
        for einsum_name, sim_list in self.sims.items():
            einsum_rank_index_to_loops[einsum_name] = {}
            for sim in sim_list:
                for rank_index, loop in enumerate(sim.compatibility.loops):
                    x = einsum_rank_index_to_loops[einsum_name].setdefault(
                        loop.rank_variable_name, {}
                    )
                    x.setdefault(rank_index, []).append(loop)
        return einsum_rank_index_to_loops

    def get_tensors(self, *einsums: str):
        return set.union(*(self.einsum2tensors[e] for e in einsums))

    def get_possible_translations(self, t: Compatibility, to_einsum: str):
        pairwise_equivalent_ranks = self.pairwise_equivalent_ranks
        full_equivalent_ranks = self.full_equivalent_ranks
        right_ranks = self.einsum2ranks[to_einsum]

        def translate_loop(l: Loop):
            compatible_ranks = (
                set.union(*(full_equivalent_ranks[n] for n in l.rank_variable_names))
                & right_ranks
            )
            pairwise_compatible_ranks = (
                set.union(
                    *(pairwise_equivalent_ranks[n] for n in l.rank_variable_names)
                )
                & right_ranks
            )
            if len(pairwise_compatible_ranks) > 1:
                return
            for n in compatible_ranks:
                yield Loop(fzs((n,)), l.bound, l.is_spatial)

        for loops in itertools.product(*map(translate_loop, t.loops)):
            yield Compatibility(loops, t.tensors, t.tags)
