import copy
import logging
import os
import time

logging.basicConfig(level=logging.WARN)

from fastfusion.accelerated_imports import np
from fastfusion.mapper.FFM._join_pmappings.sim import PmappingGroup, TensorReservation
from tests.util import TEST_TMP_DIR

import logging
from pathlib import Path

from bindings.config import Config

import pickle

from fastfusion.mapper.simanneal.simanneal import mapper
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import join_pmappings
from fastfusion.visualization.ski_slope import plot_ski_slope
from fastfusion.mapper.simanneal.process_results import Metrics
from pytimeloop.frontend.v4fused import Specification
from pytimeloop.fastfusion.filter_mappings import get_tileflow_tag_mha, get_ffmt_tag_mha, get_layernorm_tag_mha, get_looptree_tag_mha, get_optimus_tag

from tests.util import CONFIG_DIR, TEST_TMP_DIR

from fastfusion.accelerated_imports import pd

RESULTS_DIR = f"workloads/results"

def get_results_dir(name: str):
    d = Path(RESULTS_DIR) / name
    d.mkdir(parents=True, exist_ok=True)
    return d

class EvaluationsScoreTracker():
    def __init__(
        self,
        max_evaluations: int,
        stop_at_score: float,
        print_period: int = 10
    ):
        self.max_evaluations = max_evaluations
        self.stop_at_score = stop_at_score
        self.evaluations = 0
        self.score = float("inf")
        self.history = [(0, float("inf"))]
        self._scale_by = 1
        self.print_period = print_period
        self.prev_print_time = None
        self.print_stopped_text = False
        self.n_mappings = {}
        self.runtime = {}

    def add_evaluation(self, n_evaluations: int, best_score: float):
        self.evaluations += n_evaluations * self._scale_by
        self.score = min(self.score, best_score)
        # Same score as before, remove the last entry
        if len(self.history) > 2 and self.history[-2][1] == self.score:
            self.history.pop(-1)
        self.history.append((self.evaluations, self.score))

        cur_time = time.time()
        if self.prev_print_time is None or cur_time - self.prev_print_time > self.print_period:
            self.prev_print_time = cur_time
            print(f"Evaluations: {self.evaluations}, Score: {self.score}")

        if self.max_evaluations is not None and self.evaluations > self.max_evaluations:
            self.clean_history()
            if not self.print_stopped_text:
                print(f'Stopping due to evaluations {self.evaluations} > {self.max_evaluations}')
                self.print_stopped_text = True
            return True
        if self.stop_at_score is not None and self.score < self.stop_at_score:
            self.clean_history()
            if not self.print_stopped_text:
                print(f'Stopping due to score {self.score} < {self.stop_at_score}')
                self.print_stopped_text = True
            return True
        return False

    def multiply_scale_by(self, scale_by: float):
        self._scale_by *= scale_by

    def __repr__(self):
        return f"Evaluations: {self.evaluations}, Score: {self.score}"

    def __str__(self):
        return f"Evaluations: {self.evaluations}, Score: {self.score}"

    def clean_history(self):
        keep_indices = [0]
        for i in range(1, len(self.history) - 1):
            if self.history[i][1] != self.history[i-1][1] or self.history[i][1] != self.history[i+1][1]:
                keep_indices.append(i)
        keep_indices.append(len(self.history)-1)
        self.history = [self.history[i] for i in keep_indices]

    def merge_with(self, other: "EvaluationsScoreTracker"):
        self.score = min(self.score, other.score)
        self.evaluations += other.evaluations

        i, j = 1, 1
        history = [(0, float("inf"))]
        cur_score = float("inf")
        cur_evaluations = 0
        while i < len(self.history) or j < len(other.history):
            # Grab whichever has the lowest evaluations
            if i < len(self.history) and  (j == len(other.history) or self.history[i][0] < other.history[j][0]):
                new_evaluations = self.history[i][0] - self.history[i-1][0]
                new_score = self.history[i][1]
                cur_evaluations += new_evaluations
                cur_score = min(cur_score, new_score)
                history.append((cur_evaluations, cur_score))
                i += 1
            elif j < len(other.history):
                new_evaluations = other.history[j][0] - other.history[j-1][0]
                new_score = other.history[j][1]
                cur_evaluations += new_evaluations
                cur_score = min(cur_score, new_score)
                history.append((cur_evaluations, cur_score))
                j += 1
        self.history = history
        self.clean_history()

    def increase_all_evaluations(self, n_evaluations: int):
        self.evaluations += n_evaluations
        self.history = [(e + n_evaluations, s) for e, s in self.history]

class Experiment:
    def __init__(self,
                 name: str,
                 workload_fname: str,
                 prune_intra: bool = True,
                 taggers: tuple[callable] = tuple(),
                 dataflow: str = None,
                 fuse: bool = True,
                 size_scale: float = 1.0,
        ):
        self.name = name
        self.workload_fname = workload_fname
        self.constraint_config = ""
        with open(workload_fname, "r", encoding="utf-8") as f:
            self.workload_template = f.read()
        self.shape = {}
        self.equiv_ranks = None
        self.einsum2ranks = None
        self.prune_intra = prune_intra
        self.taggers = taggers
        self.dataflow = dataflow
        self.fuse = fuse
        self.intra_runtime = None
        self.inter_runtime = None
        self.size_scale = size_scale

    def configure_workload_shape(self, **kwargs):
        self.shape = kwargs
        self.workload_config = self.workload_template.format(**kwargs)

    def configure_arch(self, arch_fname: str | Path):
        with open(arch_fname, "r") as f:
            self.arch_config = f.read()

    def configure_constraints(self, constraint_fname: str | Path):
        constraint_fname = CONFIG_DIR / constraint_fname
        with open(constraint_fname, "r") as f:
            self.constraint_config = f.read()

    @property
    def workload_name(self):
        return os.path.basename(self.workload_fname).split(".")[0]

    @property
    def full_name(self):
        return " ".join(
            str(s)
            for s in [
                "".join(f"{rank}{shape}" for rank, shape in self.shape.items()),
                "pruned" if self.prune_intra else "unpruned",
                f"dflow {self.dataflow}" if self.dataflow else "dflow all",
                "fused" if self.fuse else "unfused",
                " ".join(t.__name__ for t in self.taggers) if self.taggers else "",
            ]
        )

    @property
    def intra_results_file(self):
        return get_results_dir(self.workload_name) / f"{self.full_name}.intra.pkl"

    @property
    def inter_results_file(self):
        return get_results_dir(self.workload_name) / f"{self.full_name} {self.name}.inter.pkl"

    def get_pkl_attributes_intra(self):
        return ["intra_result", "equiv_ranks", "einsum2ranks", "bindings", "max_fanout", "max_capacity", "n_mappings_intra", "intra_runtime"]

    def save_intra_results(self):
        with open(self.intra_results_file, "wb") as f:
            pickle.dump(tuple(getattr(self, attr) for attr in self.get_pkl_attributes_intra()), f)

    def load_intra_results(self):
        with open(self.intra_results_file, "rb") as f:
            for attr, val in zip(self.get_pkl_attributes_intra(), pickle.load(f)):
                setattr(self, attr, val)

    def save_inter_result(self):
        # self.inter_result.to_csv(f'{self.workload_fname}.csv')
        with open(self.inter_results_file, "wb") as f:
            pickle.dump((self.inter_result, self.n_mappings_intra, self.n_mappings_inter, self.intra_runtime, self.inter_runtime), f)

    def load_inter_result(self):
        loaded = list(pickle.load(open(self.inter_results_file, "rb")))
        self.inter_result = loaded.pop(0)
        self.n_mappings_intra = loaded.pop(0)
        self.n_mappings_inter = loaded.pop(0) if loaded else None
        self.intra_runtime = loaded.pop(0) if loaded else None
        self.inter_runtime = loaded.pop(0) if loaded else None

    @property
    def total_mappings_from_intra(self):
        return sum(len(s.mappings.data) for pmapping_groups in self.intra_result.values() for s in pmapping_groups)

    def run_intra(self, tag=True, prune=True, dataflow=None):
        config_str = self.workload_config + "\n" + self.arch_config + "\n"
        config = Config(config_str, "yaml")

        config_str += self.constraint_config
        with open(TEST_TMP_DIR / "tmp.yaml", "w") as f:
            f.write(config_str)
        spec = Specification.from_yaml_files([TEST_TMP_DIR / "tmp.yaml"])

        metrics = Metrics.ENERGY | Metrics.LATENCY
        if not prune:
            metrics |= Metrics.VALID

        t0 = time.time()
        self.intra_result, self.equiv_ranks, self.einsum2ranks, self.bindings, self.max_fanout, self.max_capacity, self.n_mappings_intra = mapper(
            config,
            explore_glb_uneven=True,
            spec=spec,
            tmp_path=TEST_TMP_DIR,
            metrics=metrics,  # | Metrics.DEBUG,
            tag_with=self.taggers,#(get_tileflow_tag_mha, get_ffmt_tag_mha, get_layernorm_tag_mha, get_looptree_tag_mha),
            four_level=True,
            prune=prune,
            dataflow=self.dataflow,
            fuse=self.fuse,
        )
        t1 = time.time()
        self.intra_runtime = t1 - t0
        print(f"Intra-layer exploration took {self.intra_runtime} seconds")
        self.bindings = {str(k): v for k, v in self.bindings.items()}
        self.max_fanout = {str(k): v for k, v in self.max_fanout.items()}
        self.max_capacity = {str(k): v for k, v in self.max_capacity.items()}

    def run_fusion(self, fuse_function, max_evaluations=None, stop_at_score=None, count_intra_evaluations: bool=False, **kwargs):
        evaluations_tracker = EvaluationsScoreTracker(
            max_evaluations=max_evaluations,
            stop_at_score=stop_at_score,
        )
        resource2capacity = {
            f"{k}_{self.bindings[k]}": v
            for k, v in self.max_capacity.items()
        }

        t0 = time.time()
        self.inter_result = fuse_function(
            self.intra_result,
            resource2capacity=resource2capacity,
            pairwise_equivalent_ranks=self.equiv_ranks,
            einsum2ranks=self.einsum2ranks,
            evaluations_tracker=evaluations_tracker,
            size_scale=self.size_scale,
            **kwargs,
        )
        t1 = time.time()
        self.inter_runtime = t1 - t0
        print(f"Inter-layer exploration took {self.inter_runtime} seconds")
        if count_intra_evaluations:
            evaluations_tracker.increase_all_evaluations(self.n_mappings_intra)
            evaluations_tracker.n_mappings = {
                "Intra-Layer": self.n_mappings_intra,
                **evaluations_tracker.n_mappings,
            }
            evaluations_tracker.runtime = {
                "Intra-Layer": self.intra_runtime,
                **evaluations_tracker.runtime,
            }
        self.n_mappings_inter = evaluations_tracker
        print(evaluations_tracker)
        # self.inter_result = _free_to_loop_index(self.inter_result, -1)
        # self.inter_result["Occupancy"] = self.inter_result[nameloop2col(1, -1)]
        # self.inter_result = paretofy_by(self.inter_result, ["Occupancy", "Offchip Accesses"])

    def plot_ski_slope(self, **kwargs):
        fig, ax = plot_ski_slope(self.inter_result, **kwargs)
        fig.tight_layout()
        return fig, ax


def clear_tags(e):
    for pmapping_groups in e.intra_result.values():
        for s in pmapping_groups:
            s.set_tags()


def filter_first_mapping(e):
    e.intra_result.pop(next(iter(e.intra_result)))
    for k, pmapping_groups in e.intra_result.items():
        e.intra_result[k] = [
            s
            for s in pmapping_groups
            if not s.compatibility.has_tensor(TensorReservation("I_I_to_Q_K_V", "*", 1, "*"))
        ]


def filter_tensors(e, tensors_filter):
    new_intra = {}
    for k, pmapping_groups in e.intra_result.items():
        new_intra[k] = PmappingGroup.filter_by_tensors(pmapping_groups, tensors_filter)
        if not new_intra[k]:
            raise ValueError(f"No mappings for {k} with memory filter {tensors_filter}")
    e.intra_result = new_intra

def filter_layernorm(e):
    for k, pmapping_groups in e.intra_result.items():
        e.intra_result[k] = [s for s in pmapping_groups if "LAYERNORM_INVALID" not in s.compatibility.tags]
        for s in e.intra_result[k]:
            s.set_tags(*(t for t in s.compatibility.tags if "LAYERNORM" not in t))


def tileflow(e):
    filter_layernorm(e)
    for k, pmapping_groups in e.intra_result.items():
        # print(f"Mappings for Einsum {k}")
        e.intra_result[k] = [s for s in pmapping_groups if "TILEFLOW_VALID" in s.compatibility.tags]
        for s in e.intra_result[k]:
            s.set_tags(*(t for t in s.compatibility.tags if "LOOPS_ABOVE_GLB" in t))
            # print(f"\t{s.compatibility}")

def looptree(e):
    filter_layernorm(e)
    for k, pmapping_groups in e.intra_result.items():
        # print(f"Mappings for Einsum {k}")
        e.intra_result[k] = [s for s in pmapping_groups if "LOOPTREE_VALID" in s.compatibility.tags]
        for s in e.intra_result[k]:
            s.set_tags(*(t for t in s.compatibility.tags if "FUSED_LOOPS" in t))
            # print(f"\t{s.compatibility}")

def rearrange_wrap(order, f):
    def wrapped(e):
        e.intra_result = {k: e.intra_result[k] for k in order}
        return f(e)

    return wrapped


def fastfusion_minus_first_input(e):
    filter_layernorm(e)
    filter_first_mapping(e)
    clear_tags(e)


def fastfusion_full(e):
    filter_layernorm(e)
    clear_tags(e)


def unfused(e):
    filter_layernorm(e)
    e.intra_result = {
        k: PmappingGroup.filter_by_tensors(pmapping_groups, {TensorReservation("*", "*", 0, "*")})
        for k, pmapping_groups in e.intra_result.items()
    }
    clear_tags(e)


def ffmt(e):
    filter_layernorm(e)
    if next(iter(e.intra_result)) == "I":
        filter_first_mapping(e)
    # FFMT only fuses Q, QK, AV, and Z
    WEIGHT_TAGS = ("FFMT_WEIGHT_TILED", "FFMT_WEIGHT_UNTILED")
    for k, pmapping_groups in e.intra_result.items():
        # print(f"Mappings for Einsum {k}")
        pmapping_groups = [
            s
            for s in pmapping_groups
            if "FFMT_VALID" in s.compatibility.tags
            and "FFMT_WEIGHTS_INVALID" not in s.compatibility.tags
        ]
        r = []
        for s in pmapping_groups:
            # The tag is actually a set of tags, all of which must match, so we
            # combine them into a frozenset in a tuple. In the tag class, we'll
            # get a frozenset(frozenset(tag0, tag1...)).
            tags = frozenset(t for t in s.compatibility.tags if t in WEIGHT_TAGS)
            assert len(tags) <= 1, "Only one weight tag should be present"
            tag = next(iter(tags)) if tags else None
            if tag is None:
                for t in WEIGHT_TAGS:
                    s.set_tags(t)
                    r.append(copy.deepcopy(s))
            else:
                s.set_tags(tag)
                r.append(s)
            # print(f'\t{s.compatibility}')
        e.intra_result[k] = r

def filter_optimus(e):
    filter_layernorm(e)
    for k, pmapping_groups in e.intra_result.items():
        # print(f"Mappings for Einsum {k}")
        e.intra_result[k] = [s for s in pmapping_groups if "OPTIMUS_VALID" in s.compatibility.tags]
        for s in e.intra_result[k]:
            s.set_tags(*(t for t in s.compatibility.tags if "OPTIMUS" in t))
    print(e)

def run_experiment(
    name: str,
    workload_name: str,
    shape: dict = None,
    load_intra: bool = True,
    load_inter: bool = True,
    tensors_filter: set[TensorReservation] = None,
    callfunction: callable = None,
    lookahead_filter: bool = True,
    save_results: bool = True,
    run_inter: bool = True,
    prune_intra: bool = True,
    fuse_function: callable = join_pmappings,
    taggers: tuple[callable] = tuple(),
    dataflow: str = None,
    fuse: bool = True,
    max_evaluations: int = None,
    stop_at_score: float = None,
    size_scale: float = 1.0,
    load_inter_or_fail: bool = False,
):
    exp = Experiment(
        name,
        f"workloads/{workload_name}.yaml",
        prune_intra=prune_intra,
        taggers=taggers,
        dataflow=dataflow,
        fuse=fuse,
        size_scale=size_scale,
    )
    if shape is not None:
        exp.configure_workload_shape(**shape)
    else:
        exp.workload_config = exp.workload_template
    exp.configure_arch("architecture/four_level.arch.yaml")

    try:
        if not load_inter:
            raise FileNotFoundError
        exp.load_inter_result()
        print(f"Loaded data for {name}")
        return exp
    except (FileNotFoundError, EOFError):
        if load_inter_or_fail:
            raise FileNotFoundError(f"Could not load inter results for {name}")

    try:
        if not load_intra:
            raise FileNotFoundError
        exp.load_intra_results()
    except (FileNotFoundError, EOFError):
        t0 = time.time()
        exp.run_intra(prune=prune_intra, dataflow=dataflow)
        print(f'Intra-layer exploration took {time.time() - t0} seconds')
        if save_results:
            exp.save_intra_results()

    if not run_inter:
        return exp

    if tensors_filter is not None:
        filter_tensors(exp, tensors_filter)

    if callfunction is not None:
        callfunction(exp)

    for k, pmapping_groups in exp.intra_result.items():
        live_tensors = list(pmapping_groups[0].tensors)
        exp.intra_result[k] = PmappingGroup.combine_combineable(pmapping_groups, live_tensors)

    t0 = time.time()
    exp.run_fusion(
        lookahead_filter=lookahead_filter,
        fuse_function=fuse_function,
        max_evaluations=max_evaluations,
        stop_at_score=stop_at_score,
        count_intra_evaluations=prune_intra
    )
    print(f'Inter-layer exploration took {time.time() - t0} seconds')
    if save_results:
        exp.save_inter_result()

    del exp.intra_result # Free up memory by deleting intra-layer results

    return exp


def get_average_percent_reduction(a: pd.DataFrame, b: pd.DataFrame, x: str, y: str):
    a = a.sort_values(x).reset_index(drop=True)
    b = b.sort_values(x).reset_index(drop=True)
    ax, ay = a[x], a[y]
    bx, by = b[x], b[y]

    start = min(ax.min(), bx.min())
    end = max(ax.max(), bx.max())

    n_points = 10000
    x = np.linspace(start, end, n_points)
    count = 0
    total = 0
    max_reduction = 0
    points = []
    # print(f'Min x: {start} Max x: {end}')
    for i in range(n_points):
        # a_i is the highest index in a that is less than x[i]
        a_i = np.searchsorted(ax, x[i], side='right') - 1
        b_i = np.searchsorted(bx, x[i], side='right') - 1

        assert a_i != -1 and b_i != -1, f"One of the mappings is valid at a lower occupancy than the other"

        if a_i is None or b_i is None:
            continue
        # total += (by[b_i] / ay[a_i])# / by[b_i]
        points.append(by[b_i] / ay[a_i])
        max_reduction = max(max_reduction, by[b_i] / ay[a_i])
        count += 1

    def geomean(points):
        excess = 0
        counter = 1
        for p in points:
            counter *= p
            while counter > 2:
                excess += 1
                counter /= 2
            while counter < 0.5:
                excess -= 1
                counter *= 2
        excess = excess / len(points)
        return counter ** (1 / len(points)) * (2 ** excess)
    return geomean(points), max_reduction


def pad_to_max(results, x, y):
    max_x = max(r[x].max() for r in results.values())
    for k, r in results.items():
        if r[x].max() == max_x:
            continue
        min_y = r[y].min()
        # r1 = r1.append({x: max_x, y: min_y}, ignore_index=True)
        new_df = pd.DataFrame({x: [max_x], y: [min_y]})
        results[k] = pd.concat([r, new_df], ignore_index=True)
    return results