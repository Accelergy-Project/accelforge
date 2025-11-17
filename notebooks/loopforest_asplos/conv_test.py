import fastfusion
from fastfusion.mapper import Metrics
import time

from paths import ARCH_DIR, WORKLOADS_DIR

from fastfusion.util import set_n_parallel_jobs

set_n_parallel_jobs(12)

def run_make_pmappings(arch_fname, workload_fname):
    spec = fastfusion.Specification.from_yaml(
        ARCH_DIR / arch_fname,
        WORKLOADS_DIR / workload_fname,
    )
    spec.mapper.ffm.metrics = Metrics.ENERGY | Metrics.RESOURCE_USAGE
    # spec.mapper.ffm.max_fused_loops = 3
    # spec.mapper.ffm.max_fused_loops_per_rank_variable = 1
    start = time.time()
    pmappings = fastfusion.mapper.FFM.make_pmappings(spec)
    end = time.time()
    duration = end - start
    return pmappings, duration

def run_join_pmappings(arch_fname, workload_fname, pmappings):
    spec = fastfusion.Specification.from_yaml(
        ARCH_DIR / arch_fname,
        WORKLOADS_DIR / workload_fname,
    )
    spec.mapper.ffm.metrics = Metrics.ENERGY | Metrics.RESOURCE_USAGE
    # spec.mapper.ffm.max_fused_loops = 3
    # spec.mapper.ffm.max_fused_loops_per_rank_variable = 1
    start = time.time()
    mappings = fastfusion.mapper.FFM.join_pmappings(spec, pmappings)
    end = time.time()
    duration = end - start
    return mappings, duration

pmappings, make_pmappings_duration = run_make_pmappings("snowcat_conv_test.arch.yaml", "conv_test.yaml")