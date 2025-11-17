import pytimeloop.timeloopfe.v4 as tl
import time
import csv

from paths import DATA_DIR, TIMELOOP_CONFIG_DIR, TIMELOOP_WORKLOAD_DIR

NUM_THREADS = 12

WORKLOAD_NAMES = [
    "gemm",
    "mha",
]

def evaluate(workload_name, evaluated_sizes, use_hint=False):
    if use_hint:
        config_fname = TIMELOOP_CONFIG_DIR / f"{workload_name}_hint.yaml"
    else:
        config_fname = TIMELOOP_CONFIG_DIR / f"{workload_name}.yaml"
    workload_fname = TIMELOOP_WORKLOAD_DIR / f"{workload_name}.yaml"
    result_fname = DATA_DIR / f"timeloop_{workload_name}{'_hint' if use_hint else ''}.csv"

    spec = tl.Specification.from_yaml_files(
        config_fname,
        workload_fname,
        "ert/tpu_like.yaml"
    )
    spec.mapper.num_threads = NUM_THREADS
    with open(result_fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['evaluations', 'threads', 'mapper_time', 'energy', 'latency', 'edp'])
        for evaluated_size in evaluated_sizes:
            spec.mapper.evaluated_size = evaluated_size
            output = None
            try:
                start = time.time()
                output = tl.call_mapper(spec, output_dir="outputs/timeloop")
                end = time.time()
            except:
                pass
            mapper_time = end - start
            if output is not None:
                writer.writerow([evaluated_size, NUM_THREADS, mapper_time*NUM_THREADS, output.energy, output.latency, output.energy*output.latency])
            else:
                writer.writerow([evaluated_size, NUM_THREADS, mapper_time*NUM_THREADS, None, None, None])

evaluate('mha', [10000, 25000, 50000], use_hint=False)
evaluate('mha', [10000, 25000, 50000], use_hint=True)

# evaluate('timeloop_gemm_16k_hint.csv',
#          'configs/timeloop_gemm_tpu_hint.yaml',
#          [50000, 100000, 150000, 200000])
