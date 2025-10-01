import pytimeloop.timeloopfe.v4 as tl
import time
import csv

from .paths import DATA_DIR

NUM_THREADS = 12

def evaluate(result_fname, config_fname, search_sizes):
    result_fname = DATA_DIR / result_fname
    spec = tl.Specification.from_yaml_files(
        config_fname,
        "ert/tpu_like.yaml"
    )
    spec.mapper.num_threads = NUM_THREADS
    with open(result_fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['search_size', 'threads', 'mapper_time', 'energy', 'latency'])
        for search_size in search_sizes:
            start = time.time() 
            spec.mapper.search_size = search_size
            output = tl.call_mapper(spec, output_dir="outputs/timeloop")
            end = time.time()
            mapper_time = end - start
            writer.writerow([search_size, NUM_THREADS, mapper_time, output.energy, output.latency])

# Without hint, finding valid mappings is easier and takes faster so search size should be larger here
evaluate('timeloop_gemm_16k.csv',
         'configs/timeloop_gemm_tpu.yaml',
         [4000, 8000, 16000, 32000, 64000, 128000])

evaluate('timeloop_gemm_16k_hint.csv',
         'configs/timeloop_gemm_tpu_hint.yaml',
         [500, 1000, 2000, 4000, 8000])
