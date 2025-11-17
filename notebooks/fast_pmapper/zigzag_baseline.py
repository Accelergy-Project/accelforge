"""
The following code is adapted from main.py in the ZigZag repository.

To run the code, first install ZigZag with `pip3 install zigzag-dse`.
"""
from datetime import datetime
import time
import csv

from paths import DATA_DIR, ZIGZAG_MAPPING_DIR, ZIGZAG_ARCHITECTURE_DIR, ZIGZAG_WORKLOAD_DIR

from zigzag import api

ZIGZAG_ESTIMATED_MODEL_TPUT = 222 # evals/s

WORKLOAD_NAMES = [
    # "gemm_16k",
    "mha",
]

accelerator_path = "architectures/zigzag/tpu_like_no_llb.yaml"
mapping_path = "mapping/zigzag/tpu_custom.yaml"
experiment_id = datetime.now()
dump_folder = f"outputs/zigzag/{experiment_id}"
pickle_filename = f"outputs/zigzag/{experiment_id}/cmes.pickle"

def run_experiment(accelerator_fname, workload_name, max_lpf, target_evals):
    workload_fname = ZIGZAG_WORKLOAD_DIR / f"{workload_name}.yaml"
    result_fname = DATA_DIR / f"zigzag_{workload_name}.csv"
    mapping_fname = ZIGZAG_MAPPING_DIR / f"tpu_{workload_name}.yaml"

    print(f"Running experiments for {workload_name}...")
    print("Results in", result_fname)
    print("Using workload", workload_fname)
    print("Using mapping", mapping_fname)

    with open(result_fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['lfp', 'mapper_time', 'evaluations', 'energy', 'latency', 'edp'])

        for lpf_limit in range(4, max_lpf+1):
            start = time.time()
            energy, latency, cmes = api.get_hardware_performance_zigzag(
                workload=str(workload_fname),
                accelerator=accelerator_fname,
                mapping=str(mapping_fname),
                opt="EDP",
                dump_folder=dump_folder,
                pickle_filename=pickle_filename,
                lpf_limit=lpf_limit
            )
            end = time.time()
            duration = end - start
            print(f"lpf = {lpf_limit}")
            print(f"Time = {duration:.2e} s")
            print(f"Total energy = {energy:.2e} pJ")
            print(f"Total latency = {latency:.2e} cycles")

            estimated_evals = duration*ZIGZAG_ESTIMATED_MODEL_TPUT

            energy = energy/1e12 # pJ -> J
            latency = latency/1e9 # cycles -> seconds (assuming 1GHz)
            writer.writerow([lpf_limit,
                             duration,
                             estimated_evals,
                             energy,
                             latency,
                             energy*latency])

            if estimated_evals >= target_evals:
                break


for workload_name in WORKLOAD_NAMES:
    run_experiment("architectures/zigzag/tpu_like.yaml", workload_name, max_lpf=10, target_evals=16000)
# run_experiment("zigzag_gemm_16k.csv", "architectures/zigzag/tpu_like.yaml")
# run_experiment("zigzag_gemm_16k_no_llb.csv", "architectures/zigzag/tpu_like_no_llb.yaml")