"""
The following code is adapted from main.py in the ZigZag repository.

To run the code, first install ZigZag with `pip3 install zigzag-dse`.
"""
from datetime import datetime
import time

from zigzag import api

workload_path = "workloads/zigzag/gemm_16k.yaml"
accelerator_path = "architectures/zigzag/tpu_like.yaml"
mapping_path = "mapping/zigzag/tpu_custom.yaml"
experiment_id = datetime.now()
dump_folder = f"outputs/zigzag/{experiment_id}"
pickle_filename = f"outputs/zigzag/{experiment_id}/cmes.pickle"

import csv
with open('zigzag_gemm_16k.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['lfp', 'mapper_time', 'energy', 'latency'])

    for lpf_limit in range(4, 11):
        start = time.time()
        energy, latency, cmes = api.get_hardware_performance_zigzag(
            workload=workload_path,
            accelerator=accelerator_path,
            mapping=mapping_path,
            opt="energy",
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

        writer.writerow([lpf_limit, duration, energy/1e12, latency/1e9])