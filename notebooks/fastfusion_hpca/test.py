# Imports
import fastfusion as ff
from IPython.display import SVG, display
import os
ff.set_n_parallel_jobs(os.cpu_count(), print_message=True)

for BATCH_SIZE in [4, 16, 64]:
    for N_TOKENS in [4096, 8192, 16384, 32768, 65536]:
        FUSE = True
        
        print(f'\n\n\n')
        for i in range(3):
            print(f'=' * 100)
        print(f"Running with BATCH_SIZE={BATCH_SIZE} and N_TOKENS={N_TOKENS}")
        for i in range(3):
            print(f'=' * 100)

        spec = ff.Spec.from_yaml(
            "arches/tpu_v4i_like.yaml",
            # "arches/tpu_v4i_like_constrained.yaml",
            # "arches/simple.arch.yaml",
            "workloads/gpt3_6.7B.yaml",
            jinja_parse_data=dict(
                BATCH_SIZE=BATCH_SIZE,
                N_TOKENS=N_TOKENS,
            )
        )

        # WARNING: tpu_v4i_like is pretty constrained

        # If fusion is disabled, keep all tensors in main memory.
        if not FUSE:
            spec.arch.nodes["MainMemory"].constraints.tensors.keep = "All()"

        # display(SVG(spec.workload.render()))
        
        pmappings = ff.mapper.FFM.make_pmappings(
            spec,
            can_combine_multiple_runs=False,
            cache_dir="cache",
        )
