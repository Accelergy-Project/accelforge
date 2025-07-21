import os
from pathlib import Path
import pickle
from fastfusion import Specification
from fastfusion.mapper.FFM.ffm import make_pmappings, join_pmappings
from fastfusion.mapper.FFM.exploration.mapping_filter_tags import get_one_split_tag, get_ffmt_tag
from fastfusion.mapper.metrics import Metrics


ARCH_DIR          = Path('architecture/')
WORKLOAD_DIR      = Path('workloads/')
MAPPINGS_SIMS_DIR = Path('results/sims/')
MAPPINGS_DATA_DIR = Path('results/data/')


REFRESH_BOTH = {'SIMS', 'DATA'}
REFRESH_SIMS = {'SIMS'}
REFRESH_DATA = {'DATA'}


def one_split_tagger(compatibility):
    return get_one_split_tag(compatibility, "MainMemory")


NAME_TO_TAGGER = {
    'one_split': one_split_tagger
}


def get_experiment_name(tagger_name, arch_name: str, workload_name: str):
    return f'{workload_name}.{arch_name}.{tagger_name}'


def get_sims_with_cache(tagger_name=None,
                        refresh_cache=False,
                        arch_name: str='snowcat',
                        workload_name: str='matmuls8_mixed'):
    data_name  = get_experiment_name(tagger_name, arch_name, workload_name)
    result_pickle_name = MAPPINGS_DATA_DIR / f'{data_name}.pkl'
    if result_pickle_name.is_file() and not refresh_cache:
        with open(result_pickle_name, 'rb') as f:
            mappings = pickle.load(f)
            print(f'Loaded final results from cache {result_pickle_name}')
            return mappings

    if tagger_name is None:
        tagger = None
    else:
        tagger = NAME_TO_TAGGER[tagger_name]

    spec = Specification.from_yaml(
        ARCH_DIR / f'{arch_name}.arch.yaml',
        WORKLOAD_DIR / f'{workload_name}.workload.yaml'
    )
    spec.estimate_energy_area()
    workload = spec.workload
    renames = spec.renames
    flattened_architecture = spec.get_flattened_architecture()

    sims_name = get_experiment_name(tagger_name, arch_name, workload_name)
    pmappings_pickle_name = MAPPINGS_SIMS_DIR / f'{sims_name}.pmappings.pkl'
    if pmappings_pickle_name.is_file() and not refresh_cache:
        with open(pmappings_pickle_name, 'rb') as f:
            pmappings = pickle.load(f)
            print(f'Loaded pmappings from {pmappings_pickle_name}')
    else:
        pmappings = make_pmappings(spec)

    with open(pmappings_pickle_name, 'wb') as f:
        pickle.dump(pmappings, f)

    mappings = join_pmappings(spec, pmappings)

    with open(result_pickle_name, 'wb') as f:
        pickle.dump(mappings, f)
        print(f'Saved results to cache {result_pickle_name}')

    return mappings