import matplotlib.pyplot as plt

from pytimeloop.looptree.reuse.isl.des import IslReuseAnalysisOutput


def plot_occupancy_graph(output: IslReuseAnalysisOutput, workload):
    einsum_rank_to_shape = {
        einsum: {
            rank: workload.get_rank_shape(rank)
            for rank in workload.einsum_ospace_dimensions(einsum)
        }
        for einsum in workload.einsum_id_to_name()
    }

    
