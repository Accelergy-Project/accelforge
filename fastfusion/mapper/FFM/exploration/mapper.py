"""
Key requirements for single-Einsum mapper:
- Fast. Ideally, ~10,000 mappings/s
- Provide multi-Einsum joining part with:
  - pre-grouped into SIMs
  - pre-Pareto pruned
  single-Einsum mappings.

Consideration for convolution:
- TODO: redefine compatibility. E.g., the following are compatible but
  doesn't match exactly:
    ConvB inter-Einsum mapping PB, A in GLB, QB with tile shape (1, 1) 
    ConvA inter-Einsum mapping PA, PB, A in GLB with tile shape (shape 3, stride 1; shape 3, stride 1)
  TODO: unresolved semantics (see above). Good to discuss.
  - Who is responsible for keeping A? The backing for ConvA or ConvB?
  - How about residuals?

Key optimizations for speed:
- Caching of results for similar Einsums.
  Keep in mind: must define clearly what "similar" is in the code. E.g.,
  - Einsums with matching tile shape but with different constraints
    should not be deemed similar.
- Precompilation of models just before tile shape exploration.
  Note: I like this approach because the model code is easy to understand
- Vectorization:
  - Vectorization of model call during tile shape exploration.
    Note: quite nicely compatible with precompilation
  - Vectorization of parts of process results (see "Stuff to change" below)
- Reuse results (Python code is slow, avoid extraneous work)
  - Large parts of the SIMs can be pre-computed because they are the same for
    all mappings with only tile shape difference.

Design changes from last prototype:
- process_results is *slow* and conceptually unclear. E.g.,
  - Does process_results compute energy (part of model) or generate SIMs (data post-processing)
  Thus, we should:
  - TODO: redesign process_result into several functions
  - TODO: take energy and memory_latency computation into model.
  - TODO: what is pre-compiled is *reuse analysis*.
  - TODO: define what the data post-processing for SIMs really need as inputs and outputs.
  - TODO: optimize SIM post-processing with reuse and vectorization.
"""


def mapper(
    config,
    spec,
    ert,
    metrics,
):
    # Group similar Einsums

    # Single-Einsum exploration

    # Regenerate data for each Einsum
    pass


def single_einsum_exploration(
    subspaces,
    config,
    bindings,
):
    # Generate subspaces except for tile shape exploration

    # Parallelize tile shape exploration

    # Pareto prune each group (SIM)
    pass


def single_einsum_tile_shape_exploration():
    # Compile mapping into lambdas

    # Generate tile shapes as numpy arrays

    # Call lambdas with possible tile shapes
    pass