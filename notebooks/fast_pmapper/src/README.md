# Sunstone Scheduler Source Files

This repository contains the source files for the core algorithms used in the Sunstone scheduler. These algorithms are based on the main Sunstone mapping principles discussed in the [research paper](https://people.ece.ubc.ca/sasha/papers/ispass2023.pdf).

## Tiling

The `tiling.py` file provides a Python class that can represent tiles and dataflow, as well as useful methods for managing them.

## Tile Graph

The `tile_graph.py` file provides algorithms for building the search trees discussed in the paper and picking the best spatial and temporal tile candidates.

## Generic Optimizers and Optimization Utilities

The `generic_optimizers.py` and `optimization_utils.py` files contain algorithms for optimizing the tiling configuration of a specific level using the tile graph search methods.