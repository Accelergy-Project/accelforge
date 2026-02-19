Citing AccelForge
=================

**Please cite all of the following papers if you use this work.** This work is the
combination of the following:

- **CiMLoop**: The architecture and component specification.
- **Fast & Fusiest**: The multi-Einsum mapper.
- **LoopTree**: The mapping specification.
- **LoopForest**: The mapspace specification.
- **Turbo-Charged**: The single-Einsum mapper (and an essential first step for Fast &
  Fusiest).

They are available as the following:

.. code-block:: latex

    \cite{cimloop, fast_fusiest, looptree, turbo_charged}

.. code-block:: bibtex

    @INPROCEEDINGS{cimloop,
    author={Andrulis, Tanner and Emer, Joel S. and Sze, Vivienne},
    booktitle={2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
    title={CiMLoop: A Flexible, Accurate, and Fast Compute-In-Memory Modeling Tool},
    year={2024},
    volume={},
    number={},
    pages={10-23},
    keywords={Performance evaluation;Accuracy;Computational modeling;Computer architecture;Artificial neural networks;In-memory computing;Data models;Compute-In-Memory;Processing-In-Memory;Analog;Deep Neural Networks;Systems;Hardware;Modeling;Open-Source},
    doi={10.1109/ISPASS61541.2024.00012}}

    @misc{fast_fusiest,
    title={Fast and Fusiest: An Optimal Fusion-Aware Mapper for Accelerator Modeling and Evaluation},
    author={Tanner Andrulis and Michael Gilbert and Vivienne Sze and Joel S. Emer},
    year={2026},
    eprint={2602.15166},
    archivePrefix={arXiv},
    primaryClass={cs.AR},
    url={https://arxiv.org/abs/2602.15166},
    }

    @INPROCEEDINGS{looptree,
    author={Gilbert, Michael and Wu, Yannan Nellie and Parashar, Angshuman and Sze, Vivienne and Emer, Joel S.},
    booktitle={2023 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
    title={LoopTree: Enabling Exploration of Fused-layer Dataflow Accelerators},
    year={2023},
    volume={},
    number={},
    pages={316-318},
    keywords={Deep learning;Analytical models;Systematics;Neural networks;Bandwidth;Software;Energy efficiency;analytical modeling;layer fusion;accelerators},
    doi={10.1109/ISPASS57527.2023.00038}}

    @misc{turbo_charged,
    title={The Turbo-Charged Mapper: Fast and Optimal Mapping for Accelerator Modeling and Evaluation},
    author={Michael Gilbert and Tanner Andrulis and Vivienne Sze and Joel S. Emer},
    year={2026},
    eprint={2602.15172},
    archivePrefix={arXiv},
    primaryClass={cs.AR},
    url={https://arxiv.org/abs/2602.15172},
    }

TODO: Add citation for LoopForest