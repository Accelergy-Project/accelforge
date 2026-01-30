# AccelForge

AccelForge is a framework to model and design tensor algebra accelerators. Key features
include:

- Flexible and user-defined specifications for components, architectures, and workloads,
  including a suite of example accelerators and deep neural networks.
- An easy-to-use Python API for specifying and manipulating input specifications.
- Novel mapping algorithms that maps workloads onto architectures optimally, and in
  orders-of-magnitude less time than prior approaches.

To learn more, see the [AccelForge
documentation](https://accelergy-project.github.io/accelforge/).

AccelForge uses [HardwareComponents](https://github.com/accelergy-project/hwcomponents)
as a backend to model the area, energy, latency, and leak power of hardware components.

## Installation

### Native

AccelForge is available on PyPI:

```bash
pip install accelforge
```

### Docker

TODO

## Notebooks and Examples

Examples can be found in the [`notebooks`](notebooks) directory. Examples of the input
files can be found in the [`examples`](examples) directory.

TODO:
- Update doc with:
  - New mapper interface
  - Bits per value & other symbols available from the Einsum
  - Per-Einsum parsing
