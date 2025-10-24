# Fast & Fusiest & More
This repository contains the code for the latest series of projects by Michael and
Tanner, including:

- Fast & Fusiest: Fast and Fusiest: A Fast and Optimal Fusing Mapper for Tensor Algebra
  Accelerators
- LoopForest: LoopForest: Exploring an Expanded Fusion Mapspace for Reduced Data
  Movement and Memory Usage

## Installation

### Native

Run the following:

```bash
# Clone and enter the repo
git clone https://github.com/Accelergy-Project/fastfusion.git
cd fastfusion

make install-hwcomponents  # Install hwcomponents component models
pip3 install -e .          # Install this package
```

### Docker

```bash
# Build the docker image
make build-docker

# Run the docker image
make run-docker

```

Running the Docker container will mount the current directory as `/home/workspace`. A
link to a Jupyterlab server will be posted to the terminal.

Some additional installation will occur when you run the container for the first time.
This is done inside the container so that you can update the package without rebuilding
the container. If the install fails, you can run the following from inside the container
to try again:
```bash
cd /home/workspace && pip install -e .
```

### Running the Examples

Examples can be found in the `notebooks/examples` directory.
