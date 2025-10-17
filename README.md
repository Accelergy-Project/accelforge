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

sudo make install-ntl      # Install NTL (ISL dependency)
sudo make install-barvinok # Install Barvinok (ISL dependency)
make install-islpy         # Install ISL
make install-hwcomponents  # Install hwcomponents
pip3 install -e .          # Install this package
```

### Docker

```bash
# Build the docker image
make build-docker

# Run the docker image
make run-docker

# Some additional installation will occur when you run the container for the first time.
# This is done inside the container so that you can update the package without rebuilding
# the container. If the install fails, you can run the following command to try again:
# FROM INSIDE THE CONTAINER: cd /home/workspace && pip install -e .

```

Running the Docker container will mount the current directory as `/home/workspace`. A
link to a Jupyterlab server will be posted to the terminal.

### Running the Examples

Examples can be found in the `notebooks/examples` directory.
