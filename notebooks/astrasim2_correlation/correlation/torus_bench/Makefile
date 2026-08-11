CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?=
NVCC ?= $(CUDA_HOME)/bin/nvcc
GXX ?= g++
ARCH ?= -arch=sm_90

.PHONY: clean

NCCL_INC := $(if $(NCCL_HOME),-I$(NCCL_HOME)/include)
NCCL_LIB := $(if $(NCCL_HOME),-L$(NCCL_HOME)/lib)

# Default target (first in this file): the GPU/NCCL executor. Requires nvcc plus a CUDA
# toolkit and NCCL headers/libs; NOT buildable in this development environment (no nvcc here
# -- see the work-package report). Provided so the GPU target compiles cleanly, by construction,
# on the target AWS p5.48xlarge instance, where CUDA_HOME/NCCL_HOME should be set as needed.
torus_bench: torus_bench.cu
	$(NVCC) -O3 $(ARCH) $(NCCL_INC) $(NCCL_LIB) -o $@ $< -lnccl

# Simulator target: forces g++ to treat the .cu file as plain C++ (-x c++), not CUDA source.
# This is the local, GPU-free test vehicle for the schedule-building, edge-assertion, and
# per-collective reduction/routing logic -- see the file header of torus_bench.cu for the full
# schedule-as-data rationale. This is the target exercised by the acceptance criteria.
torus_bench_sim: torus_bench.cu
	$(GXX) -x c++ -std=c++17 -O2 -Wall -Wextra -DTORUS_SIM -o $@ $<

clean:
	rm -f torus_bench torus_bench_sim
