# CUDA Fragment GEMM Solver

This project implements a high-performance matrix-vector multiplication pipeline for fragment-based GEMM using CUDA, cuBLAS, and CCCL (Thrust + libcudacxx).

## Features

- Multi-stream overlap of gather, GEMM, and scatter
- Modular solver backend (cuBLAS, CUTLASS, etc.)
- Host-resident global matrix with efficient GPU fragment execution

## Build

```bash
mkdir build && cd build
cmake ..
make -j
