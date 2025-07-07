# cuda_fragment_solver

**cuda_fragment_solver** is a high-performance CUDA-based solver designed to accelerate matrix fragment computations by leveraging batched GEMM operations, multi-threaded stream management, and advanced memory reuse strategies. It is optimized for workloads involving large batches of small-to-medium matrix operations, typical in scientific simulation, imaging, or fragment-based modeling applications.

---

## 🚀 Features

- **Single-task and batched-task support** for matrix fragment GEMMs.
- **Thread-to-stream binding** for concurrent execution.
- **Asynchronous memory allocation** using `cudaMallocAsync`.
- **cuBLAS-based GEMM and GEMM-batched kernels** for performance-critical compute.
- **Memory pooling** via `GPUWorkspace` and `GPUBatchWorkspace` for reuse and minimal allocations.
- **Flexible task scheduling** via `StreamThreadPool` and `BatchStreamThreadPool`.
- **Automatic workspace resizing** and efficient buffer reuse.

---

## 🧠 Algorithm Overview

### 1. Problem Structure

Each task solves a linear system of the form:

```
C = A × B
```

with optional low-rank decomposition:

```
A = Q × R
```

where:

- `Q`: shape (M, R)
- `R`: shape (R, K)
- `B`: shape (K, N)
- `C`: shape (M, N)

Depending on the rank and decomposition, the solver may perform one or two GEMMs per task.

### 2. Execution Modes

- **Single-task mode (`GPUTask`)**:  
  Each task is executed independently, performing one or two GEMMs (for QR-based solve).
- **Batched-task mode (`GPUBatchTask`)**:  
  Groups similar tasks (with the same padded `(M, K, N)`) for `cublasCgemmBatched`, maximizing GPU throughput.

### 3. Thread-Stream Binding

- Each worker thread is bound to a unique CUDA stream.
- Tasks are distributed to threads, which execute their assigned work on their dedicated stream.
- This is managed by:
  - `StreamThreadPool` (for `GPUTask`)
  - `BatchStreamThreadPool` (for `GPUBatchTask`)

### 4. Memory Management

- Each thread maintains its own workspace (`GPUWorkspace` or `GPUBatchWorkspace`).
- Temporary matrices (e.g., `localB`, `localC`, intermediates like `Q × (R × B)`) are reused and dynamically resized.
- Memory is allocated asynchronously and pooled for efficiency, minimizing allocation overhead.

---

## 🏗️ Implementation Details

### Core Components

| File                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `GEMMSolver.cu`         | Wraps single `cublasCgemm` with stream-bound execution.                     |
| `GEMMBatchSolver.cu`    | Wraps `cublasCgemmBatched` using double pointer arrays for batched compute. |
| `GPUTask.cu`            | Implements per-task logic for GEMM or QR-based solve.                       |
| `GPUBatchTask.cu`       | Implements batched version of `GPUTask`.                                    |
| `StreamThreadPool.cu`   | Manages multi-threaded `GPUTask` execution.                                 |
| `BatchStreamThreadPool.cu` | Manages grouped `GPUBatchTask` execution.                                |
| `GPUWorkspace.cu`       | Thread-local workspace with per-task memory buffers.                        |
| `GPUBatchWorkspace.cu`  | Batched layout and pointer indirection for group tasks.                     |
| `TaskManager.cu`        | Coordinates task creation, distribution, and launching.                     |

### Task Scheduling & Execution

- **TaskManager** splits the workload into tasks, grouping them by matrix shape for batching.
- **Thread pools** assign tasks to threads, each with its own CUDA stream and workspace.
- **Batched execution** uses pointer indirection and contiguous memory layouts for efficient `cublasCgemmBatched` calls.

### Memory Pooling

- **GPUWorkspace**: Allocates and reuses device buffers for each thread.
- **GPUBatchWorkspace**: Allocates batched buffers and manages pointer arrays for batched GEMMs.
- **cudaMallocAsync**: Used for non-blocking allocations, reducing synchronization overhead.

### Error Handling & Synchronization

- All CUDA and cuBLAS calls are checked for errors.
- Streams are synchronized only when necessary, allowing for maximum concurrency.

---

## 📊 Example Workflow

1. **Input Preparation**:  
   User provides matrices `A`, `B`, and (optionally) low-rank factors `Q`, `R`.
2. **Task Creation**:  
   `TaskManager` splits the problem into tasks, grouping by shape for batching.
3. **Workspace Allocation**:  
   Each thread allocates or reuses device memory for its tasks.
4. **Computation**:  
   - For each task:
     - If low-rank: compute `tmp = R × B`, then `C = Q × tmp`.
     - Else: compute `C = A × B`.
   - Batched tasks use `cublasCgemmBatched` for groups of similar shapes.
5. **Result Collection**:  
   Results are copied back to host memory as needed.

---

## 🛠️ Build & Install

### Dependencies

- **CUDA Toolkit** ≥ 11.0 (cuBLAS required)
- **CMake** ≥ 3.18
- **C++17**

### Build Instructions

```bash
git clone https://github.com/<yourname>/cuda_fragment_solver.git
cd cuda_fragment_solver
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This will generate the executable at `./bin/cuda_fragment_solver`.

