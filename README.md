
# Apple SoC Benchmarking Suite

A Julia-based benchmarking tool to measure single-core and multi-core performance on Apple Silicon processors. The code is from [Petar Insights original YT video][def]

## Overview

This benchmark script evaluates:

- **Vector Addition (CPU)**: Single-core performance and memory bandwidth via broadcast addition
- **BLAS AXPY (CPU)**: Multi-core throughput using `axpy!`
- **Vector Addition (GPU)**: Metal-backed with sync `axpy!` on `MtlArray`
- **Matrix Multiplication (CPU/GPU)**: Peak compute throughput with `mul!`

## Requirements

- Julia 1.6+
- `BenchmarkTools.jl`
- `LinearAlgebra.jl`
- `Metal.jl` (for Apple Silicon GPU benchmarks)
- `Plots.jl` (for saving benchmark charts)

## Usage

Install dependencies (first run):

ZSH:

```zsh
julia -e 'using Pkg; Pkg.add(["BenchmarkTools", "Metal", "Plots"])'
```

OR

BASH:

```bash
julia -e 'using Pkg; Pkg.add(["BenchmarkTools", "Metal", "Plots"])'
```

Then run the benchmark:

```julia
julia apple_soc_test.jl
```

## Metrics

The script reports:

- **Execution time** (milliseconds), using BenchmarkTools trial mean
- **Memory bandwidth** (GB/s) for vector and AXPY kernels
- **Compute performance** (GFLOPS) for matrix multiplication
- **Saved plots** in `plots/` by default:
- `vector_add_cpu_ms.png`
- `axpy_cpu_ms.png`
- `axpy_gpu_ms.png`
- `matmul_cpu_ms.png`
- `matmul_gpu_ms.png`

## Configuration

Modify the vector size in the script:

```julia
N = 2^30  # 1 billion elements
```

Modify the matrix multiplication size separately:

```julia
N = 16384
```

Optional environment overrides:

```bash
VECTOR_N=67108864 MATMUL_N=4096 MATMUL_SECONDS=30 PLOT_DIR=plots julia apple_soc_test.jl
```

## Notes

- Uses 32-bit floats (`Float32`) for realistic memory bandwidth testing
- Uses `mean(trial).time` from BenchmarkTools and converts ns -> ms/s for reporting
- GPU benchmarks require Apple Silicon plus a working Metal.jl setup
- Results vary based on thermal conditions and system load
- Modify the size of N depending on available memory as swap usage decreases overall performance

## Benching Results on Mac Mini M4 Base Model

| Mac mini M4 16GB RAM/256GB SSD                                  |                            |                              |
|:---------------------------------------------------------------:|:--------------------------:|:----------------------------:|
|                                                                 | OBS Recording in Bacground | Virtually No Background apps |
| CPU SC Bandwidth \(in GB/s\)                                    | 80\.376                    | 96\.428                      |
| CPU MultiCore Bandwidth \(in GB/s\)                             | 96\.597                    | 106\.787                     |
| GPU Bandwidth \(in GB/s\)                                       | 94\.273                    | 104\.952                     |
| CPU Compute Throughput \(in GFLOP\)                             | 557\.227                   | 610\.144                     |
| GPU Compute Throughput FP32 \(in GFLOP\)                        | 1748\.57                   | 3245\.443                    |
| GPU Compute Throughput FP16 \(in GFLOP\)                        | 2187\.939                  | 3766\.512                    |
| GPU Compute Throughput FP16/FP32 \(Times Faster than baseline\) | 1\.251                     | 1\.161                       |

[def]: https://www.youtube.com/watch?v=wV7bK8IhUn4
