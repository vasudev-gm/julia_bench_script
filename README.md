
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

## Usage

Install dependencies (first run):

ZSH:

```zsh
julia -e 'using Pkg; Pkg.add(["BenchmarkTools", "Metal"])'
```

OR

BASH:

```bash
julia -e 'using Pkg; Pkg.add(["BenchmarkTools", "Metal"])'
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

## Configuration

Modify the vector size in the script:

```julia
N = 2^30  # 1 billion elements
```

Modify the matrix multiplication size separately:

```julia
N = 16384
```

## Notes

- Uses 32-bit floats (`Float32`) for realistic memory bandwidth testing
- Uses `mean(trial).time` from BenchmarkTools and converts ns -> ms/s for reporting
- GPU benchmarks require Apple Silicon plus a working Metal.jl setup
- Results vary based on thermal conditions and system load
- Modify the size of N depending on available memory as swap usage decreases overall performance

[def]: https://www.youtube.com/watch?v=wV7bK8IhUn4
