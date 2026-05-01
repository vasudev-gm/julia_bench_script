using BenchmarkTools
using LinearAlgebra
using Metal
using Plots

const DEFAULT_VECTOR_N = 2^26
const DEFAULT_MATMUL_N = 4096
const DEFAULT_MATMUL_SECONDS = 30
const DEFAULT_ALPHA = 1.0f0
const DEFAULT_PLOT_DIR = "plots"

function env_int(name::String, default::Int)
    parsed = tryparse(Int, get(ENV, name, string(default)))
    return isnothing(parsed) ? default : parsed
end

function env_str(name::String, default::String)
    value = get(ENV, name, default)
    return isempty(strip(value)) ? default : value
end

function mean_time_stats(trial::BenchmarkTools.Trial)
    ns = mean(trial).time
    return (ns=ns, ms=ns / 1e6, s=ns / 1e9)
end

bandwidth_gbps(elements::Int, seconds::Float64; arrays_touched::Int=3, elbytes::Int=4) =
    ((elements * elbytes * arrays_touched) / seconds) / 1e9

matmul_gflops(n::Int, seconds::Float64) = (2.0 * n^3) / (seconds * 1e9)

function axpy_gpu_sync!(alpha::Float32, y_m, x_m)
    Metal.@sync axpy!(alpha, y_m, x_m)
    return nothing
end

function matmul_gpu_sync!(C_m, A_m, B_m)
    Metal.@sync mul!(C_m, A_m, B_m)
    return nothing
end

function save_trial_plot(trial::BenchmarkTools.Trial, title::String, out_file::String)
    times_ms = trial.times ./ 1e6
    p = histogram(
        times_ms,
        bins=:auto,
        xlabel="Sample Time (ms)",
        ylabel="Frequency",
        title=title,
        legend=false,
        color=:steelblue,
    )
    vline!(p, [mean(times_ms)], color=:red, linewidth=2)
    savefig(p, out_file)
end

function save_plots(results::NamedTuple, plot_dir::String)
    mkpath(plot_dir)

    save_trial_plot(results.vec_cpu.trial, "Vector Add CPU Trial Times", joinpath(plot_dir, "vector_add_cpu_ms.png"))
    save_trial_plot(results.axpy_cpu.trial, "AXPY CPU Trial Times", joinpath(plot_dir, "axpy_cpu_ms.png"))
    save_trial_plot(results.axpy_gpu.trial, "AXPY GPU Trial Times", joinpath(plot_dir, "axpy_gpu_ms.png"))
    save_trial_plot(results.mm_cpu.trial, "MatMul CPU Trial Times", joinpath(plot_dir, "matmul_cpu_ms.png"))
    save_trial_plot(results.mm_gpu.trial, "MatMul GPU Trial Times", joinpath(plot_dir, "matmul_gpu_ms.png"))
end

function benchmark_vector_add_cpu(vector_n::Int)
    x = rand(Float32, vector_n)
    y = rand(Float32, vector_n)

    y .+= x
    trial = @benchmark $y .+= $x
    t = mean_time_stats(trial)

    return (
        bytes=length(y) * sizeof(Float32) * 3,
        time_ms=t.ms,
        bandwidth_gbps=bandwidth_gbps(length(y), t.s),
        trial=trial,
        x=x,
        y=y,
    )
end

function benchmark_axpy_cpu!(x::Vector{Float32}, y::Vector{Float32}, alpha::Float32)
    axpy!(alpha, y, x)
    trial = @benchmark axpy!($alpha, $y, $x)
    t = mean_time_stats(trial)

    return (
        bytes=length(y) * sizeof(Float32) * 3,
        time_ms=t.ms,
        bandwidth_gbps=bandwidth_gbps(length(y), t.s),
        trial=trial,
    )
end

function benchmark_axpy_gpu!(x::Vector{Float32}, y::Vector{Float32}, alpha::Float32)
    x_m = MtlArray(x)
    y_m = MtlArray(y)

    axpy_gpu_sync!(alpha, y_m, x_m)
    trial = @benchmark axpy_gpu_sync!($alpha, $y_m, $x_m)
    t = mean_time_stats(trial)

    return (
        bytes=length(y) * sizeof(Float32) * 3,
        time_ms=t.ms,
        bandwidth_gbps=bandwidth_gbps(length(y), t.s),
        trial=trial,
    )
end

function benchmark_matmul_cpu(matmul_n::Int, seconds::Int)
    A = rand(Float32, matmul_n, matmul_n)
    B = rand(Float32, matmul_n, matmul_n)
    C = similar(B)

    mul!(C, A, B)
    trial = @benchmark mul!($C, $A, $B) seconds = seconds
    t = mean_time_stats(trial)

    return (
        time_ms=t.ms,
        gflops=matmul_gflops(matmul_n, t.s),
        trial=trial,
        A=A,
        B=B,
        C=C,
    )
end

function benchmark_matmul_gpu(A::Matrix{Float32}, B::Matrix{Float32}, C::Matrix{Float32}, seconds::Int)
    A_m = MtlArray(A)
    B_m = MtlArray(B)
    C_m = MtlArray(C)

    matmul_gpu_sync!(C_m, A_m, B_m)
    trial = @benchmark matmul_gpu_sync!($C_m, $A_m, $B_m) seconds = seconds
    t = mean_time_stats(trial)

    return (
        time_ms=t.ms,
        gflops=matmul_gflops(size(A, 1), t.s),
        trial=trial,
    )
end

function run_benchmarks()
    vector_n = env_int("VECTOR_N", DEFAULT_VECTOR_N)
    matmul_n = env_int("MATMUL_N", DEFAULT_MATMUL_N)
    matmul_seconds = env_int("MATMUL_SECONDS", DEFAULT_MATMUL_SECONDS)
    plot_dir = env_str("PLOT_DIR", DEFAULT_PLOT_DIR)
    alpha = DEFAULT_ALPHA

    println("Benchmarking vector addition for Peak Single Core Performance and Memory Bandwidth....")
    vec_cpu = benchmark_vector_add_cpu(vector_n)
    println("Length of y: ", vec_cpu.bytes, " bytes")
    println("Mean time for vector addition: ", vec_cpu.time_ms, " ms")
    println("Estimated bandwidth: ", vec_cpu.bandwidth_gbps, " GigaBytes/s")

    println("\nBenchmarking BLAS kernel for Peak Multi Core Performance and Memory Bandwidth....")
    axpy_cpu = benchmark_axpy_cpu!(vec_cpu.x, vec_cpu.y, alpha)
    println("Length of y: ", axpy_cpu.bytes, " bytes")
    println("Mean time for BLAS kernel: ", axpy_cpu.time_ms, " ms")
    println("Estimated bandwidth: ", axpy_cpu.bandwidth_gbps, " GigaBytes/s (More is Better)")

    println("\nBenchmarking vector addition on Apple Silicon GPU for Peak Performance and Memory Bandwidth....")
    axpy_gpu = benchmark_axpy_gpu!(vec_cpu.x, vec_cpu.y, alpha)
    println("Length of y: ", axpy_gpu.bytes, " bytes")
    println("Mean time for vector addition on Apple Silicon GPU: ", axpy_gpu.time_ms, " ms")
    println("Estimated bandwidth on Apple Silicon GPU: ", axpy_gpu.bandwidth_gbps, " GigaBytes/s (More is Better)")

    println("\nBenchmarking matrix multiplication on Apple Silicon CPU for Peak Compute Performance....")
    mm_cpu = benchmark_matmul_cpu(matmul_n, matmul_seconds)
    println("Mean time for matrix multiplication on Apple Silicon CPU: ", mm_cpu.time_ms, " ms")
    println("Estimated compute FLOPS for matrix multiplication on Apple Silicon CPU: ", mm_cpu.gflops, " GFLOPS (More is Better)")

    println("\nBenchmarking matrix multiplication on Apple Silicon GPU for Peak Compute Performance....")
    mm_gpu = benchmark_matmul_gpu(mm_cpu.A, mm_cpu.B, mm_cpu.C, matmul_seconds)
    println("Mean time for matrix multiplication on Apple Silicon GPU: ", mm_gpu.time_ms, " ms")
    println("Estimated compute FLOPS for matrix multiplication on Apple Silicon GPU: ", mm_gpu.gflops, " GFLOPS (More is Better)")

    save_plots((vec_cpu=vec_cpu, axpy_cpu=axpy_cpu, axpy_gpu=axpy_gpu, mm_cpu=mm_cpu, mm_gpu=mm_gpu), plot_dir)
    println("\nSaved plots to: ", plot_dir)
    println(" - ", joinpath(plot_dir, "vector_add_cpu_ms.png"))
    println(" - ", joinpath(plot_dir, "axpy_cpu_ms.png"))
    println(" - ", joinpath(plot_dir, "axpy_gpu_ms.png"))
    println(" - ", joinpath(plot_dir, "matmul_cpu_ms.png"))
    println(" - ", joinpath(plot_dir, "matmul_gpu_ms.png"))
end

run_benchmarks()