using BenchmarkTools
using LinearAlgebra
using Metal
using Plots

# Define a large Number of iterations for benchmarking
N = 2^30

x = rand(Float32, N)
y = rand(Float32, N)

# Benchmark the using vector addition for SC performance and bandwidth
println("Benchmarking vector addition for Peak Single Core Performance and Memory Bandwidth....")
time_taken = @benchmark $y .+= $x

# Use mean time from the benchmark results to calculate the best time in seconds and milliseconds
best_time_ns = mean(time_taken).time
best_time_ms = best_time_ns / 1e6
best_time_s = best_time_ns / 1e9

println("Length of y: ", length(y) * 4 * 3, " bytes") # Each Float32 is 4 bytes, and we have 3 arrays (x, y, and the result)
println("Best time for vector addition: ", best_time_ms, " ms")

print("Estimated bandwidth: ", ((length(y) * 4 * 3) / best_time_s) / 1e9, " GigaBytes/s") # Bandwidth = Total data processed / Time taken

# BLAS Kernel benchmark for MC performance and bandwidth by reusing the x and y values
println("\nBenchmarking BLAS kernel for Peak Multi Core Performance and Memory Bandwidth....")
a = 1.0f0
time_taken_blas = @benchmark axpy!(a, $y, $x)

# Use mean time from the benchmark results to calculate the best time in seconds and milliseconds
best_time_ns = mean(time_taken_blas).time
best_time_ms = best_time_ns / 1e6
best_time_s = best_time_ns / 1e9

println("Length of y: ", length(y) * 4 * 3, " bytes") # Each Float32 is 4 bytes, and we have 3 arrays (x, y, and the result)
println("Best time for BLAS kernel: ", best_time_ms, " ms")

print("Estimated bandwidth: ", ((length(y) * 4 * 3) / best_time_s) / 1e9, " GigaBytes/s (More is Better)") # Bandwidth = Total data processed / Time taken

# Use Metal backend to test Apple Silicon based GPU for increased parallelism to achieve peak performance and bandwidth by reusing the x and y values declared as new metal arrays
x_m = MtlArray(x)
y_m = MtlArray(y)
println("\nBenchmarking vector addition on Apple Silicon GPU for Peak Performance and Memory Bandwidth....")
time_taken_gpu = @benchmark Metal.@sync axpy!(a, $y_m, $x_m)
# Use mean time from the benchmark results to calculate the best time in seconds and milliseconds
best_time_ns_gpu = mean(time_taken_gpu).time
best_time_ms_gpu = best_time_ns_gpu / 1e6
best_time_s_gpu = best_time_ns_gpu / 1e9
println("Length of y: ", length(y) * 4 * 3, " bytes") # Each Float32 is 4 bytes, and we have 3 arrays (x, y, and the result)
println("Best time for vector addition on Apple Silicon GPU: ", best_time_ms_gpu, " ms")
print("Estimated bandwidth on Apple Silicon GPU: ", ((length(y) * 4 * 3) / best_time_s_gpu) / 1e9, " GigaBytes/s (More is Better)") # Bandwidth = Total data processed / Time taken

# Use Matrix Multiplication to Peak Compute Performance and reduce max Numbers size
N = 16384
A = rand(Float32, N, N)
B = rand(Float32, N, N)
C = similar(B)
println("\nBenchmarking matrix multiplication on Apple Silicon CPU for Peak Compute Performance....")
time_taken_mm_cpu = @benchmark mul!($C, $A, $B) seconds = 30

# Use mean time from the benchmark results to calculate the best time in seconds and milliseconds
best_time_ns_mm_cpu = mean(time_taken_mm_cpu).time
best_time_ms_mm_cpu = best_time_ns_mm_cpu / 1e6
best_time_s_mm_cpu = best_time_ns_mm_cpu / 1e9
println("Best time for matrix multiplication on Apple Silicon CPU: ", best_time_ms_mm_cpu, " ms")
print("Estimated compute FLOPS for matrix multiplication on Apple Silicon CPU: ", ((2e-9 * N^3) / best_time_s_mm_cpu), " GFLOPS (More is Better)")

# Use Metal backend to test Apple Silicon based GPU for increased parallelism to achieve peak performance in FLOPS for matrix multiplication
a = MtlArray(A)
b = MtlArray(B)
c = MtlArray(C)
println("\nBenchmarking matrix multiplication on Apple Silicon GPU for Peak Compute Performance....")
time_taken_mmc_gpu = @benchmark mul!($c, $a, $b) seconds = 30
# Use mean time from the benchmark results to calculate the best time in seconds and milliseconds
best_time_ns_mmc_gpu = mean(time_taken_mmc_gpu).time
best_time_ms_mmc_gpu = best_time_ns_mmc_gpu / 1e6
best_time_s_mmc_gpu = best_time_ns_mmc_gpu / 1e9
println("Best time for matrix multiplication on Apple Silicon GPU: ", best_time_ms_mmc_gpu, " ms")
print("Estimated compute FLOPS for matrix multiplication on Apple Silicon GPU: ", ((2e-9 * N^3) / best_time_s_mmc_gpu), " GFLOPS (More is Better)")