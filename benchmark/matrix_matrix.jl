using LinearAlgebra
using BenchmarkTools

n = 4000

C = zeros(n, n)
A = randn(n, n)
B = randn(n, n)

BLAS.set_num_threads(1)
@btime mul!(C, A, B)


BLAS.set_num_threads(4)
@btime mul!(C, A, B)
println("done")
