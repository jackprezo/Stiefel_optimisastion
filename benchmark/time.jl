# Jack Prezerowitz
# Benchmark temps : RGD (polar), RGD (qr), Landing, POGO, pour PCA et Procrustes,
# à 3 tailles (n,p), en faisant varier le nombre de threads BLAS de 1 à 6.
# Un seul appel final ; pas α (et λ pour Landing) optimaux codés en dur
# (trouvés par balayage préalable via params.jl, niter réduit).

using BenchmarkTools
using LinearAlgebra
using Printf
using Statistics

grad_C(∇f, X) = ∇f - X * ∇f' * X

@views function grad_C!(gradf, ∇fX_X, X)
    mul!(∇fX_X, gradf', X)          # ∇fX_X = ∇f_X' * X   (p×p)
    mul!(gradf, X, ∇fX_X, -1, 1)    # gradf = ∇f_X - X * ∇fX_X
    return gradf
end

@views function retract_polar!(X, X_new, E, ε)
    e = Inf
    while e > ε
        mul!(E, X_new', X_new)
        @inbounds for j in 1:size(E, 1)
            E[j, j] -= 1
        end
        e = norm(E)
        @inbounds for j in 1:size(E, 1)
            E[j, j] += 1
        end
        X .= X_new
        mul!(X_new, X, E, -0.5, 1.5)
    end
    X .= X_new
    return X
end

@views function retract_qr!(X, X_new)
    X .= Matrix(qr(X_new).Q)
    return X
end

@views function rgd(X0::AbstractMatrix{T}, ∇f!, f, grad, α_init; niter=1000, retract=:polar) where T
    n, p = size(X0)
    X = Matrix(qr(X0).Q)

    gradf = Matrix{eltype(X)}(undef, n, p)
    ∇fX_X = Matrix{eltype(X)}(undef, p, p)
    X_new = Matrix{eltype(X)}(undef, n, p)

    ε = sqrt(eps(T) * 10 * sqrt(p))

    for i in 1:niter
        ∇f!(gradf, X)
        grad_C!(gradf, ∇fX_X, X)

        α = α_init
        @. X_new = X - α * gradf

        if retract === :polar
            retract_polar!(X, X_new, ∇fX_X, ε)
        elseif retract === :qr
            retract_qr!(X, X_new)
        else
            error("retract inconnu: $retract (attendu :polar ou :qr)")
        end
    end
    return X
end

@views function landing!(∇f!, f, X0, λ, α_init; niter=10000)
    n, p = size(X0)
    X = Matrix(qr(X0).Q)

    ∇f_X = Matrix{eltype(X)}(undef, n, p)
    XX = Matrix{eltype(X)}(undef, p, p)       # X'X, then X'X - I_p
    Λ_λ = Matrix{eltype(X)}(undef, n, p)      # ψ_X, then Λ_λ
    ∇N_X = Matrix{eltype(X)}(undef, n, p)     # X*(∇f_X'X), then ∇N_X

    for i in 1:niter
        ∇f!(∇f_X, X)

        # ψ_X = ∇f_X * (X'X) - X * (∇f_X'X)
        mul!(XX, X', X)
        mul!(Λ_λ, ∇f_X, XX)
        mul!(XX, ∇f_X', X)
        mul!(∇N_X, X, XX)
        Λ_λ .-= ∇N_X

        # ∇N_X = X * (X'X - I_p)
        mul!(XX, X', X)
        @inbounds for j in 1:p
            XX[j, j] -= 1
        end
        mul!(∇N_X, X, XX)

        # Λ_λ = ψ_X + λ * ∇N_X
        @. Λ_λ += λ * ∇N_X

        α = α_init
        @. X -= α * Λ_λ
    end

    return X
end

@views function pogo!(X0, ∇f!, f, α_init, λ; niter=1000)
    n, p = size(X0)
    X = Matrix(qr(X0).Q)

    ∇f_X = Matrix{eltype(X)}(undef, n, p)
    XX = Matrix{eltype(X)}(undef, p, p)   # X'X, then ∇f_X'X, then M'M
    ψ = Matrix{eltype(X)}(undef, n, p)    # ∇f_X*(X'X), then ψ
    MM = Matrix{eltype(X)}(undef, n, p)   # X*(∇f_X'X), then M*(M'M)
    M = Matrix{eltype(X)}(undef, n, p)

    for i in 1:niter
        ∇f!(∇f_X, X)

        # ψ = 0.5 * (∇f_X * (X'X) - X * (∇f_X'X))
        mul!(XX, X', X)
        mul!(ψ, ∇f_X, XX)
        mul!(XX, ∇f_X', X)
        mul!(MM, X, XX)
        ψ .= 0.5 .* (ψ .- MM)

        α = α_init
        # M = X - α * ψ
        @. M = X - α * ψ

        # X = M + λ * M * (I_p - M'M) = (1+λ)*M - λ * M * (M'M)
        mul!(XX, M', M)
        mul!(MM, M, XX)
        @. X = (1 + λ) * M - λ * MM
    end

    return X
end

dist_stiefel(X) = norm(X' * X - I)

# make_pca_problem(n, p) : construit (f, q, ∇f!, X0) pour le problème PCA.
function make_pca_problem(n, p)
    B1 = randn(n, n)
    A = B1' * B1
    U = Diagonal(Array(p:-1:1))
    V = eigen(Symmetric(A)).vectors[:, end:-1:end-p+1]

    f(X) = -tr(X' * A * X * U)
    q(X) = (f(X) - f(V)) / abs(f(V))
    _u_diag = U.diag
    @views function ∇f!(out, X)
        mul!(out, A, X)
        out .*= -2 .* _u_diag'
        return out
    end

    return f, q, ∇f!, randn(n, p)
end

# make_procrustes_problem(n, p) : construit (f, q, ∇f!, X0) pour le problème Procrustes.
function make_procrustes_problem(n, p)
    B1 = randn(n, n)
    A = B1' * B1
    X_opt = Matrix(qr(randn(n, p)).Q)
    B2 = A * X_opt

    f(X) = norm(A * X - B2)^2
    q(X) = f(X) / norm(B2)^2
    AtA = A' * A
    AtB2 = A' * B2
    @views function ∇f!(out, X)
        mul!(out, AtA, X)
        out .*= 2
        out .-= 2 .* AtB2
        return out
    end

    return f, q, ∇f!, randn(n, p)
end

# ---------------------------------------------------------------------------
# Benchmark : PCA et Procrustes, 3 tailles (n,p), RGD-polar/RGD-qr/Landing/POGO,
# threads BLAS 1 à 6. Pas α (et λ) optimaux codés en dur par (problème, taille),
# trouvés par balayage préalable (params.jl, niter réduit).
# ---------------------------------------------------------------------------

niter = 20
λ_pogo = 0.5

# (n, p, α_rgd, α_landing, λ_landing, α_pogo)
pca_sizes = [
    (100, 10, 0.0002442053094548652, 0.0002395026619987486, 3290.3445623126677, 0.0002442053094548652),
    (500, 50, 1.2328467394420658e-5, 5.994842503189409e-6, 21544.346900318822, 1.2328467394420658e-5),
    (1000, 100, 4.328761281083061e-6, 2.1544346900318822e-6, 166810.0537200059, 4.328761281083061e-6),
]

procrustes_sizes = [
    (100, 10, 4.328761281083061e-6, 4.641588833612782e-6, 46415.88833612782, 4.328761281083061e-6),
    (500, 50, 1.0e-6, 1.2915496650148827e-7, 100000.0, 1.0e-6),
    (1000, 100, 6.579332246575682e-8, 1.2915496650148827e-8, 1.0e6, 6.579332246575682e-8),
]

# (nom, make_problem, sizes)
problems = [
    ("PCA", make_pca_problem, pca_sizes),
    ("Procrustes", make_procrustes_problem, procrustes_sizes),
]

datadir = joinpath(@__DIR__, "data")
mkpath(datadir)
outfile = joinpath(datadir, "time_benchmark.txt")

open(outfile, "w") do io
    @printf(io, "%-9s %-10s %-6s %-6s %-12s %-12s %-14s %-14s %-14s\n",
        "nthread", "problem", "n", "p", "method", "alpha", "residual q(X)", "dist Stiefel", "time (s)")

    for nthreads in 1:6
        BLAS.set_num_threads(nthreads)

        for (name, make_problem, sizes) in problems
            for (n, p, α_rgd, α_landing, λ_landing, α_pogo) in sizes
                @printf("\n=== threads=%d, problem=%s, n=%d, p=%d ===\n", nthreads, name, n, p)

                f, q, ∇f!, X0 = make_problem(n, p)

                print("RGD (polar): ")
                b = @benchmark rgd($copy($X0), $∇f!, $f, $grad_C, $α_rgd; niter=$niter, retract=:polar)
                display(b)
                println()
                X = rgd(copy(X0), ∇f!, f, grad_C, α_rgd; niter=niter, retract=:polar)
                @printf(io, "%-9d %-10s %-6d %-6d %-12s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n",
                    nthreads, name, n, p, "RGD (polar)", α_rgd, q(X), dist_stiefel(X), mean(b.times) / 1e9, std(b.times) / 1e9)
                flush(io)

                print("RGD (qr):    ")
                b = @benchmark rgd($copy($X0), $∇f!, $f, $grad_C, $α_rgd; niter=$niter, retract=:qr)
                display(b)
                println()
                X = rgd(copy(X0), ∇f!, f, grad_C, α_rgd; niter=niter, retract=:qr)
                @printf(io, "%-9d %-10s %-6d %-6d %-12s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n",
                    nthreads, name, n, p, "RGD (qr)", α_rgd, q(X), dist_stiefel(X), mean(b.times) / 1e9, std(b.times) / 1e9)
                flush(io)

                print("Landing:     ")
                b = @benchmark landing!($∇f!, $f, $copy($X0), $λ_landing, $α_landing; niter=$niter)
                display(b)
                println()
                X = landing!(∇f!, f, copy(X0), λ_landing, α_landing; niter=niter)
                @printf(io, "%-9d %-10s %-6d %-6d %-12s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n",
                    nthreads, name, n, p, "Landing", α_landing, q(X), dist_stiefel(X), mean(b.times) / 1e9, std(b.times) / 1e9)
                flush(io)

                print("POGO:        ")
                b = @benchmark pogo!($copy($X0), $∇f!, $f, $α_pogo, $λ_pogo; niter=$niter)
                display(b)
                println()
                X = pogo!(copy(X0), ∇f!, f, α_pogo, λ_pogo; niter=niter)
                @printf(io, "%-9d %-10s %-6d %-6d %-12s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n",
                    nthreads, name, n, p, "POGO", α_pogo, q(X), dist_stiefel(X), mean(b.times) / 1e9, std(b.times) / 1e9)
                flush(io)
            end
        end
    end
end

println("Résultats enregistrés dans ", outfile)
