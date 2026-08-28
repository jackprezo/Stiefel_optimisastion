using BenchmarkTools
using LinearAlgebra
using Printf
using Statistics



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

@views function rgd(X0::AbstractMatrix{T}, ∇f!, f, grad, α_init; niter=1000, backtracking=false, retract=:polar) where T

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




@views function pogo!(X0, ∇f!, f, α_init, λ; niter=1000, backtracking=false)

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

# ---------------------------------------------------------------------------
# Tâches à exécuter
# ---------------------------------------------------------------------------

run_blas_threads_benchmark = false
run_retraction_comparison = false
run_all_methods_comparison = true    
# ---------------------------------------------------------------------------
# Benchmark PCA : RGD vs Landing vs POGO, BLAS threads 1 à 6
# ---------------------------------------------------------------------------

n = 1000
p = 500
niter = 5


α_rgd = 3.5e-7
α_landing = 2e-7
α_pogo = 8e-7
λ_landing = 7.2e4
λ_pogo = 0.5 

B1 = randn(n, n)
A = B1' * B1
U = Diagonal(Array(p:-1:1))
ϵ = eigen(Symmetric(A))
V = ϵ.vectors[:, end:-1:end-p+1]

f(X) = -tr(X' * A * X * U)
∇f(X) = -2 * A * X * U
q(X) = (f(X) - f(V)) / abs(f(V))

const _u_diag = U.diag

@views function ∇f!(out, X)
    mul!(out, A, X)
    out .*= -2 .* _u_diag'
    return out
end

grad_C(∇f, X) = ∇f - X * ∇f' * X

@views function grad_C!(gradf, ∇fX_X, X)
    mul!(∇fX_X, gradf', X)          # ∇fX_X = ∇f_X' * X   (p×p)
    mul!(gradf, X, ∇fX_X, -1, 1)    # gradf = ∇f_X - X * ∇fX_X
    return gradf
end

X0 = randn(n, p)

datadir = joinpath(@__DIR__, "data")
mkpath(datadir)

if run_blas_threads_benchmark
outfile = joinpath(datadir, "time_benchmark.txt")

open(outfile, "w") do io
    @printf(io, "%-9s %-10s %-10s %-12s\n", "nthread", "algo", "q(X)", "time (s)")

    for nthreads in 1:6
        BLAS.set_num_threads(nthreads)
        @printf("\n=== BLAS threads = %d ===\n", nthreads)

        print("POGO:      ")
        b_pogo = @benchmark pogo!($copy(X0), $∇f!, $f, $α_pogo, $λ_pogo; niter=$niter)
        display(b_pogo)
        println()
        X_pogo = pogo!(copy(X0), ∇f!, f, α_pogo, λ_pogo; niter=niter)
        q_pogo = q(X_pogo)
        t_pogo_mean = mean(b_pogo.times) / 1e9
        t_pogo_std = std(b_pogo.times) / 1e9

        print("RGD polar: ")
        b_rgd = @benchmark rgd($copy(X0), $∇f!, $f, $grad_C, $α_rgd; niter=$niter, retract=:polar)
        display(b_rgd)
        println()
        X_rgd = rgd(copy(X0), ∇f!, f, grad_C, α_rgd; niter=niter, retract=:polar)
        q_rgd = q(X_rgd)
        t_rgd_mean = mean(b_rgd.times) / 1e9
        t_rgd_std = std(b_rgd.times) / 1e9

        print("RGD qr:    ")
        b_rgd_qr = @benchmark rgd($copy(X0), $∇f!, $f, $grad_C, $α_rgd; niter=$niter, retract=:qr)
        display(b_rgd_qr)
        println()
        X_rgd_qr = rgd(copy(X0), ∇f!, f, grad_C, α_rgd; niter=niter, retract=:qr)
        q_rgd_qr = q(X_rgd_qr)
        t_rgd_qr_mean = mean(b_rgd_qr.times) / 1e9
        t_rgd_qr_std = std(b_rgd_qr.times) / 1e9

        print("Landing:   ")
        b_landing = @benchmark landing!($∇f!, $f, $copy(X0), $λ_landing, $α_landing; niter=$niter)
        display(b_landing)
        println()
        X_landing = landing!(∇f!, f, copy(X0), λ_landing, α_landing; niter=niter)
        q_landing = q(X_landing)
        t_landing_mean = mean(b_landing.times) / 1e9
        t_landing_std = std(b_landing.times) / 1e9

        @printf(io, "%-9d %-10s %-10.4g %.4f ± %.4f\n", nthreads, "RGD polar", q_rgd, t_rgd_mean, t_rgd_std)
        @printf(io, "%-9d %-10s %-10.4g %.4f ± %.4f\n", nthreads, "RGD qr", q_rgd_qr, t_rgd_qr_mean, t_rgd_qr_std)
        @printf(io, "%-9d %-10s %-10.4g %.4f ± %.4f\n", nthreads, "Landing", q_landing, t_landing_mean, t_landing_std)
        @printf(io, "%-9d %-10s %-10.4g %.4f ± %.4f\n", nthreads, "POGO", q_pogo, t_pogo_mean, t_pogo_std)
        flush(io)
    end
end

println("Résultats enregistrés dans ", outfile)
end

# ---------------------------------------------------------------------------
# Comparaison RGD : retraction polaire vs retraction QR, pour différents (n, p)
# ---------------------------------------------------------------------------

function compare_retractions(datadir)
    niter_retract = 20

    retract_sizes = [
        (100, 10, 0.0002442053094548652),
        (500, 50, 1.0e-5),
        (1000, 100, 2.3713737056616552e-6),
    ]

    outfile_retract = joinpath(datadir, "retraction_comparison.txt")

    open(outfile_retract, "w") do io
        @printf(io, "%-8s %-6s %-6s %-8s %-12s %-12s\n", "nthread", "n", "p", "retract", "q(X)", "time (s)")

        for nthreads in 1:6
            BLAS.set_num_threads(nthreads)

            for (n_r, p_r, α_r) in retract_sizes
                @printf("\n=== threads=%d, n=%d, p=%d, α=%.4g ===\n", nthreads, n_r, p_r, α_r)

                B1_r = randn(n_r, n_r)
                A_r = B1_r' * B1_r
                U_r = Diagonal(Array(p_r:-1:1))
                V_r = eigen(Symmetric(A_r)).vectors[:, end:-1:end-p_r+1]

                f_r(X) = -tr(X' * A_r * X * U_r)
                q_r(X) = (f_r(X) - f_r(V_r)) / abs(f_r(V_r))
                _u_diag_r = U_r.diag
                @views function ∇f_r!(out, X)
                    mul!(out, A_r, X)
                    out .*= -2 .* _u_diag_r'
                    return out
                end

                X0_r = randn(n_r, p_r)

                for retract in (:polar, :qr)
                    print("RGD ($retract): ")
                    b = @benchmark rgd($copy($X0_r), $∇f_r!, $f_r, $grad_C, $α_r; niter=$niter_retract, retract=$retract)
                    display(b)
                    println()

                    q_final = q_r(rgd(copy(X0_r), ∇f_r!, f_r, grad_C, α_r; niter=niter_retract, retract=retract))
                    t_mean, t_std = mean(b.times) / 1e9, std(b.times) / 1e9

                    @printf(io, "%-8d %-6d %-6d %-8s %-12.4g %.4f ± %.4f\n", nthreads, n_r, p_r, String(retract), q_final, t_mean, t_std)
                    flush(io)
                end
            end
        end
    end

    println("Résultats enregistrés dans ", outfile_retract)
end

if run_retraction_comparison
    compare_retractions(datadir)
end

# ---------------------------------------------------------------------------
# Comparaison des 4 méthodes (RGD-polar, RGD-qr, Landing, POGO) pour 3 tailles,
# avec le pas α (et λ pour Landing) optimal propre à chaque méthode et taille.
# ---------------------------------------------------------------------------

dist_stiefel(X) = norm(X' * X - I)

# make_pca_problem(n, p) : construit (f, q, ∇f!, X0) pour le problème PCA (n, p).
function make_pca_problem(n_m, p_m)
    B1_m = randn(n_m, n_m)
    A_m = B1_m' * B1_m
    U_m = Diagonal(Array(p_m:-1:1))
    V_m = eigen(Symmetric(A_m)).vectors[:, end:-1:end-p_m+1]

    f_m(X) = -tr(X' * A_m * X * U_m)
    q_m(X) = (f_m(X) - f_m(V_m)) / abs(f_m(V_m))
    _u_diag_m = U_m.diag
    @views function ∇f_m!(out, X)
        mul!(out, A_m, X)
        out .*= -2 .* _u_diag_m'
        return out
    end

    return f_m, q_m, ∇f_m!, randn(n_m, p_m)
end

# make_procrustes_problem(n, p) : construit (f, q, ∇f!, X0) pour le problème Procrustes (n, p).
function make_procrustes_problem(n_m, p_m)
    B1_m = randn(n_m, n_m)
    A_m = B1_m' * B1_m
    X_opt_m = Matrix(qr(randn(n_m, p_m)).Q)
    B2_m = A_m * X_opt_m

    f_m(X) = norm(A_m * X - B2_m)^2
    q_m(X) = f_m(X) / norm(B2_m)^2
    AtA_m = A_m' * A_m
    AtB2_m = A_m' * B2_m
    @views function ∇f_m!(out, X)
        mul!(out, AtA_m, X)
        out .*= 2
        out .-= 2 .* AtB2_m
        return out
    end

    return f_m, q_m, ∇f_m!, randn(n_m, p_m)
end

# compare_all_methods(datadir, outname, make_problem, method_sizes) :
# balaie nthreads=1:6 × method_sizes × {RGD-polar, RGD-qr, Landing, POGO},
# écrit (α, résidu q(X), distance Stiefel, temps) dans datadir/outname.
function compare_all_methods(datadir, outname, make_problem, method_sizes)
    niter_methods = 20

    outfile_methods = joinpath(datadir, outname)

    open(outfile_methods, "w") do io
        @printf(io, "%-8s %-6s %-6s %-14s %-12s %-14s %-14s %-14s\n", "nthread", "n", "p", "method", "alpha", "residual q(X)", "dist Stiefel", "time (s)")

        for nthreads in 1:6
            BLAS.set_num_threads(nthreads)

            for (n_m, p_m, α_rgd_m, α_landing_m, λ_landing_m, α_pogo_m) in method_sizes
                @printf("\n=== threads=%d, n=%d, p=%d ===\n", nthreads, n_m, p_m)

                f_m, q_m, ∇f_m!, X0_m = make_problem(n_m, p_m)

                print("RGD (polar): ")
                b_rgd_polar = @benchmark rgd($copy($X0_m), $∇f_m!, $f_m, $grad_C, $α_rgd_m; niter=$niter_methods, retract=:polar)
                display(b_rgd_polar)
                println()
                X_rgd_polar = rgd(copy(X0_m), ∇f_m!, f_m, grad_C, α_rgd_m; niter=niter_methods, retract=:polar)
                t_rgd_polar = mean(b_rgd_polar.times) / 1e9
                @printf(io, "%-8d %-6d %-6d %-14s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n", nthreads, n_m, p_m, "RGD (polar)", α_rgd_m, q_m(X_rgd_polar), dist_stiefel(X_rgd_polar), t_rgd_polar, std(b_rgd_polar.times) / 1e9)
                flush(io)

                print("RGD (qr):    ")
                b_rgd_qr = @benchmark rgd($copy($X0_m), $∇f_m!, $f_m, $grad_C, $α_rgd_m; niter=$niter_methods, retract=:qr)
                display(b_rgd_qr)
                println()
                X_rgd_qr = rgd(copy(X0_m), ∇f_m!, f_m, grad_C, α_rgd_m; niter=niter_methods, retract=:qr)
                t_rgd_qr = mean(b_rgd_qr.times) / 1e9
                @printf(io, "%-8d %-6d %-6d %-14s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n", nthreads, n_m, p_m, "RGD (qr)", α_rgd_m, q_m(X_rgd_qr), dist_stiefel(X_rgd_qr), t_rgd_qr, std(b_rgd_qr.times) / 1e9)
                flush(io)

                print("Landing:     ")
                b_landing = @benchmark landing!($∇f_m!, $f_m, $copy($X0_m), $λ_landing_m, $α_landing_m; niter=$niter_methods)
                display(b_landing)
                println()
                X_landing = landing!(∇f_m!, f_m, copy(X0_m), λ_landing_m, α_landing_m; niter=niter_methods)
                t_landing = mean(b_landing.times) / 1e9
                @printf(io, "%-8d %-6d %-6d %-14s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n", nthreads, n_m, p_m, "Landing", α_landing_m, q_m(X_landing), dist_stiefel(X_landing), t_landing, std(b_landing.times) / 1e9)
                flush(io)

                print("POGO:        ")
                b_pogo = @benchmark pogo!($copy($X0_m), $∇f_m!, $f_m, $α_pogo_m, 0.5; niter=$niter_methods)
                display(b_pogo)
                println()
                X_pogo = pogo!(copy(X0_m), ∇f_m!, f_m, α_pogo_m, 0.5; niter=niter_methods)
                t_pogo = mean(b_pogo.times) / 1e9
                @printf(io, "%-8d %-6d %-6d %-14s %-12.4g %-14.4g %-14.4g %.4f ± %.4f\n", nthreads, n_m, p_m, "POGO", α_pogo_m, q_m(X_pogo), dist_stiefel(X_pogo), t_pogo, std(b_pogo.times) / 1e9)
                flush(io)
            end
        end
    end

    println("Résultats enregistrés dans ", outfile_methods)
end

# (n, p, α_rgd, α_landing, λ_landing, α_pogo) : trouvés par balayage pour chaque taille.
pca_method_sizes = [
    (100, 10, 0.0002442053094548652, 0.0002395026619987486, 3290.3445623126677, 0.00047148663634573947),
    (500, 50, 1.0e-5, 3.162277660168379e-6, 10000.0, 1.0e-5),
    (1000, 100, 2.3713737056616552e-6, 1.0e-6, 1.0e6, 1.778279410038923e-6),
]

procrustes_method_sizes = [
    (100, 10, 4.328761281083061e-6, 4.641588833612782e-6, 46415.88833612782, 4.328761281083061e-6),
    (500, 50, 1.7782794100389227e-7, 6.30957344480193e-8, 100000.0, 3.162277660168379e-7),
    (1000, 100, 5.6234132519034905e-8, 1.5848931924611143e-8, 1.0e6, 1.0e-7),
]

if run_all_methods_comparison
    compare_all_methods(datadir, "methods_comparison_procrustes.txt", make_procrustes_problem, procrustes_method_sizes)
end