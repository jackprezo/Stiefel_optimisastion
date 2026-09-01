# Jack Prezerowitz
# Simple comparison: Landing vs RGD, and POGO vs RGD.

using LinearAlgebra
using Plots
using LaTeXStrings

include("rgd.jl")
include("landing.jl")
include("pogo.jl")
BLAS.set_num_threads(1)
plotsdir = joinpath(@__DIR__, "plots")
mkpath(plotsdir)

# Configuration esthétique
plot_font = "Computer Modern"
default(fontfamily=plot_font, linewidth=2, framestyle=:box, label=nothing, grid=false)
resetfontsizes()
scalefontsizes(1.3)

plasma = cgrad(:plasma)
color_rgd = plasma[0.1]
color_rgd_qr = plasma[0.3]
color_landing = plasma[0.6]
color_pogo = plasma[0.9]


pca_boolean = false
procruste_boolean = true

n = 100
p = 10
niter = 2000

τ = 0.5
r = 1e-4
ρ = 0.5

ε = 1e-16

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
if pca_boolean

    α_rgd = 2e-4 # 0.00005
    α_landing = 2e-4 # 0.00005
    α_pogo = 2e-4 # 0.00005
    λ_landing = 1e3 # 10000
    λ_pogo = 0.5 # 0.5

    B1 = randn(n, n)
    A = B1' * B1
    U = Diagonal(Array(p:-1:1))
    ϵ = eigen(Symmetric(A))
    V = ϵ.vectors[:, end:-1:end-p+1]

    f(X) = -tr(X' * A * X * U)
    _u_diag = U.diag
    @views function ∇f!(out, X)
        mul!(out, A, X)
        out .*= -2 .* _u_diag'
        return out
    end
    grad_C(∇f, X) = ∇f - X * ∇f' * X

    X0 = randn(n, p)
    q(X) = (f(X)-f(V))/abs(f(V))

    _, res_rgd, dist_rgd, _, _ = rgd(X0, ∇f!, f, q, grad_C, α_rgd, τ, r; niter=niter, retract=:polar)
    _, res_rgd_qr, dist_rgd_qr, _, _ = rgd(X0, ∇f!, f, q, grad_C, α_rgd, τ, r; niter=niter, retract=:qr)
    _, res_landing, dist_landing, _, _ = landing_flow(∇f!, f, q, X0, λ_landing, α_landing, τ, r; niter=niter)
    _, res_pogo, dist_pogo, _, _ = pogo(X0, ∇f!, f, q, α_pogo, τ, r, λ_pogo; niter=niter)

    pl1 = plot(xlabel="Iteration", ylabel=L"q(X)", res_pogo, yaxis=:log10, label="Pogo", alpha=1.0, color=color_pogo,
        linestyle=:dash, minorticks=true, tick_direction=:in, linewidth=3)
    plot!(pl1, res_landing, yaxis=:log10, label="landing", alpha=0.8, color=color_landing, linestyle=:dot)
    plot!(pl1, res_rgd, yaxis=:log10, label="RGD (Polar)", color=color_rgd, alpha=0.8)
    plot!(pl1, res_rgd_qr, yaxis=:log10, label="RGD (QR)", color=color_rgd_qr, alpha=0.8)

    pl2 = plot(xlabel="Iteration", ylabel=L"\|| XX^\top - I_p\||", max.(dist_pogo, ε), yaxis=:log10, label="POGO", alpha=0.8, color=color_pogo,
        linestyle=:dash, minorticks=true, tick_direction=:in)
    plot!(pl2, max.(dist_landing, ε), yaxis=:log10, label="Landing", alpha=0.8, color=color_landing, linestyle=:dot)
    plot!(pl2, max.(dist_rgd, ε), label="RGD (Polar)", color=color_rgd, alpha=0.8)
    plot!(pl2, max.(dist_rgd_qr, ε), label="RGD (QR)", color=color_rgd_qr, alpha=0.8)

    pl_pca = plot(pl1, pl2, layout=(1, 2), size=(1000, 300), left_margin=5Plots.mm, bottom_margin=10Plots.mm)

    savefig(pl_pca, joinpath(plotsdir, "pca.pdf"))

end

# ---------------------------------------------------------------------------
# Procrustes
# ---------------------------------------------------------------------------

if procruste_boolean


    α_rgd_proc = 4.328761281083061e-6      # optimal trouvé par params.jl (balayage 1D, n=100, p=10)
    α_landing_proc = 4.641588833612782e-6  # idem (balayage 2D Landing)
    α_pogo_proc = 4.328761281083061e-6     # idem
    λ_landing_proc = 46415.88833612782     # idem
    λ_pogo_proc = 0.5

    B1_proc = randn(n, n)
    A_proc = B1_proc' * B1_proc
    X_opt_proc = Matrix(qr(randn(n, p)).Q)
    B2_proc = A_proc * X_opt_proc


    grad_E(∇f, X) = ∇f - X * Symmetric(X' * ∇f)
    grad_C(∇f, X) = ∇f - X * ∇f' * X

    f_proc(X) = norm(A_proc * X - B2_proc)^2
    AtA_proc = A_proc' * A_proc
    AtB2_proc = A_proc' * B2_proc
    @views function ∇f_proc!(out, X)
        mul!(out, AtA_proc, X)
        out .*= 2
        out .-= 2 .* AtB2_proc
        return out
    end
    q_proc(X) = f_proc(X) / norm(A_proc)^2

    X0_proc = randn(n, p)

    _, res_rgd_proc, dist_rgd_proc, _, _ = rgd(X0_proc, ∇f_proc!, f_proc, q_proc, grad_C, α_rgd_proc, τ, r; niter=niter, retract=:polar)
    _, res_rgd_qr_proc, dist_rgd_qr_proc, _, _ = rgd(X0_proc, ∇f_proc!, f_proc, q_proc, grad_C, α_rgd_proc, τ, r; niter=niter, retract=:qr)
    _, res_landing_proc, dist_landing_proc, _, _ = landing_flow(∇f_proc!, f_proc, q_proc, X0_proc, λ_landing_proc, α_landing_proc, τ, r; niter=niter)
    _, res_pogo_proc, dist_pogo_proc, _, _ = pogo(X0_proc, ∇f_proc!, f_proc, q_proc, α_pogo_proc, τ, r, λ_pogo_proc; niter=niter)

    pl3 = plot(xlabel="Iteration", ylabel=L"p(X)", res_pogo_proc, yaxis=:log10, label="Pogo" , color=color_pogo,
    linestyle=:dash, minorticks=true, tick_direction=:in, linewidth=3)
    plot!(pl3, res_landing_proc, yaxis=:log10, label="Landing", alpha=0.8, color=color_landing, linestyle=:dot)
    plot!(pl3, res_rgd_proc, yaxis=:log10, label="RGD (Polar)", color=color_rgd, alpha=0.8 )
    plot!(pl3, res_rgd_qr_proc, yaxis=:log10, label="RGD (QR)", color=color_rgd_qr, alpha=0.8 )

    pl4 = plot(xlabel="Iteration", ylabel=L"\|| XX^\top - I_p\||", max.(dist_pogo_proc, ε), yaxis=:log10, label="POGO", alpha=0.8, color=color_pogo,
    linestyle=:dash, minorticks=true, tick_direction=:in)
    plot!(pl4, max.(dist_landing_proc, ε), yaxis=:log10, label="Landing", alpha=0.8, color=color_landing, linestyle=:dot)
    plot!(pl4, max.(dist_rgd_proc, ε), label="RGD (Polar)", color=color_rgd, alpha=0.8)
    plot!(pl4, max.(dist_rgd_qr_proc, ε), label="RGD (QR)", color=color_rgd_qr, alpha=0.8 )

    pl_proc = plot(pl3, pl4, layout=(1, 2), size=(1000, 300), left_margin=5Plots.mm, bottom_margin=10Plots.mm)

    savefig(pl_proc, joinpath(plotsdir, "procrustes.pdf"))
end