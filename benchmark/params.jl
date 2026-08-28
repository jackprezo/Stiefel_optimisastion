# Jack Prezerowitz
# Recherche du pas (α) optimal pour chaque méthode, sans backtracking.
# Simulation de type Monte-Carlo : chaque point de la grille est moyenné sur
# n_runs tirages aléatoires indépendants de (A, X0).
# Pour Landing, balayage 2D (α, λ) car la méthode a deux hyperparamètres → heatmap.

using LinearAlgebra
using Plots
using Statistics
using LaTeXStrings

include("rgd.jl")
include("landing.jl")
include("pogo.jl")

# Configuration esthétique
plot_font = "Computer Modern"
default(fontfamily=plot_font, linewidth=2, framestyle=:box, label=nothing, grid=false,
    minorticks=true, tick_direction=:in, tickfontsize=10)
resetfontsizes()
scalefontsizes(1.3)

plasma = cgrad(:plasma)
color_rgd_pca = plasma[0.05]
color_rgd_qr_pca = plasma[0.2]
color_pogo_pca = plasma[0.35]
color_rgd_proc = plasma[0.55]
color_rgd_qr_proc = plasma[0.7]
color_pogo_proc = plasma[0.9]

plotsdir = joinpath(@__DIR__, "plots")
mkpath(plotsdir)

# Supprime les println de debug internes à rgd/landing/pogo pendant le balayage
silent(f) = redirect_stdout(f, devnull)

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
n = 100
p = 10
niter = 150
n_runs = 1   # nombre de tirages Monte-Carlo par point de la grille
ε = 1e-16

τ = 0.5
r = 1e-2
λ_pogo = 0.5

grad_C(∇f, X) = ∇f - X * ∇f' * X

# Grilles 1D (RGD, POGO)
α_grid = 10.0 .^ range(-7, -4, length=12)

# Grille 2D (Landing) : α (pas) × λ (poids de pénalité)
α_grid_landing = 10.0 .^ range(-8, -4, length=10)
λ_grid_landing = 10.0 .^ range(-2, 10, length=10)

final_residual(res) = isfinite(res[end]) ? res[end] : NaN
best(grid, res) = grid[argmin(abs.(replace(res, NaN => Inf)))]
clip(res, cap) = map(x -> (!isfinite(x) || x > cap) ? cap : x, res)

# ===========================================================================
# Problème PCA
# ===========================================================================
B1_pca = randn(n, n)
A_pca = B1_pca' * B1_pca
N_pca = Diagonal(Array(p:-1:1))
ϵ_pca = eigen(Symmetric(A_pca))
V_pca = ϵ_pca.vectors[:, end:-1:end-p+1]
f_pca(X) = -tr(X' * A_pca * X * N_pca)
_n_pca_diag = N_pca.diag
@views function ∇f_pca!(out, X)
    mul!(out, A_pca, X)
    out .*= -2 .* _n_pca_diag'
    return out
end
X0_pca = randn(n, p)
q_pca(X) = (f_pca(X) - f_pca(V_pca)) / abs(f_pca(V_pca))

# ===========================================================================
# Problème Procrustes
# ===========================================================================
B1_proc = randn(n, n)
A_proc = B1_proc' * B1_proc
X_opt_proc = Matrix(qr(randn(n, p)).Q)
B2_proc = A_proc * X_opt_proc
f_proc(X) = norm(A_proc * X - B2_proc)^2
AtA_proc = A_proc' * A_proc
AtB2_proc = A_proc' * B2_proc
@views function ∇f_proc!(out, X)
    mul!(out, AtA_proc, X)
    out .*= 2
    out .-= 2 .* AtB2_proc
    return out
end
X0_proc = randn(n, p)
q_proc(X) = f_proc(X) / norm(B2_proc)^2

# ===========================================================================
# Balayage 1D : RGD et POGO, sur PCA et Procrustes
# ===========================================================================

println("--- Balayage 1D (RGD, POGO) ---")

res_rgd_pca = zeros(length(α_grid))
res_rgd_qr_pca = zeros(length(α_grid))
res_pogo_pca = zeros(length(α_grid))
res_rgd_proc = zeros(length(α_grid))
res_rgd_qr_proc = zeros(length(α_grid))
res_pogo_proc = zeros(length(α_grid))

for (k, α) in enumerate(α_grid)
    _, res, _, _, _ = silent() do
        rgd(X0_pca, ∇f_pca!, f_pca, q_pca, grad_C, α, τ, r; niter=niter, backtracking=false, retract=:polar)
    end
    res_rgd_pca[k] = final_residual(res)

    _, res, _, _, _ = silent() do
        rgd(X0_pca, ∇f_pca!, f_pca, q_pca, grad_C, α, τ, r; niter=niter, backtracking=false, retract=:qr)
    end
    res_rgd_qr_pca[k] = final_residual(res)

    _, res, _, _, _ = silent() do
        pogo(X0_pca, ∇f_pca!, f_pca, q_pca, α, τ, r, λ_pogo; niter=niter, backtracking=false)
    end
    res_pogo_pca[k] = final_residual(res)

    _, res, _, _, _ = silent() do
        rgd(X0_proc, ∇f_proc!, f_proc, q_proc, grad_C, α, τ, r; niter=niter, backtracking=false, retract=:polar)
    end
    res_rgd_proc[k] = final_residual(res)

    _, res, _, _, _ = silent() do
        rgd(X0_proc, ∇f_proc!, f_proc, q_proc, grad_C, α, τ, r; niter=niter, backtracking=false, retract=:qr)
    end
    res_rgd_qr_proc[k] = final_residual(res)

    _, res, _, _, _ = silent() do
        pogo(X0_proc, ∇f_proc!, f_proc, q_proc, α, τ, r, λ_pogo; niter=niter, backtracking=false)
    end
    res_pogo_proc[k] = final_residual(res)
end

finite_vals_1d = filter(isfinite, vcat(res_rgd_pca, res_rgd_qr_pca, res_pogo_pca, res_rgd_proc, res_rgd_qr_proc, res_pogo_proc))
res_cap_1d = quantile(finite_vals_1d, 0.9)

clipped_rgd_pca = max.(abs.(clip(res_rgd_pca, res_cap_1d)), ε)
clipped_rgd_qr_pca = max.(abs.(clip(res_rgd_qr_pca, res_cap_1d)), ε)
clipped_pogo_pca = max.(abs.(clip(res_pogo_pca, res_cap_1d)), ε)
clipped_rgd_proc = max.(abs.(clip(res_rgd_proc, res_cap_1d)), ε)
clipped_rgd_qr_proc = max.(abs.(clip(res_rgd_qr_proc, res_cap_1d)), ε)
clipped_pogo_proc = max.(abs.(clip(res_pogo_proc, res_cap_1d)), ε)

α_rgd_pca_opt = best(α_grid, res_rgd_pca)
α_rgd_qr_pca_opt = best(α_grid, res_rgd_qr_pca)
α_pogo_pca_opt = best(α_grid, res_pogo_pca)
α_rgd_proc_opt = best(α_grid, res_rgd_proc)
α_rgd_qr_proc_opt = best(α_grid, res_rgd_qr_proc)
α_pogo_proc_opt = best(α_grid, res_pogo_proc)

println("α optimal RGD Polar  - PCA:        ", α_rgd_pca_opt)
println("α optimal RGD QR     - PCA:        ", α_rgd_qr_pca_opt)
println("α optimal POGO         - PCA:        ", α_pogo_pca_opt)
println("α optimal RGD Polar  - Procrustes: ", α_rgd_proc_opt)
println("α optimal RGD QR     - Procrustes: ", α_rgd_qr_proc_opt)
println("α optimal POGO         - Procrustes: ", α_pogo_proc_opt)

# ---------------------------------------------------------------------------
# Plot combiné : les 6 courbes (RGD-polar/RGD-qr/POGO × PCA/Procrustes) sur un même axe
# ---------------------------------------------------------------------------

all_finite_1d = filter(isfinite, vcat(clipped_rgd_pca, clipped_rgd_qr_pca, clipped_pogo_pca,
    clipped_rgd_proc, clipped_rgd_qr_proc, clipped_pogo_proc))
ylim_lo = minimum(all_finite_1d) / 2
ylim_hi = maximum(all_finite_1d) * 2

pl_alpha = plot(xlabel=L"\alpha", ylabel=L"q(X)", xaxis=:log10, yaxis=:log10,
    framestyle=:box, grid=true, gridalpha=0.3, ylims=(ylim_lo, ylim_hi))
plot!(pl_alpha, α_grid, clipped_rgd_pca, label="RGD Polar", color=color_rgd_pca, marker=:circle, markersize=3)
plot!(pl_alpha, α_grid, clipped_rgd_qr_pca, label="RGD QR", color=color_rgd_qr_pca, marker=:circle, markersize=3)
plot!(pl_alpha, α_grid, clipped_pogo_pca, label="POGO", color=color_pogo_pca, marker=:circle, markersize=3)
plot!(pl_alpha, α_grid, clipped_rgd_proc, label="RGD Polar : Procrustes", color=color_rgd_proc, marker=:circle, markersize=3)
plot!(pl_alpha, α_grid, clipped_rgd_qr_proc, label="RGD QR : Procrustes", color=color_rgd_qr_proc, marker=:circle, markersize=3)
plot!(pl_alpha, α_grid, clipped_pogo_proc, label="POGO : Procrustes", color=color_pogo_proc, marker=:circle, markersize=3)

# xlims/ylims figés explicitement : scatter! sur axe log peut sinon
# ré-étendre les limites et déformer le rendu du plot.
xl_alpha, yl_alpha = xlims(pl_alpha), ylims(pl_alpha)
res_rgd_pca_opt = clipped_rgd_pca[argmin(abs.(replace(res_rgd_pca, NaN => Inf)))]
res_rgd_qr_pca_opt = clipped_rgd_qr_pca[argmin(abs.(replace(res_rgd_qr_pca, NaN => Inf)))]
res_pogo_pca_opt = clipped_pogo_pca[argmin(abs.(replace(res_pogo_pca, NaN => Inf)))]
res_rgd_proc_opt = clipped_rgd_proc[argmin(abs.(replace(res_rgd_proc, NaN => Inf)))]
res_rgd_qr_proc_opt = clipped_rgd_qr_proc[argmin(abs.(replace(res_rgd_qr_proc, NaN => Inf)))]
res_pogo_proc_opt = clipped_pogo_proc[argmin(abs.(replace(res_pogo_proc, NaN => Inf)))]

scatter!(pl_alpha, [α_rgd_pca_opt], [res_rgd_pca_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_pca, xlims=xl_alpha, ylims=yl_alpha, label=nothing)
scatter!(pl_alpha, [α_rgd_qr_pca_opt], [res_rgd_qr_pca_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_qr_pca, xlims=xl_alpha, ylims=yl_alpha, label=nothing)
scatter!(pl_alpha, [α_pogo_pca_opt], [res_pogo_pca_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_pogo_pca, xlims=xl_alpha, ylims=yl_alpha, label=nothing)
scatter!(pl_alpha, [α_rgd_proc_opt], [res_rgd_proc_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_proc, xlims=xl_alpha, ylims=yl_alpha, label=nothing)
scatter!(pl_alpha, [α_rgd_qr_proc_opt], [res_rgd_qr_proc_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_qr_proc, xlims=xl_alpha, ylims=yl_alpha, label=nothing)
scatter!(pl_alpha, [α_pogo_proc_opt], [res_pogo_proc_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_pogo_proc, xlims=xl_alpha, ylims=yl_alpha, label=nothing)

pl_alpha = plot(pl_alpha, size=(900, 250), left_margin=5Plots.mm, bottom_margin=7Plots.mm, legend=:topleft, legendfontsize=7)
savefig(pl_alpha, joinpath(plotsdir, "alpha_optimal.pdf"))

# ===========================================================================
# Balayage 2D : Landing (α, λ), sur PCA et Procrustes
# ===========================================================================

println("--- Balayage 2D (Landing) ---")

res_landing_pca = zeros(length(λ_grid_landing), length(α_grid_landing))
res_landing_proc = zeros(length(λ_grid_landing), length(α_grid_landing))

for (i, λ) in enumerate(λ_grid_landing), (j, α) in enumerate(α_grid_landing)
    _, res, _, _, _ = silent() do
        landing_flow(∇f_pca!, f_pca, q_pca, X0_pca, λ, α, τ, r; niter=niter, backtracking=false)
    end
    res_landing_pca[i, j] = final_residual(res)

    _, res, _, _, _ = silent() do
        landing_flow(∇f_proc!, f_proc, q_proc, X0_proc, λ, α, τ, r; niter=niter, backtracking=false)
    end
    res_landing_proc[i, j] = final_residual(res)
end

finite_vals_pca = filter(isfinite, vec(res_landing_pca))
finite_vals_proc = filter(isfinite, vec(res_landing_proc))
res_cap_pca = quantile(finite_vals_pca, 0.9)
res_cap_proc = quantile(finite_vals_proc, 0.9)

clipped_landing_pca = max.(abs.(clip(res_landing_pca, res_cap_pca)), ε)
clipped_landing_proc = max.(abs.(clip(res_landing_proc, res_cap_proc)), ε)

best_idx_pca = argmin(abs.(replace(res_landing_pca, NaN => Inf)))
λ_landing_pca_opt = λ_grid_landing[best_idx_pca[1]]
α_landing_pca_opt = α_grid_landing[best_idx_pca[2]]

best_idx_proc = argmin(abs.(replace(res_landing_proc, NaN => Inf)))
λ_landing_proc_opt = λ_grid_landing[best_idx_proc[1]]
α_landing_proc_opt = α_grid_landing[best_idx_proc[2]]

println("(α, λ) optimal Landing - PCA:        (", α_landing_pca_opt, ", ", λ_landing_pca_opt, ")")
println("(α, λ) optimal Landing - Procrustes: (", α_landing_proc_opt, ", ", λ_landing_proc_opt, ")")

# ---------------------------------------------------------------------------
# Plot combiné : les 2 heatmaps Landing (PCA, Procrustes) côte à côte
# ---------------------------------------------------------------------------

pl_landing_pca = heatmap(
    α_grid_landing, λ_grid_landing, log10.(clipped_landing_pca),
    xlabel=L"\alpha", ylabel=L"\lambda", xaxis=:log10, yaxis=:log10,
    colorbar_title="\n" * L"\log_{10} p(X)", titlefontsize=12,
    clims=(log10(minimum(clipped_landing_pca)), log10(res_cap_pca)),
    c=cgrad(:plasma, rev=true), framestyle=:box,  left_margin=5Plots.mm, bottom_margin=6Plots.mm, right_margin=10Plots.mm,size=(900, 300)
)

xl_pca, yl_pca = xlims(pl_landing_pca), ylims(pl_landing_pca)
vline!(pl_landing_pca, α_grid_landing, color=:white, alpha=0.3, linewidth=0.5, xlims=xl_pca, ylims=yl_pca)
hline!(pl_landing_pca, λ_grid_landing, color=:white, alpha=0.3, linewidth=0.5, xlims=xl_pca, ylims=yl_pca)
scatter!(pl_landing_pca, [α_landing_pca_opt], [λ_landing_pca_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=:black, xlims=xl_pca, ylims=yl_pca)

pl_landing_proc = heatmap(
    α_grid_landing, λ_grid_landing, log10.(clipped_landing_proc),
    xlabel=L"\alpha", ylabel=L"\lambda", xaxis=:log10, yaxis=:log10,
    colorbar_title="\n" * L"\log_{10} p(X)", titlefontsize=12,
    clims=(log10(minimum(clipped_landing_proc)), log10(res_cap_proc)),
    c=cgrad(:plasma, rev=true), framestyle=:box,
)
xl_proc, yl_proc = xlims(pl_landing_proc), ylims(pl_landing_proc)
vline!(pl_landing_proc, α_grid_landing, color=:white, alpha=0.3, linewidth=0.5, xlims=xl_proc, ylims=yl_proc)
hline!(pl_landing_proc, λ_grid_landing, color=:white, alpha=0.3, linewidth=0.5, xlims=xl_proc, ylims=yl_proc)
scatter!(pl_landing_proc, [α_landing_proc_opt], [λ_landing_proc_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=:black, xlims=xl_proc, ylims=yl_proc)

pl_landing = plot(pl_landing_pca, pl_landing_proc, layout=(1, 2), size=(900, 300),
    left_margin=5Plots.mm, bottom_margin=8Plots.mm, right_margin=10Plots.mm)
savefig(pl_landing_pca, joinpath(plotsdir, "landing_alpha_lambda_combined.pdf"))

# ===========================================================================
# Plot combiné PCA seul : à gauche RGD/POGO (α), à droite heatmap Landing (α, λ)
# ===========================================================================

pl_alpha_pca = plot(xlabel=L"\alpha", ylabel=L"q(X)", xaxis=:log10, yaxis=:log10,
    framestyle=:box, grid=true, gridalpha=0.3)
plot!(pl_alpha_pca, α_grid, clipped_rgd_pca, label="RGD Polar", color=color_rgd_pca, marker=:circle, markersize=3)
plot!(pl_alpha_pca, α_grid, clipped_rgd_qr_pca, label="RGD QR", color=color_rgd_qr_pca, marker=:circle, markersize=3)
plot!(pl_alpha_pca, α_grid, clipped_pogo_pca, label="POGO", color=color_pogo_pca, marker=:circle, markersize=3)

xl_alpha_pca, yl_alpha_pca = xlims(pl_alpha_pca), ylims(pl_alpha_pca)
scatter!(pl_alpha_pca, [α_rgd_pca_opt], [res_rgd_pca_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_pca, xlims=xl_alpha_pca, ylims=yl_alpha_pca, label=nothing)
scatter!(pl_alpha_pca, [α_rgd_qr_pca_opt], [res_rgd_qr_pca_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_qr_pca, xlims=xl_alpha_pca, ylims=yl_alpha_pca, label=nothing)
scatter!(pl_alpha_pca, [α_pogo_pca_opt], [res_pogo_pca_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_pogo_pca, xlims=xl_alpha_pca, ylims=yl_alpha_pca, label=nothing)
pl_alpha_pca = plot(pl_alpha_pca, legend=:topleft, legendfontsize=7)

pl_pca_params = plot(pl_alpha_pca, pl_landing_pca, layout=(1, 2), size=(900, 300),
    left_margin=5Plots.mm, bottom_margin=8Plots.mm, right_margin=10Plots.mm)
savefig(pl_pca_params, joinpath(plotsdir, "pca_params.pdf"))

# ===========================================================================
# Plot combiné Procrustes seul : à gauche RGD/POGO (α), à droite heatmap Landing (α, λ)
# ===========================================================================

pl_alpha_proc = plot(xlabel=L"\alpha", ylabel=L"p(X)", xaxis=:log10, yaxis=:log10,
    framestyle=:box, grid=true, gridalpha=0.3)
plot!(pl_alpha_proc, α_grid, clipped_rgd_proc, label="RGD Polar", color=color_rgd_proc, marker=:circle, markersize=3)
plot!(pl_alpha_proc, α_grid, clipped_rgd_qr_proc, label="RGD QR", color=color_rgd_qr_proc, marker=:circle, markersize=3)
plot!(pl_alpha_proc, α_grid, clipped_pogo_proc, label="POGO", color=color_pogo_proc, marker=:circle, markersize=3)

xl_alpha_proc, yl_alpha_proc = xlims(pl_alpha_proc), ylims(pl_alpha_proc)
scatter!(pl_alpha_proc, [α_rgd_proc_opt], [res_rgd_proc_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_proc, xlims=xl_alpha_proc, ylims=yl_alpha_proc, label=nothing)
scatter!(pl_alpha_proc, [α_rgd_qr_proc_opt], [res_rgd_qr_proc_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_rgd_qr_proc, xlims=xl_alpha_proc, ylims=yl_alpha_proc, label=nothing)
scatter!(pl_alpha_proc, [α_pogo_proc_opt], [res_pogo_proc_opt], marker=:star5, markersize=10,
    color=:white, markerstrokecolor=color_pogo_proc, xlims=xl_alpha_proc, ylims=yl_alpha_proc, label=nothing)
pl_alpha_proc = plot(pl_alpha_proc, legend=:topleft, legendfontsize=7)

pl_proc_params = plot(pl_alpha_proc, pl_landing_proc, layout=(1, 2), size=(900, 300),
    left_margin=5Plots.mm, bottom_margin=8Plots.mm, right_margin=10Plots.mm)
savefig(pl_proc_params, joinpath(plotsdir, "procrustes_params.pdf"))
