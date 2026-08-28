# Jack Prezerowitz
# Visualisation 2D : trajectoire de RGD et POGO sur le cercle unité
# (variété de Stiefel St(2,1) = cercle unité de R^2), pour 3 scénarios
# différents (orientation de A, donc de l'optimum V, et point de départ X0),
# à conditionnement κ fixe. Pour chaque scénario, un même pas α (optimal
# pour les deux méthodes conjointement) est utilisé.

using LinearAlgebra
using Plots
using LaTeXStrings

include("rgd.jl")
include("pogo.jl")
include("landing.jl")

plot_font = "Computer Modern"
default(fontfamily=plot_font, linewidth=2, framestyle=:box, label=nothing, grid=false,
    tickfontsize=8, guidefontsize=10)
resetfontsizes()

plotsdir = joinpath(@__DIR__, "plots")
mkpath(plotsdir)

plasma = cgrad(:plasma)
color_rgd = plasma[0.1]
color_landing = plasma[0.5]
color_pogo = plasma[0.9]

# ---------------------------------------------------------------------------
# Problème jouet en dimension 2 : n = 2, p = 1
# ---------------------------------------------------------------------------

niter = 5
τ, r = 0.5, 1e-4
λ_pogo = 0.5
α_landing, λ_landing = 0.5, 1.0   # pas propre à Landing, trouvé par balayage
κ = 2.0   # conditionnement fixe pour les 3 scénarios

rot(φ) = [cos(φ) -sin(φ); sin(φ) cos(φ)]

scenarios = [
    (0.0, [-0.3, -1.0], 0.9),
    (π / 4, [1.0, 0.0], 0.9),
    (π / 2, [0.9, -0.436], 0.9),
]

silent(f) = redirect_stdout(f, devnull)
fmt(v) = "($(round(v[1], digits=2)), $(round(v[2], digits=2)))"

# Première passe : calcule (iter_rgd, iter_pogo, V, bbox) pour chaque scénario.
results = []
for (φ, x0_raw, α) in scenarios
    R = rot(φ)
    A = R * Diagonal([κ, 1.0]) * R'
    X0 = (x0_raw ./ norm(x0_raw))[:, :]

    f(X) = -tr(X' * A * X)
    V = eigen(Symmetric(A)).vectors[:, end:end]   # solution optimale
    @views function ∇f!(out, X)
        mul!(out, A, X)
        out .*= -2
        return out
    end
    grad_C(∇f, X) = ∇f - X * ∇f' * X
    q(X) = (f(X) - f(V)) / abs(f(V))

    _, res_rgd, _, _, iter_rgd = silent() do
        rgd(X0, ∇f!, f, q, grad_C, α, τ, r; niter=niter)
    end
    _, res_landing, _, _, iter_landing = silent() do
        landing_flow(∇f!, f, q, X0, λ_landing, α_landing, τ, r; niter=niter)
    end
    _, res_pogo, _, _, iter_pogo = silent() do
        pogo(X0, ∇f!, f, q, α, τ, r, λ_pogo; niter=niter)
    end

    println("X0=$(fmt(iter_rgd[:, 1, 1])) X*=$(fmt(V)) (φ=$φ, α=$α)  Résidu final RGD: ", res_rgd[end],
        "  Landing: ", res_landing[end], "  POGO: ", res_pogo[end])

    push!(results, (iter_rgd, iter_landing, iter_pogo, V))
end

# Étendue commune : le plus grand écart (X ou Y) observé sur tous les
# scénarios + marge, appliqué de façon identique aux 3 panneaux pour que
# chaque sous-figure occupe le même espace, centrée sur sa propre trajectoire.
pad = 0.05
span = maximum(r -> begin
        all_x = vcat(r[1][1, 1, :], r[2][1, 1, :], r[3][1, 1, :], r[4][1])
        all_y = vcat(r[1][2, 1, :], r[2][2, 1, :], r[3][2, 1, :], r[4][2])
        max(maximum(all_x) - minimum(all_x), maximum(all_y) - minimum(all_y))
    end, results) + 2pad

subplots = []
for (i, (iter_rgd, iter_landing, iter_pogo, V)) in enumerate(results)
    all_x = vcat(iter_rgd[1, 1, :], iter_landing[1, 1, :], iter_pogo[1, 1, :], V[1])
    all_y = vcat(iter_rgd[2, 1, :], iter_landing[2, 1, :], iter_pogo[2, 1, :], V[2])
    cx = (minimum(all_x) + maximum(all_x)) / 2
    cy = (minimum(all_y) + maximum(all_y)) / 2
    xl = (cx - span / 2, cx + span / 2)
    yl = (cy - span / 2, cy + span / 2)

    # Seul le dernier panneau affiche une légende, à l'intérieur du subplot.
    leg = i == length(results) ? :bottomright : false

    θ = range(0, 2π, length=400)
    pl = plot(cos.(θ), sin.(θ), aspect_ratio=:equal, color=:gray, alpha=0.5,
        label="St(2,1)", xlabel=L"X_1", ylabel=L"X_2", xlims=xl, ylims=yl,
        legend=leg, legendfontsize=6)

    plot!(pl, iter_rgd[1, 1, :], iter_rgd[2, 1, :],
        color=color_rgd, marker=:circle, markersize=6, linewidth=2.5, label="RGD")
    plot!(pl, iter_landing[1, 1, :], iter_landing[2, 1, :],
        color=color_landing, marker=:circle, markersize=6, linewidth=2.5, linestyle=:dashdot, label="Landing")
    plot!(pl, iter_pogo[1, 1, :], iter_pogo[2, 1, :],
        color=color_pogo, marker=:circle, markersize=6, linewidth=2.5, linestyle=:dash, label="POGO")

    scatter!(pl, [iter_rgd[1, 1, 1]], [iter_rgd[2, 1, 1]], color=:red, marker=:star5, markersize=10, label="Départ")
    scatter!(pl, [V[1]], [V[2]], color=:green, marker=:star5, markersize=10, label="Optimum")

    push!(subplots, pl)
end

pl_all = plot(subplots..., layout=(1, 3), size=(900, 300),
    left_margin=3.5Plots.mm, right_margin=1Plots.mm, top_margin=1Plots.mm, bottom_margin=5Plots.mm)
savefig(pl_all, joinpath(plotsdir, "trajectory_circle_scenarios.pdf"))
