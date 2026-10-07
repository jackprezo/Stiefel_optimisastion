using LinearAlgebra
using Printf
using Random
using Plots
using LaTeXStrings
using StiefelBenchmarks

isdefined(@__MODULE__, :initialize_experiment) || include("common.jl")

"""Construct a starting matrix with a prescribed Stiefel deviation."""
function make_deviated_start(n, p, target_deviation; rng=Random.default_rng())
    1 <= p <= n || throw(ArgumentError("Expected 1 <= p <= n."))
    isfinite(target_deviation) && target_deviation > 0 ||
        throw(ArgumentError("The target deviation must be positive and finite."))

    Q = Matrix(qr(randn(rng, n, p)).Q)
    perturbation = randn(rng, n, p)
    perturbation ./= norm(perturbation)
    deviation_at(scale) = stiefel_distance(Q + scale .* perturbation)

    lo, hi = 0.0, 1.0
    for _ in 1:64
        deviation_at(hi) >= target_deviation && break
        hi *= 2
    end
    deviation_at(hi) >= target_deviation ||
        error("Could not bracket the requested initial deviation.")
    for _ in 1:80
        mid = (lo + hi) / 2
        if deviation_at(mid) < target_deviation
            lo = mid
        else
            hi = mid
        end
    end
    return Q + ((lo + hi) / 2) .* perturbation
end

"""Evaluate the deviation after one POGO step, treating divergence as infinite."""
function pogo_step_deviation(gradient!, X, alpha)
    direction = pogo_direction(gradient!, X)
    deviation = stiefel_distance(pogo_step(X, direction, alpha))
    return (!isfinite(deviation) || deviation > 1e4) ? Inf : deviation
end

"""Find the first unstable step ratio by geometric bisection."""
function stability_boundary(gradient!, X, epsilon, alpha_theory; bisections=40)
    isfinite(alpha_theory) && alpha_theory > 0 ||
        error("The theoretical step must be positive and finite.")
    stable(ratio) = pogo_step_deviation(gradient!, X, ratio * alpha_theory) <= epsilon

    lo_ratio, hi_ratio = 1.0, 2.0
    # Roundoff can move the theoretical endpoint outside the stable interval.
    for _ in 1:64
        stable(lo_ratio) && break
        hi_ratio = lo_ratio
        lo_ratio /= 2
    end
    stable(lo_ratio) || error("Could not find a stable lower step bound.")
    for _ in 1:64
        !stable(hi_ratio) && break
        hi_ratio *= 2
    end
    !stable(hi_ratio) || error("Could not find an unstable upper step bound.")

    for _ in 1:bisections
        mid_ratio = sqrt(lo_ratio * hi_ratio)
        if stable(mid_ratio)
            lo_ratio = mid_ratio
        else
            hi_ratio = mid_ratio
        end
    end
    return hi_ratio
end

"""Recompute the experimental step bound from the current POGO direction."""
function adaptive_step_bound(X, direction)
    deviation = stiefel_distance(X)
    direction_norm = norm(direction)
    (deviation <= 0 || deviation >= 1 || direction_norm == 0) && return 0.0
    return sqrt((sqrt(deviation) - deviation) /
                ((1.0 + deviation) * direction_norm^2))
end

"""Record deviations, including the initial state, with an adaptive step size."""
function adaptive_deviation_trajectory(gradient!, X0; nsteps)
    deviations = zeros(nsteps + 1)
    alphas = zeros(nsteps)
    X = copy(X0)
    deviations[1] = stiefel_distance(X)
    for k in 1:nsteps
        direction = pogo_direction(gradient!, X)
        alpha = adaptive_step_bound(X, direction)
        alphas[k] = alpha
        X = pogo_step(X, direction, alpha)
        deviation = stiefel_distance(X)
        deviations[k + 1] = (!isfinite(deviation) || deviation > 1e6) ? Inf : deviation
    end
    return deviations, alphas
end

"""Run the stability boundary sweep and adaptive deviation experiment."""
function run_analysis(; output_dir=DEFAULT_OUTPUT_DIR, seed=0, quick=false)
    rng = initialize_experiment(seed)
    paths = output_paths(output_dir)
    epsilons = quick ? [1e-2, 1e-8] : [1e-2, 1e-5, 1e-8, 1e-12]
    sizes = quick ? [(20, 3)] : [(100, 10), (500, 50), (1000, 100)]
    problems = [("PCA", make_pca_problem), ("Procrustes", make_procrustes_problem)]
    nsteps = quick ? 8 : 15
    initial_deviations = quick ? [1e-2, 1e-6, 1e-12] : [1e-2, 1e-4, 1e-6, 1e-9, 1e-12]
    trajectory_size = quick ? (20, 3) : (100, 10)
    outfile = joinpath(paths.data, "analysis_stability.txt")

    open(outfile, "w") do io
        @printf(io, "%-12s %-6s %-6s %-12s %-14s %-14s %-12s %-10s %-14s %s\n",
            "problem", "n", "p", "epsilon", "norm_psi0", "alpha_theory", "alpha",
            "ratio", "dev_1step", "status")
        for (problem_name, make_problem) in problems
            for (n, p) in sizes
                gradient! = make_problem(n, p; rng=rng).gradient!
                for epsilon in epsilons
                    X0 = make_deviated_start(n, p, epsilon; rng=rng)
                    gradient = similar(X0)
                    gradient!(gradient, X0)
                    skew = gradient * X0' - X0 * gradient'
                    skew_norm = norm(skew)
                    # The boundary uses the full n-by-n skew matrix norm.
                    alpha_theory = sqrt((sqrt(epsilon) - epsilon) /
                                        ((1.0 + epsilon) * skew_norm^2))
                    ratio = stability_boundary(gradient!, X0, epsilon, alpha_theory)
                    alpha = ratio * alpha_theory
                    deviation = pogo_step_deviation(gradient!, X0, alpha)
                    status = isfinite(deviation) ? "INVALID" : "DIVERGENCE"
                    @printf(io, "%-12s %-6d %-6d %-12.4g %-14.6e %-14.6e %-12.6e %-10.4f %-14.6e %s\n",
                        problem_name, n, p, epsilon, skew_norm, alpha_theory, alpha,
                        ratio, deviation, status)
                    @printf("%s n=%d p=%d epsilon=%.2e: boundary/theory=%.4f\n",
                        problem_name, n, p, epsilon, ratio)
                    flush(io)
                end
            end
        end
    end

    # The adaptive experiment uses the n-by-p direction norm, as in the original
    # experiment. It is distinct from the full skew norm in the boundary sweep.
    rng = initialize_experiment(seed)
    n, p = trajectory_size
    gradient! = make_pca_problem(n, p; rng=rng).gradient!
    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    markers = [:circle, :rect, :utriangle, :diamond, :star5]
    palette = [cgrad(:plasma)[t] for t in range(0.05, 0.85, length=length(initial_deviations))]
    configure_plots(minorticks=true, tick_direction=:in)
    trajectory_plot = plot(xlabel="Iteration", ylabel=L"\|| X^\top X - I_p\||_F",
        yaxis=:log10, legend=:topright, legendfontsize=9, size=(1000, 300),
        left_margin=6Plots.mm, bottom_margin=7Plots.mm)
    trajectories = Vector{Vector{Float64}}()
    trajectory_alphas = Vector{Vector{Float64}}()
    for (index, epsilon) in enumerate(initial_deviations)
        X0 = make_deviated_start(n, p, epsilon; rng=rng)
        deviations, alphas = adaptive_deviation_trajectory(gradient!, X0; nsteps=nsteps)
        push!(trajectories, deviations)
        push!(trajectory_alphas, alphas)
        color = palette[index]
        iterations = 0:nsteps
        values = logclip(deviations; floor=1e-300)
        plot!(trajectory_plot, iterations, values, color=color, linewidth=3,
            linestyle=linestyles[index],
            label=L"\varepsilon_0 = 10^{%$(round(Int, log10(epsilon)))}")
        marker_indices = 1:6:length(iterations)
        scatter!(trajectory_plot, collect(iterations)[marker_indices], values[marker_indices],
            color=color, marker=markers[index], markersize=7, markerstrokecolor=:black,
            markerstrokewidth=1, label="")
        hline!(trajectory_plot, [epsilon], color=color, linestyle=:dot, linewidth=2, label="")
    end
    trajectory_datafile = joinpath(paths.data, "analysis_deviation_trajectory.txt")
    open(trajectory_datafile, "w") do io
        println(io, "epsilon0 iteration deviation alpha")
        for (index, epsilon) in enumerate(initial_deviations)
            for iteration in 0:nsteps
                # Alpha is the outgoing step; the final state has no next step.
                alpha = iteration < nsteps ? trajectory_alphas[index][iteration + 1] : NaN
                @printf(io, "%.17e %d %.17e %.17e\n", epsilon, iteration,
                    trajectories[index][iteration + 1], alpha)
            end
        end
    end
    plotfile = joinpath(paths.plots, "analysis_deviation_trajectory.pdf")
    savefig(trajectory_plot, plotfile)
    println("Stability results saved to ", outfile)
    println("Deviation plot saved to ", plotfile)
    return (; datafile=outfile, trajectory_datafile, plotfile, trajectories)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis(; experiment_options(ARGS)...)
end
