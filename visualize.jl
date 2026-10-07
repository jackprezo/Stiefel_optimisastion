using LinearAlgebra
using Printf
using Plots
using LaTeXStrings
using StiefelBenchmarks

isdefined(@__MODULE__, :initialize_experiment) || include("common.jl")

"""Construct a two-dimensional PCA problem for a circle trajectory."""
function make_circle_problem(angle, initial_point, condition_number)
    rotation = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    A = rotation * Diagonal([condition_number, 1.0]) * rotation'
    X0 = reshape(initial_point ./ norm(initial_point), 2, 1)
    return make_pca_problem(A, X0)
end

function circle_trajectory_coordinates(result)
    x = vcat(result.rgd.iterates[1, 1, :], result.landing.iterates[1, 1, :],
        result.pogo.iterates[1, 1, :], result.intermediates[1, 1, :], result.problem.solution[1])
    y = vcat(result.rgd.iterates[2, 1, :], result.landing.iterates[2, 1, :],
        result.pogo.iterates[2, 1, :], result.intermediates[2, 1, :], result.problem.solution[2])
    return x, y
end

"""Plot RGD, landing, and POGO trajectories on St(2, 1) for three scenarios."""
function run_visualize(; output_dir=DEFAULT_OUTPUT_DIR, seed=0, quick=false)
    initialize_experiment(seed)
    paths = output_paths(output_dir)
    configure_plots(; scale=1.0, tickfontsize=8, guidefontsize=10)
    plasma = cgrad(:plasma)
    color_rgd, color_landing, color_pogo = plasma[0.1], plasma[0.5], plasma[0.9]

    niter = 4
    lambda_pogo = 0.5
    alpha_landing, lambda_landing = 0.5, 1.6
    condition_number = 2.0
    scenarios = [
        (0.0, [-0.3, -1.0], 0.65),
        (pi / 4, [1.0, 0.0], 0.6),
        (pi / 2, [0.9, -0.436], 0.65),
    ]
    results = map(scenarios) do (angle, initial_point, alpha)
        problem = make_circle_problem(angle, initial_point, condition_number)
        rgd_result = rgd(problem.X0, problem.gradient!, problem.f, problem.q, alpha; niter=niter)
        landing_result = landing_flow(problem.X0, problem.gradient!, problem.f, problem.q,
            lambda_landing, alpha_landing; niter=niter)
        pogo_result = pogo(problem.X0, problem.gradient!, problem.f, problem.q, alpha,
            lambda_pogo; niter=niter)
        intermediates = similar(pogo_result.iterates)
        for k in 1:niter
            intermediates[:, :, k] = pogo_intermediate(pogo_result.iterates[:, :, k],
                problem.gradient!, alpha)
        end
        println("Circle scenario angle=", angle, ", alpha=", alpha,
            ": last recorded residuals RGD=", rgd_result.residual[end],
            ", landing=", landing_result.residual[end], ", POGO=", pogo_result.residual[end])
        (; rgd=rgd_result, landing=landing_result, pogo=pogo_result, intermediates, problem)
    end

    datafile = joinpath(paths.data, "trajectory_circle_scenarios.txt")
    open(datafile, "w") do io
        println(io, "scenario method iteration x y")
        for (scenario, result) in enumerate(results)
            histories = [("RGD", result.rgd.iterates), ("Landing", result.landing.iterates),
                ("POGO", result.pogo.iterates), ("POGO_M", result.intermediates)]
            for (method, history) in histories
                for iteration in 0:(niter - 1)
                    @printf(io, "%d %s %d %.17e %.17e\n", scenario, method, iteration,
                        history[1, 1, iteration + 1], history[2, 1, iteration + 1])
                end
            end
        end
    end

    padding = 0.05
    span = maximum(results) do result
        x, y = circle_trajectory_coordinates(result)
        max(maximum(x) - minimum(x), maximum(y) - minimum(y))
    end + 2padding

    subplots = map(enumerate(results)) do (index, result)
        x, y = circle_trajectory_coordinates(result)
        center_x = (minimum(x) + maximum(x)) / 2
        center_y = (minimum(y) + maximum(y)) / 2
        xlimits = (center_x - span / 2, center_x + span / 2)
        ylimits = (center_y - span / 2, center_y + span / 2)
        grid_size = quick ? 75 : 150
        xs = range(xlimits..., length=grid_size)
        ys = range(ylimits..., length=grid_size)
        A = result.problem.A
        objective_grid(x1, x2) = -[x1, x2]' * A * [x1, x2]
        values = [objective_grid(x, y) for y in ys, x in xs]
        subplot = contour(xs, ys, values, color=:grays, alpha=0.35, levels=12,
            linewidth=0.7, aspect_ratio=:equal, xlabel=L"x_1", ylabel=L"x_2",
            xlims=xlimits, ylims=ylimits, legend=index == length(results) ? :bottomright : false,
            legendfontsize=6, colorbar=false)

        angles = range(0, 2pi, length=400)
        plot!(subplot, cos.(angles), sin.(angles), color=:black, alpha=1, label="St(2,1)")
        iter_rgd = result.rgd.iterates
        iter_landing = result.landing.iterates
        iter_pogo = result.pogo.iterates
        plot!(subplot, iter_rgd[1, 1, :], iter_rgd[2, 1, :],
            color=color_rgd, marker=:circle, markersize=6, label="RGD")
        plot!(subplot, iter_landing[1, 1, :], iter_landing[2, 1, :], color=color_landing,
            marker=:diamond, markersize=6, linewidth=2.5, linestyle=:dashdot, label="Landing")

        # Solvers record pre-update states, so the line ends at the final M_k.
        sequence_x = Vector{Float64}(undef, 2niter)
        sequence_y = Vector{Float64}(undef, 2niter)
        for k in 1:niter
            sequence_x[2k - 1], sequence_y[2k - 1] = iter_pogo[1, 1, k], iter_pogo[2, 1, k]
            sequence_x[2k], sequence_y[2k] = result.intermediates[1, 1, k], result.intermediates[2, 1, k]
        end
        plot!(subplot, sequence_x, sequence_y, color=color_pogo, linewidth=2.5,
            linestyle=:dot, label="POGO")
        scatter!(subplot, iter_pogo[1, 1, :], iter_pogo[2, 1, :], color=color_pogo,
            marker=:utriangle, markersize=7, label=L"\mathrm{POGO}(X_k)")
        scatter!(subplot, [iter_rgd[1, 1, 1]], [iter_rgd[2, 1, 1]], color=:red,
            marker=:rect, markersize=7, label=L"X_0")
        solution = result.problem.solution
        scatter!(subplot, [solution[1]], [solution[2]], color=:green, marker=:star5,
            markersize=12, label="Optimum")
        subplot
    end

    figure = plot(subplots..., layout=(1, 3), size=(900, 300), left_margin=3.5Plots.mm,
        right_margin=1Plots.mm, top_margin=1Plots.mm, bottom_margin=5Plots.mm)
    plotfile = joinpath(paths.plots, "trajectory_circle_scenarios.pdf")
    savefig(figure, plotfile)
    println("Circle trajectory plot saved to ", plotfile)
    return (; datafile, plotfile, results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_visualize(; experiment_options(ARGS)...)
end
