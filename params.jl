using StiefelBenchmarks
using Plots
using LaTeXStrings
using Statistics
using Printf

isdefined(@__MODULE__, :initialize_experiment) || include("common.jl")

function sweep_residual(solve)
    try
        residual = solve().residual[end]
        return isfinite(residual) ? (residual, "FINITE") : (NaN, "NONFINITE")
    catch error
        if is_numerical_failure(error)
            return (NaN, "RETRACTION_FAILED")
        end
        rethrow()
    end
end

function residual_cap(values)
    finite = filter(isfinite, vec(values))
    return isempty(finite) ? 1.0 : max(quantile(finite, 0.9), 1e-16)
end

display_residuals(values, cap) = map(x -> isfinite(x) ? clamp(abs(x), 1e-16, cap) : cap, values)

function best_residual_index(values)
    any(isfinite, values) || return nothing
    scores = map(x -> isfinite(x) ? abs(x) : Inf, values)
    return argmin(scores)
end

function run_params(; output_dir=DEFAULT_OUTPUT_DIR, seed=0, quick=false)
    rng = initialize_experiment(seed)
    paths = output_paths(output_dir)
    configure_plots(minorticks=true, tick_direction=:in, tickfontsize=10)
    n, p, niter = 100, 10, quick ? 5 : 50
    alpha_grid = 10.0 .^ range(-6, -2; length=quick ? 4 : 12)
    landing_alpha_grid = 10.0 .^ range(-6, -2; length=quick ? 4 : 10)
    lambda_grid = 10.0 .^ range(0, 6; length=quick ? 4 : 10)
    problems = (
        (name="PCA", file="pca", problem=make_pca_problem(n, p; rng), label=L"q(X)", colors=(0.05, 0.2, 0.35)),
        (name="Procrustes", file="procrustes", problem=make_procrustes_problem(n, p; rng), label=L"p(X)", colors=(0.55, 0.7, 0.9)))
    sweep_results = []

    open(joinpath(paths.data, "parameter_sweep.txt"), "w") do io
        println(io, "problem\tmethod\talpha\tlambda\tresidual\tstatus")
        for config in problems
            problem = config.problem
            X0, gradient!, f, q = problem.X0, problem.gradient!, problem.f, problem.q
            methods = (
                ("RGD (Polar)", alpha -> rgd(X0, gradient!, f, q, alpha; niter, retract=:polar, store_iterates=false)),
                ("RGD (QR)", alpha -> rgd(X0, gradient!, f, q, alpha; niter, retract=:qr, store_iterates=false)),
                ("POGO", alpha -> pogo(X0, gradient!, f, q, alpha, 0.5; niter, store_iterates=false)))
            residuals = fill(NaN, length(alpha_grid), length(methods))
            for (column, (name, solve)) in enumerate(methods)
                for (row, alpha) in enumerate(alpha_grid)
                    residual, status = sweep_residual(() -> solve(alpha))
                    residuals[row, column] = residual
                    @printf(io, "%s\t%s\t%.17g\t%.17g\t%.17g\t%s\n",
                        config.name, name, alpha, name == "POGO" ? 0.5 : 0.0, residual, status)
                end
            end
            landing_residuals = fill(NaN, length(lambda_grid), length(landing_alpha_grid))
            for (i, lambda) in enumerate(lambda_grid), (j, alpha) in enumerate(landing_alpha_grid)
                residual, status = sweep_residual(
                    () -> landing_flow(X0, gradient!, f, q, lambda, alpha; niter, store_iterates=false))
                landing_residuals[i, j] = residual
                @printf(io, "%s\tLanding\t%.17g\t%.17g\t%.17g\t%s\n",
                        config.name, alpha, lambda, residual, status)
            end
            push!(sweep_results, (; config, methods, residuals, landing_residuals))
        end
    end

    # Share the one-dimensional color scale across the two problems.
    cap_1d = residual_cap(vcat([vec(result.residuals) for result in sweep_results]...))
    plasma = cgrad(:plasma)
    markers = (:diamond, :circle, :utriangle)
    linestyles = (:dashdot, :dot, :solid)
    for result in sweep_results
        config = result.config
        clipped = display_residuals(result.residuals, cap_1d)
        alpha_plot = plot(xlabel=L"\alpha", ylabel=config.label, xaxis=:log10, yaxis=:log10,
                          yguidefontsize=12, legend=:topleft, legendfontsize=7)
        for (column, (name, _)) in enumerate(result.methods)
            color = plasma[config.colors[column]]
            plot!(alpha_plot, alpha_grid, clipped[:, column]; label=name, color,
                  marker=markers[column], linestyle=linestyles[column], markersize=7, linewidth=2.5)
            best = best_residual_index(result.residuals[:, column])
            if best !== nothing
                limits = (xlims=xlims(alpha_plot), ylims=ylims(alpha_plot))
                scatter!(alpha_plot, [alpha_grid[best]], [clipped[best, column]];
                         marker=:star5, markersize=10, color=:white,
                         markerstrokecolor=color, label="", limits...)
                println(config.name, " ", name, " best grid alpha: ", alpha_grid[best])
            end
        end

        cap = residual_cap(result.landing_residuals)
        landing_values = display_residuals(result.landing_residuals, cap)
        color_min, color_max = log10(minimum(landing_values)), log10(cap)
        color_max = max(color_max, color_min + 1e-6)
        landing_plot = heatmap(landing_alpha_grid, lambda_grid, log10.(landing_values);
            xlabel=L"\alpha", ylabel=L"\lambda", xaxis=:log10, yaxis=:log10,
            colorbar_title="\n" * (config.name == "PCA" ? L"\log_{10} q(X)" : L"\log_{10} p(X)"),
            colorbar_titlefontsize=12, colorbar_tickfontsize=8,
            colorbar_formatter=x -> @sprintf("%.0f", x), clims=(color_min, color_max),
            c=cgrad(:plasma; rev=true), right_margin=12Plots.mm)
        limits = (xlims=xlims(landing_plot), ylims=ylims(landing_plot))
        vline!(landing_plot, landing_alpha_grid; color=:white, alpha=0.3, linewidth=0.5, limits...)
        hline!(landing_plot, lambda_grid; color=:white, alpha=0.3, linewidth=0.5, limits...)
        best = best_residual_index(result.landing_residuals)
        if best !== nothing
            alpha, lambda = landing_alpha_grid[best[2]], lambda_grid[best[1]]
            scatter!(landing_plot, [alpha], [lambda]; marker=:star5, markersize=10,
                     color=:white, markerstrokecolor=:black, label="", limits...)
            println(config.name, " Landing best grid (alpha, lambda): ", (alpha, lambda))
        end
        figure = plot(alpha_plot, landing_plot;
            layout=grid(1, 2; widths=[0.46, 0.54]), size=(880, 270),
            left_margin=4Plots.mm, bottom_margin=6Plots.mm, right_margin=6Plots.mm, top_margin=2Plots.mm)
        savefig(figure, joinpath(paths.plots, config.file * "_params.pdf"))
    end
    println("Parameter sweep results saved to ", abspath(output_dir))
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_params(; experiment_options(ARGS)...)
end
