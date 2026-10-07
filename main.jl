using StiefelBenchmarks
using Plots
using LaTeXStrings
using Printf

isdefined(@__MODULE__, :initialize_experiment) || include("common.jl")

function plot_marked!(figure, values; label, color, marker, every=300,
                      linestyle=:solid, linewidth=2)
    plot!(figure, values; label, color, linestyle, linewidth)
    indices = firstindex(values):every:lastindex(values)
    scatter!(figure, collect(indices), values[indices]; label="", color, marker,
             markersize=5, markerstrokecolor=:black, markerstrokewidth=1)
    return figure
end

function run_main(; output_dir=DEFAULT_OUTPUT_DIR, seed=0, quick=false)
    rng = initialize_experiment(seed)
    paths = output_paths(output_dir)
    configure_plots(minorticks=true, tick_direction=:in)
    n, p, niter = 100, 10, quick ? 20 : 2000
    plasma = cgrad(:plasma)
    styles = (
        (label="RGD (Polar)", color=plasma[0.1], marker=:diamond, linestyle=:solid),
        (label="RGD (QR)", color=plasma[0.3], marker=:circle, linestyle=:solid),
        (label="Landing", color=plasma[0.6], marker=:utriangle, linestyle=:dot),
        (label="POGO", color=plasma[0.9], marker=:rect, linestyle=:dash))
    problems = (
        (name="PCA", file="pca", problem=make_pca_problem(n, p; rng),
         alpha_rgd=2e-4, alpha_landing=2e-4, alpha_pogo=2e-4, lambda=1e3,
         residual_label=L"q(X)"),
        (name="Procrustes", file="procrustes",
         problem=make_procrustes_problem(n, p; rng, normalization=:matrix),
         alpha_rgd=4.328761281083061e-6, alpha_landing=4.641588833612782e-6,
         alpha_pogo=4.328761281083061e-6, lambda=46415.88833612782,
         residual_label=L"p(X)"))

    open(joinpath(paths.data, "methods_comparison.txt"), "w") do io
        println(io, "problem\tmethod\titeration\talpha\tlambda\tresidual\tstiefel_distance\tobjective")
        for config in problems
            problem = config.problem
            X0, gradient!, f, q = problem.X0, problem.gradient!, problem.f, problem.q
            histories = (
                rgd(X0, gradient!, f, q, config.alpha_rgd; niter, retract=:polar, store_iterates=false),
                rgd(X0, gradient!, f, q, config.alpha_rgd; niter, retract=:qr, store_iterates=false),
                landing_flow(X0, gradient!, f, q, config.lambda, config.alpha_landing; niter, store_iterates=false),
                pogo(X0, gradient!, f, q, config.alpha_pogo, 0.5; niter, store_iterates=false))
            residual_plot = plot(xlabel="Iteration", ylabel=config.residual_label, yaxis=:log10)
            distance_plot = plot(xlabel="Iteration", ylabel=L"\Vert X^\top X - I_p \Vert_\mathrm{F}", yaxis=:log10)
            alphas = (config.alpha_rgd, config.alpha_rgd, config.alpha_landing, config.alpha_pogo)
            lambdas = (0.0, 0.0, config.lambda, 0.5)
            for (k, (history, style)) in enumerate(zip(histories, styles))
                plot_marked!(residual_plot, logclip(history.residual); style...,
                             every=quick ? 5 : 300, linewidth=k == 4 ? 3 : 2)
                plot_marked!(distance_plot, logclip(history.distance); style..., every=quick ? 5 : 300)
                for iteration in 1:niter
                    @printf(io, "%s\t%s\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",
                        config.name, style.label, iteration, alphas[k], lambdas[k],
                        history.residual[iteration], history.distance[iteration], history.fvals[iteration])
                end
            end
            figure = plot(residual_plot, distance_plot; layout=(1, 2), size=(1000, 300),
                          left_margin=5Plots.mm, bottom_margin=10Plots.mm)
            savefig(figure, joinpath(paths.plots, config.file * ".pdf"))
        end
    end
    println("Comparison results saved to ", abspath(output_dir))
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_main(; experiment_options(ARGS)...)
end
