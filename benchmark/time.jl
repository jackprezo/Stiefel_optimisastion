using StiefelBenchmarks
using BenchmarkTools
using Printf
using Statistics

isdefined(@__MODULE__, :initialize_experiment) || include("common.jl")

const PCA_TIMING_SETTINGS = [
    (100, 10, 0.0002442053094548652, 0.0002395026619987486, 3290.3445623126677, 0.0002442053094548652),
    (500, 50, 1.2328467394420658e-5, 5.994842503189409e-6, 21544.346900318822, 1.2328467394420658e-5),
    (1000, 100, 4.328761281083061e-6, 2.1544346900318822e-6, 166810.0537200059, 4.328761281083061e-6)]

const PROCRUSTES_TIMING_SETTINGS = [
    (100, 10, 4.328761281083061e-6, 4.641588833612782e-6, 46415.88833612782, 4.328761281083061e-6),
    (500, 50, 1.0e-6, 1.2915496650148827e-7, 100000.0, 1.0e-6),
    (1000, 100, 6.579332246575682e-8, 1.2915496650148827e-8, 1.0e6, 6.579332246575682e-8)]

function run_time(; output_dir=DEFAULT_OUTPUT_DIR, seed=0, quick=false,
                  seconds=quick ? 0.05 : 5.0, samples=quick ? 3 : 10000,
                  direction_scale=0.5)
    original_threads = BLAS.get_num_threads()
    rng = initialize_experiment(seed)
    paths = output_paths(output_dir)
    niter = 20
    thread_counts = quick ? (1:1) : (1:6)
    problems = (("PCA", make_pca_problem, PCA_TIMING_SETTINGS),
                ("Procrustes", make_procrustes_problem, PROCRUSTES_TIMING_SETTINGS))

    # Reuse each instance across thread counts so timing comparisons change only BLAS threads.
    instances = [(name, make_problem(n, p; rng), (n, p, alpha_rgd, alpha_landing, lambda, alpha_pogo))
        for (name, make_problem, settings) in problems
        for (n, p, alpha_rgd, alpha_landing, lambda, alpha_pogo) in (quick ? settings[1:1] : settings)]
    try
        open(joinpath(paths.data, "time_benchmark.txt"), "w") do io
            println(io, "threads\tproblem\tn\tp\tmethod\talpha\tlambda\tdirection_scale\tresidual\tstiefel_distance\tmean_seconds\tstd_seconds\tsamples\tstatus")
            for nthreads in thread_counts
                BLAS.set_num_threads(nthreads)
                for (name, problem, settings) in instances
                    n, p, alpha_rgd, alpha_landing, lambda, alpha_pogo = settings
                    X0, gradient!, f, q = problem.X0, problem.gradient!, problem.f, problem.q
                    methods = (
                        ("RGD (polar)", alpha_rgd, 0.0, 1.0,
                         () -> rgd(X0, gradient!, f, q, alpha_rgd; niter, retract=:polar, record_history=false).X),
                        ("RGD (qr)", alpha_rgd, 0.0, 1.0,
                         () -> rgd(X0, gradient!, f, q, alpha_rgd; niter, retract=:qr, record_history=false).X),
                        ("Landing", alpha_landing, lambda, 1.0,
                         () -> landing_flow(X0, gradient!, f, q, lambda, alpha_landing; niter, record_history=false).X),
                        ("POGO", alpha_pogo, 0.5, direction_scale,
                         () -> pogo(X0, gradient!, f, q, alpha_pogo, 0.5; niter, direction_scale, record_history=false).X))
                    for (method, alpha, weight, scale, solve) in methods
                        status = "FINITE"
                        residual, distance = NaN, NaN
                        mean_seconds, std_seconds, count = NaN, NaN, 0
                        try
                            X = solve() # Warm up compilation and check validity before timing.
                            residual, distance = q(X), stiefel_distance(X)
                            status = isfinite(residual) && isfinite(distance) ? "FINITE" : "NONFINITE"
                        catch error
                            is_numerical_failure(error) || rethrow()
                            status = "RETRACTION_FAILED"
                        end
                        if status == "FINITE"
                            trial = @benchmark $solve() seconds=seconds samples=samples evals=1
                            mean_seconds = mean(trial.times) / 1e9
                            std_seconds = std(trial.times) / 1e9
                            count = length(trial.times)
                        end
                        @printf(io, "%d\t%s\t%d\t%d\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%s\n",
                            nthreads, name, n, p, method, alpha, weight, scale, residual, distance,
                            mean_seconds, std_seconds, count, status)
                        flush(io)
                        @printf("threads=%d %s (%d,%d) %s: %.6g s [%s]\n", nthreads, name, n, p, method, mean_seconds, status)
                    end
                end
            end
        end
    finally
        BLAS.set_num_threads(original_threads)
    end
    println("Timing results saved to ", joinpath(paths.data, "time_benchmark.txt"))
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_time(; experiment_options(ARGS)...)
end
