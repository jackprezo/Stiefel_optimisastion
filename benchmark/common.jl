using LinearAlgebra
using Random

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results")

function initialize_experiment(seed=0)
    BLAS.set_num_threads(1)
    return MersenneTwister(seed)
end

function output_paths(output_dir=DEFAULT_OUTPUT_DIR)
    paths = (data=joinpath(abspath(output_dir), "data"),
             plots=joinpath(abspath(output_dir), "plots"))
    mkpath(paths.data)
    mkpath(paths.plots)
    return paths
end

function configure_plots(; scale=1.3, kwargs...)
    Plots.default()
    Plots.default(fontfamily="Computer Modern", linewidth=2, framestyle=:box,
                  label=nothing, grid=false; kwargs...)
    Plots.resetfontsizes()
    Plots.scalefontsizes(scale)
end

logclip(values; floor=1e-16) = map(x -> isfinite(x) && x > floor ? x : floor, values)

is_numerical_failure(error) = error isa DomainError ||
    (error isa ErrorException && startswith(error.msg, "polar retraction"))

function experiment_options(args)
    output_dir, seed, quick = DEFAULT_OUTPUT_DIR, 0, false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--quick"
            quick = true
        elseif arg in ("--seed", "--output-dir")
            i += 1
            i <= length(args) || throw(ArgumentError("Missing value for $arg"))
            if arg == "--seed"
                seed = parse(Int, args[i])
                seed >= 0 || throw(ArgumentError("Seed must be nonnegative"))
            else
                output_dir = args[i]
            end
        elseif arg in ("--help", "-h")
            println("Usage: julia --project=. $(basename(PROGRAM_FILE)) [--quick] [--seed INT] [--output-dir PATH]")
            exit(0)
        else
            throw(ArgumentError("Unknown option: $arg"))
        end
        i += 1
    end
    return (; output_dir, seed, quick)
end
