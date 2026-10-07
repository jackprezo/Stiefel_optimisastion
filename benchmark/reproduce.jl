for script in ("main.jl", "params.jl", "time.jl", "analysis.jl", "visualize.jl")
    include(script)
end

function reproduce(; output_dir=DEFAULT_OUTPUT_DIR, seed=0, quick=false)
    for experiment in (run_main, run_params, run_time, run_analysis, run_visualize)
        experiment(; output_dir, seed, quick)
    end
    println("All experiments completed. Results: ", abspath(output_dir))
end

if abspath(PROGRAM_FILE) == @__FILE__
    reproduce(; experiment_options(ARGS)...)
end
