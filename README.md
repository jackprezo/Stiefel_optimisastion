# Stiefel optimization benchmarks

Julia experiments comparing RGD (polar and QR), Landing, and POGO on PCA and Procrustes problems.

## Setup

Use Julia 1.11.6 and run these commands from this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run

Run all experiments:

```sh
julia --project=. reproduce.jl
```

For a quick check:

```sh
julia --project=. reproduce.jl --quick
```

Figures are saved in `results/plots/` and numerical tables in `results/data/`. The full timing benchmark takes several minutes.

Each script can also run independently with `julia --project=. <script>`:

| Script | Experiment |
| --- | --- |
| `main.jl` | Compare convergence on PCA and Procrustes. |
| `params.jl` | Compare step sizes and Landing penalty weights. |
| `time.jl` | Measure execution time across problem sizes and BLAS thread counts. |
| `analysis.jl` | Check POGO stability and changes in orthogonality. |
| `visualize.jl` | Plot method trajectories on the unit circle. |

All scripts accept `--quick`, `--seed INT` (default `0`), and `--output-dir PATH` (default `results/`). Experiment parameters are defined in each script; shared algorithms are in `src/`.

On machines without a display, prefix plotting commands with `GKSwstype=100`.
