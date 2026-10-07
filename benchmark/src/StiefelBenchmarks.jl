module StiefelBenchmarks

using LinearAlgebra
using Random

export rgd, landing_flow, pogo, make_pca_problem, make_procrustes_problem,
       stiefel_distance, canonical_gradient!, pogo_direction!, pogo_direction,
       pogo_intermediate, pogo_step, retract_polar!, retract_qr!

include("solvers.jl")
include("problems.jl")

end
