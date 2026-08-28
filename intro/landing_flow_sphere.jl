using LinearAlgebra

# Jack Prezerowitz
# 10/08/2026

# Implementation of the landing flow algorithm on the sphere.

n = 100
B = randn(n, n)
A = B * B'

ϵ = 0.001
α = 0.001
β = 0.1

f(x) = -x' * A * x
∇f(x) = -2 * A * x

ψ(x) = 1/2 *  (dot(x, x) - 1)^2 
∇ψ(x) =  (dot(x, x) - 1) * x 




function landing_flow(∇f, ∇ψ, x0, α, β; niter=10000)
    x = x0 / norm(x0)
    for i in 1:niter
        gf = ∇f(x)
        gradf = gf - dot(x, gf) * x / norm(x)^2
        

        x = x - α * gradf - β * ∇ψ(x)
        println("Iteration: ", i, " Function value: ", f(x))
    end
    return x
end

function main_landing_flow()
    x0 = randn(n)
    x_opt = landing_flow(∇f, ∇ψ, x0, α, β; niter=1000)
    println("Eigenvalues of A: ", eigvals(A))
    println("Optimal point: ", x_opt)
    println("Function value at optimal point: ", f(x_opt))
    println("Residual: ", norm(A*x_opt + f(x_opt) * x_opt))
end

main_landing_flow()