using LinearAlgebra

# Jack Prezerowitz
# 10/08/2026

# Implementation of the RGD algorithm on the sphere. 

n = 100
B = randn(n, n)
A = B * B'

f(x) = -x' * A * x
∇f(x) = -2 * A * x

α = 0.001

function rgd(∇f, x0, α; niter=10000)
    x = x0 / norm(x0) 

    for i in 1:niter
        gf = ∇f(x)
        gradf = gf - dot(x, gf)  * x
        a = x - α * gradf
        x = (a) / norm(a)
        println("Iteration: ", i, " Function value: ", f(x))
    end

    return x
end


