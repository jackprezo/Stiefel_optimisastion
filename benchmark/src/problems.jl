function check_problem_dimensions(n, p)
    n isa Integer && p isa Integer && 1 <= p <= n ||
        throw(ArgumentError("problem dimensions must satisfy 1 ≤ p ≤ n"))
    return nothing
end

"""Generate a weighted PCA problem with a Gaussian matrix and a starting point."""
function make_pca_problem(n, p; rng=Random.default_rng())
    check_problem_dimensions(n, p)
    C = randn(rng, n, n)
    A = C' * C
    X0 = randn(rng, n, p)
    return make_pca_problem(A, X0)
end

"""Build a weighted PCA problem from a real symmetric A and a starting point."""
function make_pca_problem(A::AbstractMatrix{<:Real}, X0::AbstractMatrix{<:Real})
    n, p = size(X0)
    check_problem_dimensions(n, p)
    size(A) == (n, n) || throw(ArgumentError("A must be square with size matching the rows of X0"))
    weights = collect(p:-1:1)
    U = Diagonal(weights)
    solution = eigen(Symmetric(A)).vectors[:, end:-1:end-p+1]
    f(X) = -tr(X' * A * X * U)
    optimum = f(solution)
    q(X) = (f(X) - optimum) / abs(optimum)
    function gradient!(out, X)
        mul!(out, A, X)
        out .*= -2 .* weights'
        return out
    end
    return (; f, q, gradient!, X0, A, solution)
end

"""Generate a Procrustes problem; normalize residuals by B or A."""
function make_procrustes_problem(n, p; rng=Random.default_rng(), normalization=:target)
    check_problem_dimensions(n, p)
    normalization in (:target, :matrix) ||
        throw(ArgumentError("normalization must be :target or :matrix"))
    C = randn(rng, n, n)
    A = C' * C
    solution = Matrix(qr(randn(rng, n, p)).Q)
    B = A * solution
    AtA = A' * A
    AtB = A' * B
    f(X) = norm(A * X - B)^2
    denominator = normalization === :target ? norm(B)^2 : norm(A)^2
    q(X) = f(X) / denominator
    function gradient!(out, X)
        mul!(out, AtA, X)
        out .*= 2
        out .-= 2 .* AtB
        return out
    end
    X0 = randn(rng, n, p)
    return (; f, q, gradient!, X0, A, B, solution)
end
