using LinearAlgebra

@views function pogo(X0, ∇f!, f, q, α_init, τ, r, λ; niter=1000, backtracking=false)

    n, p = size(X0)
    X = Matrix(qr(X0).Q)
    I_p = Matrix{Float64}(I, p, p)

    fvals = zeros(niter)
    distance = zeros(niter)
    residual = zeros(niter)
    iterates = Array{eltype(X)}(undef, n, p, niter)

    ∇f_X = Matrix{eltype(X)}(undef, n, p)
    XX = Matrix{eltype(X)}(undef, p, p)   # X'X, then ∇f_X'X, then M'M
    ψ = Matrix{eltype(X)}(undef, n, p)    # ∇f_X*(X'X), then ψ
    MM = Matrix{eltype(X)}(undef, n, p)   # X*(∇f_X'X), then M*(M'M)
    M = Matrix{eltype(X)}(undef, n, p)

    for i in 1:niter
        f_X = f(X)
        fvals[i] = f_X
        distance[i] = norm(X' * X - I_p)
        residual[i] = q(X)
        iterates[:, :, i] = X

        println("[pogo] Iter $i | dist Stiefel: ", distance[i])
        println("[pogo] residual: ", residual[i])

        ∇f!(∇f_X, X)

        # ψ = 0.5 * (∇f_X * (X'X) - X * (∇f_X'X))
        mul!(XX, X', X)
        mul!(ψ, ∇f_X, XX)
        mul!(XX, ∇f_X', X)
        mul!(MM, X, XX)
        ψ .=  (ψ .- MM)

        α = α_init

        # M = X - α * ψ
        @. M = X - α * ψ

        # X = M + λ * M * (I_p - M'M) = (1+λ)*M - λ * M * (M'M)
        mul!(XX, M', M)
        mul!(MM, M, XX)
        @. X = (1 + λ) * M - λ * MM
    end

    return X, residual, distance, fvals, iterates
end
