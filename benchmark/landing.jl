using LinearAlgebra

"""
Landing Flow avec fonction de mérite (merit function)
"""
@views function landing_flow(∇f!, f, q, X0, λ, α_init, τ, r; niter=10000, backtracking=false)

    n, p = size(X0)
    X = Matrix(qr(X0).Q)
    I_p = Matrix{Float64}(I, p, p)

    fvals = zeros(niter)
    distance = zeros(niter)
    residual = zeros(niter)
    iterates = Array{eltype(X)}(undef, n, p, niter)

    ∇f_X = Matrix{eltype(X)}(undef, n, p)
    XX = Matrix{eltype(X)}(undef, p, p)       # X'X, then X'X - I_p
    Λ_λ = Matrix{eltype(X)}(undef, n, p)      # ψ_X, then Λ_λ
    ∇N_X = Matrix{eltype(X)}(undef, n, p)     # X*(∇f_X'X), then ∇N_X

    for i in 1:niter
        fvals[i] = f(X)
        distance[i] = norm(X' * X - I_p)
        residual[i] = q(X)
        iterates[:, :, i] = X

        println("[landing] Iter $i | dist Stiefel: ", distance[i])
        println("[landing] residual: ", residual[i])

        ∇f!(∇f_X, X)

        # ψ_X = ∇f_X * (X'X) - X * (∇f_X'X)
        mul!(XX, X', X)
        mul!(Λ_λ, ∇f_X, XX)
        mul!(XX, ∇f_X', X)
        mul!(∇N_X, X, XX)
        Λ_λ .-= ∇N_X

        # ∇N_X = X * (X'X - I_p)
        mul!(XX, X', X)
        @inbounds for j in 1:p
            XX[j, j] -= 1
        end
        mul!(∇N_X, X, XX)

        # Λ_λ = ψ_X + λ * ∇N_X
        @. Λ_λ += λ * ∇N_X

        α = α_init
        @. X -= α * Λ_λ
    end

    return X, residual, distance, fvals, iterates
end
