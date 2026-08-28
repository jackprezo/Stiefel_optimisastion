# Jack Prezerowitz
# Riemannian Gradient Descent on the Stiefel manifold (retraction via Newton-Schulz orthonormalization)

using LinearAlgebra

grad_C(∇f, X) = ∇f - X * ∇f' * X

@views function grad_C!(gradf, ∇fX_X, X)
    mul!(∇fX_X, gradf', X)          # ∇fX_X = ∇f_X' * X   (p×p)
    mul!(gradf, X, ∇fX_X, -1, 1)    # gradf = ∇f_X - X * ∇fX_X
    return gradf
end

@views function retract_polar!(X, X_new, E, ε)
    e = Inf
    while e > ε
        mul!(E, X_new', X_new)
        @inbounds for j in 1:size(E, 1)
            E[j, j] -= 1
        end
        e = norm(E)
        @inbounds for j in 1:size(E, 1)
            E[j, j] += 1
        end
        X .= X_new
        mul!(X_new, X, E, -0.5, 1.5)
    end
    X .= X_new
    return X
end

@views function retract_qr!(X, X_new)
    X .= Matrix(qr(X_new).Q)
    return X
end

@views function rgd(X0::AbstractMatrix{T}, ∇f!, f, q, grad, α_init, τ, r; niter=1000, backtracking=false, retract=:polar) where T

    n, p = size(X0)
    X = Matrix(qr(X0).Q)
    I_p = I(p)

    fvals = zeros(niter)
    distance = zeros(niter)
    residual = zeros(niter)
    iterates = Array{eltype(X)}(undef, n, p, niter)

    gradf = Matrix{eltype(X)}(undef, n, p)
    ∇fX_X = Matrix{eltype(X)}(undef, p, p)
    X_new = Matrix{eltype(X)}(undef, n, p)

    ε = sqrt(eps(T) * 10 * sqrt(p))

    for i in 1:niter

        fvals[i] = f(X)
        distance[i] = norm(X' * X - I_p)
        residual[i] = q(X)
        iterates[:, :, i] = X

        println("[rgd] Iter $i | dist Stiefel: ", distance[i])
        println("[rgd] residual: ", residual[i])

        ∇f!(gradf, X)
        grad_C!(gradf, ∇fX_X, X)

        α = α_init

        @. X_new = X - α * gradf

        if retract === :polar
            retract_polar!(X, X_new, ∇fX_X, ε)
        elseif retract === :qr
            retract_qr!(X, X_new)
        else
            error("retract inconnu: $retract (attendu :polar ou :qr)")
        end
    end
    return X, residual, distance, fvals, iterates
end
