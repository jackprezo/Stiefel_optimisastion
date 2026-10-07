"""Frobenius norm of the Stiefel constraint violation, `X'X - I`."""
stiefel_distance(X) = norm(X' * X - I)

"""Replace an ambient gradient by the canonical gradient `G - X*G'X`."""
function canonical_gradient!(gradient, gram, X)
    mul!(gram, gradient', X)
    mul!(gradient, X, gram, -1, 1)
    return gradient
end

"""Apply Newton–Schulz orthonormalization, retaining the original stopping convention."""
function retract_polar!(X, candidate, gram, tolerance; maxiter=100)
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    for _ in 1:maxiter
        mul!(gram, candidate', candidate)
        for j in axes(gram, 1)
            gram[j, j] -= 1
        end
        error = norm(gram)
        isfinite(error) || throw(DomainError(error, "polar retraction diverged"))
        for j in axes(gram, 1)
            gram[j, j] += 1
        end
        copyto!(X, candidate)
        mul!(candidate, X, gram, -0.5, 1.5)
        if error <= tolerance
            copyto!(X, candidate)
            return X
        end
    end
    throw(ErrorException("polar retraction did not converge within $maxiter updates"))
end

"""Copy the thin QR orthonormalization of `candidate` into `X`."""
function retract_qr!(X, candidate)
    X .= Matrix(qr(candidate).Q)
    return X
end

"""Compute the POGO direction using preallocated workspaces."""
function pogo_direction!(direction, gradient, X, gram, product; direction_scale=1.0)
    mul!(gram, X', X)
    mul!(direction, gradient, gram)
    mul!(gram, gradient', X)
    mul!(product, X, gram)
    @. direction = direction_scale * (direction - product)
    return direction
end

"""Evaluate the POGO direction at `X` using an in-place ambient gradient."""
function pogo_direction(gradient!, X; direction_scale=1.0)
    gradient = similar(X)
    gradient!(gradient, X)
    direction = similar(X)
    product = similar(X)
    gram = similar(X, size(X, 2), size(X, 2))
    return pogo_direction!(direction, gradient, X, gram, product; direction_scale)
end

"""Return POGO's intermediate point before the constraint correction."""
function pogo_intermediate(X, gradient!, alpha; direction_scale=1.0)
    direction = pogo_direction(gradient!, X; direction_scale)
    return X - alpha * direction
end

function pogo_correction!(X, intermediate, lambda, gram, product)
    mul!(gram, intermediate', intermediate)
    mul!(product, intermediate, gram)
    @. X = (1 + lambda) * intermediate - lambda * product
    return X
end

"""Take one POGO step without initial orthonormalization."""
function pogo_step(X, direction, alpha; lambda=0.5)
    intermediate = X - alpha * direction
    result = similar(intermediate)
    gram = similar(intermediate, size(X, 2), size(X, 2))
    product = similar(intermediate)
    return pogo_correction!(result, intermediate, lambda, gram, product)
end

function initialize_solver(X0::AbstractMatrix{<:Real}, niter, record_history, store_iterates)
    n, p = size(X0)
    1 <= p <= n || throw(ArgumentError("X0 must have dimensions n × p with 1 ≤ p ≤ n"))
    niter isa Integer && niter >= 0 || throw(ArgumentError("niter must be a nonnegative integer"))
    X = Matrix(qr(X0).Q)
    history_length = record_history ? niter : 0
    iterate_length = record_history && store_iterates ? niter : 0
    result = (
        X=X,
        residual=zeros(history_length),
        distance=zeros(history_length),
        fvals=zeros(history_length),
        iterates=Array{eltype(X)}(undef, n, p, iterate_length),
    )
    return result
end

function record_iteration!(result, f, q, iteration)
    result.fvals[iteration] = f(result.X)
    result.distance[iteration] = stiefel_distance(result.X)
    result.residual[iteration] = q(result.X)
    if size(result.iterates, 3) > 0
        @views result.iterates[:, :, iteration] .= result.X
    end
    return nothing
end

"""Run RGD; histories record pre-update states, while X is the final state."""
function rgd(X0, gradient!, f, q, alpha; niter=1000, retract=:polar,
             record_history=true, store_iterates=true)
    retract in (:polar, :qr) || throw(ArgumentError("retract must be :polar or :qr"))
    result = initialize_solver(X0, niter, record_history, store_iterates)
    X = result.X
    p = size(X, 2)
    gradient = similar(X)
    gram = similar(X, p, p)
    candidate = similar(X)
    tolerance = sqrt(eps(eltype(X)) * 10 * sqrt(p))

    for iteration in 1:niter
        record_history && record_iteration!(result, f, q, iteration)
        gradient!(gradient, X)
        canonical_gradient!(gradient, gram, X)
        @. candidate = X - alpha * gradient
        if retract === :polar
            retract_polar!(X, candidate, gram, tolerance)
        else
            retract_qr!(X, candidate)
        end
    end
    return result
end

"""Run Landing with the same history options as RGD."""
function landing_flow(X0, gradient!, f, q, lambda, alpha; niter=10000,
                      record_history=true, store_iterates=true)
    result = initialize_solver(X0, niter, record_history, store_iterates)
    X = result.X
    p = size(X, 2)
    gradient = similar(X)
    gram = similar(X, p, p)
    direction = similar(X)
    product = similar(X)

    for iteration in 1:niter
        record_history && record_iteration!(result, f, q, iteration)
        gradient!(gradient, X)
        pogo_direction!(direction, gradient, X, gram, product)
        mul!(gram, X', X)
        for j in 1:p
            gram[j, j] -= 1
        end
        mul!(product, X, gram)
        @. direction += lambda * product
        @. X -= alpha * direction
    end
    return result
end

"""Run POGO; direction_scale=0.5 retains the historical timing variant."""
function pogo(X0, gradient!, f, q, alpha, lambda; niter=1000, direction_scale=1.0,
              record_history=true, store_iterates=true)
    result = initialize_solver(X0, niter, record_history, store_iterates)
    X = result.X
    p = size(X, 2)
    gradient = similar(X)
    gram = similar(X, p, p)
    direction = similar(X)
    product = similar(X)
    intermediate = similar(X)

    for iteration in 1:niter
        record_history && record_iteration!(result, f, q, iteration)
        gradient!(gradient, X)
        pogo_direction!(direction, gradient, X, gram, product; direction_scale)
        @. intermediate = X - alpha * direction
        pogo_correction!(X, intermediate, lambda, gram, product)
    end
    return result
end
