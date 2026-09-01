using LinearAlgebra
using Statistics
using Printf

BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------
# Paramètres du problème
# ---------------------------------------------------------------------------
pca_boolean = false
procruste_boolean = true

n = 1000
p = 10
niter = 1

# Paramètres de la borne théorique
K_factor = sqrt(2)         # Constante issue de la démonstration rigoureuse
ε_margin = 10.0            # ε_threshold = ε_margin × δ_X0 (marge au-dessus du bruit numérique de X0)

# ---------------------------------------------------------------------------
# Initialisation du problème (PCA ou Procrustes)
# ---------------------------------------------------------------------------
if procruste_boolean
    prob_name = "Procrustes"
    B1_proc = randn(n, n)
    A_proc = B1_proc' * B1_proc
    A_proc ./= opnorm(A_proc)   # normalisation : ||A_proc|| = 1, gradient à échelle raisonnable indép. de n
    X_opt_proc = Matrix(qr(randn(n, p)).Q)
    B2_proc = A_proc * X_opt_proc

    f_proc(X) = norm(A_proc * X - B2_proc)^2
    AtA_proc = A_proc' * A_proc
    AtB2_proc = A_proc' * B2_proc
    @views function ∇f_proc!(out, X)
        mul!(out, AtA_proc, X)
        out .*= 2
        out .-= 2 .* AtB2_proc
        return out
    end
    
    X0 = Matrix(qr(randn(n, p)).Q)   # retraction initiale sur la variété de Stiefel
    ∇f! = ∇f_proc!

elseif pca_boolean
    prob_name = "PCA"
    B1 = randn(n, n)
    A = B1' * B1
    A ./= opnorm(A)   # normalisation : ||A|| = 1, gradient à échelle raisonnable indép. de n
    U = Diagonal(Array(p:-1:1))
    _u_diag = U.diag

    f(X) = -tr(X' * A * X * U)
    @views function ∇f_pca!(out, X)
        mul!(out, A, X)
        out .*= -2 .* _u_diag'
        return out
    end

    X0 = Matrix(qr(randn(n, p)).Q)   # retraction initiale sur la variété de Stiefel
    ∇f! = ∇f_pca!
else
    error("Veuillez activer au moins un problème (pca_boolean ou procruste_boolean).")
end

# ---------------------------------------------------------------------------
# Calcul de la borne théorique alpha_théorique à X0
# ---------------------------------------------------------------------------
∇f_0 = zeros(n, p)
∇f!(∇f_0, X0)


ψ_0 = ∇f_0 * X0' - X0 * ∇f_0'
norm_ψ0 = norm(ψ_0)

δ_X0 = norm(X0' * X0 - I(p))
ε_threshold =  δ_X0

# Borne garantie : \alpha <= sqrt( (sqrt(2)-1) * \epsilon / ((1+\epsilon) * ||\psi(X0)||_F^2) )
α_theory_max = sqrt(((K_factor - 1.0) * ε_threshold) / ((1.0 + ε_threshold) * norm_ψ0^2))

# ---------------------------------------------------------------------------
# En-tête du Rapport
# ---------------------------------------------------------------------------
println("\n" * "═"^80)
println("   ANALYSE DE STABILITÉ EXPÉRIMENTALE VS BORNE THÉORIQUE (ALGORITHME POGO)")
println("═"^80)
@printf(" Problème étudié                : %s (n = %d, p = %d)\n", prob_name, n, p)
@printf(" Déviation initiale ||X0'X0-I||_F: %.6e\n", δ_X0)
@printf(" Norme ||ψ(X0)||_F              : %.6e\n", norm_ψ0)
@printf(" Seuil d'orthogonalité ε cible  : %.6e (= %.1f × δ_X0)\n", ε_threshold, ε_margin)
println("─"^80)
@printf(" ==> PAS MAXIMAL THÉORIQUE GARANTI : α_théorique <= %.6e\n", α_theory_max)
println("═"^80)

# ---------------------------------------------------------------------------
# Grid Search sur le pas \alpha (de 10^-2 * α_théorique à 10^2 * α_théorique)
# ---------------------------------------------------------------------------
num_points = 25
alphas_test = 10 .^ range(log10(α_theory_max * 1e-2), log10(α_theory_max * 1e5), length=num_points)

println("\n" * "┌" * "─"^78 * "┐")
@printf("│ %-14s │ %-20s │ %-15s │ %-18s │\n", "Pas α", "Max ||X'X - I_p||_F", "Ratio α/α_théo", "Statut Stabilité")
println("├" * "─"^78 * "┤")

for α in alphas_test
    X_k = copy(X0)
    max_dev = norm(X_k' * X_k - I(p))
    diverged = false
    
    for k in 1:niter
        ∇f_k = zeros(n, p)
        ∇f!(∇f_k, X_k)
        ψ_k = ∇f_k * X_k' - X_k * ∇f_k'
        
        
        M_k = X_k - α * (ψ_k * X_k)
        X_k = 1.5 * M_k - 0.5 * (M_k * (M_k' * M_k))
        
        dev = norm(X_k' * X_k - I(p))
        
        if isnan(dev) || dev > 1e4
            diverged = true
            max_dev = Inf
            break
        end
        max_dev = max(max_dev, dev)
    end
    
    ratio = α / α_theory_max
    
    # Formatage du statut
    statut_str = if diverged
        "DIVERGENCE (Inf)"
    elseif max_dev <= ε_threshold
        "STABLE (<= ε)"
    else
        "INVALIDE (> ε)"
    end
    
    # Indicateur visuel pour la borne théorique
    is_near_theory = abs(log10(ratio)) < 0.1
    prefix = is_near_theory ? "👉" : "  "

    if diverged
        @printf("%s│ %-14.6e │ %-20s │ %-15.3f │ %-18s │\n", prefix, α, "       Inf          ", ratio, statut_str)
    else
        @printf("%s│ %-14.6e │ %-20.6e │ %-15.3f │ %-18s │\n", prefix, α, max_dev, ratio, statut_str)
    end
end

println("└" * "─"^78 * "┘")
println("Note : 👉 indique le pas le plus proche de la borne théorique α_théorique.\n")