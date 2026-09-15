# fl_origin.jl
# Exact derivation of the critical flexure number Fl_c = f0_c / kappa.
#
# The full Jacobian of the equations of motion is:
#
#   J(kappa, f0) = mu * [ H_bond  +  kappa * H_bend  +  f0 * H_act ]
#
# At a stationary instability (real eigenvalue crossing zero):
#
#   [ H_bond + kappa * H_bend + f0_c * H_act ] * v  =  0
#   =>  B(kappa) * v  =  -f0_c * H_act * v     where B = H_bond + kappa*H_bend
#   =>  M * v         =  -(1/f0_c) * v          where M = B^{-1} * H_act
#
# So f0_c = -1 / lambda_min(M),  with lambda_min the most-negative real
# eigenvalue of M restricted to the shape subspace.
# Fl_c = f0_c / kappa.
#
# If bonds are stiff (ks >> kappa), B ~ kappa*H_bend and Fl_c ~ const.
# The script shows this by computing Fl_c at several kappa values.
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/fl_origin.jl
#   julia --project scripts/fl_origin.jl configs/default.toml

using TOML, LinearAlgebra, Printf

const PROJECT_ROOT_FO = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_FO, "src", "EngineRing.jl"));        using .EngineRing
include(joinpath(PROJECT_ROOT_FO, "src", "InitialConditions.jl")); using .InitialConditions

config_path = length(ARGS) >= 1 ? ARGS[1] :
              joinpath(PROJECT_ROOT_FO, "configs", "default.toml")
cfg = TOML.parsefile(config_path)
ep  = cfg["params"]

N     = Int(ep["N"])
l     = ep["l"];  ks   = ep["ks"]
sigma = ep["sigma"]; epsilon = ep["epsilon"]
theta_max = ep["theta_max"]; theta_buf = ep["theta_buf"]
k_stop = ep["k_stop"]; mu = ep["mu"]
kappa_ref = ep["kappa"]

bead_types = InitialConditions.antipolar_bead_types(N)
p_ref = EngineRing.Params(N=N, l=l, ks=ks, κ=1.0,
            theta_max=theta_max, theta_buf=theta_buf, k_stop=k_stop,
            sigma=sigma, epsilon=epsilon, topology=:ring,
            f0=0.0, mu=mu, kBT=0.0, dt=ep["dt"])
state0 = InitialConditions.perfect_circle_ic(p_ref, bead_types)
r0     = copy(state0.r)

println("=" ^ 65)
println("Fl_c origin  |  N=$N  ks=$ks  mu=$mu")
println("=" ^ 65)

# ── Finite-difference Jacobians ───────────────────────────────────────────────
function fd_jac(force_fn, r0; delta=1e-5)
    dof = 2N
    rf  = vec(copy(r0))
    J   = zeros(dof, dof)
    for j in 1:dof
        rp = copy(rf); rp[j] += delta
        rm = copy(rf); rm[j] -= delta
        J[:, j] = vec(force_fn(reshape(rp,2,N)) .- force_fn(reshape(rm,2,N))) ./ (2*delta)
    end
    return J
end

p_bond = EngineRing.Params(N=N, l=l, ks=ks,  κ=0.0, theta_max=theta_max,
             theta_buf=theta_buf, k_stop=0.0, sigma=sigma, epsilon=0.0,
             topology=:ring, f0=0.0, mu=mu, kBT=0.0, dt=ep["dt"])
p_bend = EngineRing.Params(N=N, l=l, ks=0.0, κ=1.0, theta_max=theta_max,
             theta_buf=theta_buf, k_stop=0.0, sigma=sigma, epsilon=0.0,
             topology=:ring, f0=0.0, mu=mu, kBT=0.0, dt=ep["dt"])
p_wca  = EngineRing.Params(N=N, l=l, ks=0.0, κ=0.0, theta_max=theta_max,
             theta_buf=theta_buf, k_stop=0.0, sigma=sigma, epsilon=epsilon,
             topology=:ring, f0=0.0, mu=mu, kBT=0.0, dt=ep["dt"])
p_act  = EngineRing.Params(N=N, l=l, ks=0.0, κ=0.0, theta_max=theta_max,
             theta_buf=theta_buf, k_stop=0.0, sigma=sigma, epsilon=0.0,
             topology=:ring, f0=1.0, mu=mu, kBT=0.0, dt=ep["dt"])

println("\nBuilding H_bond …")
H_bond = fd_jac(r -> EngineRing.bond_forces(r, p_bond), r0)

println("Building H_bend …")
H_bend = fd_jac(r -> begin
    F = zeros(2,N); EngineRing.bending_forces!(F, r, p_bend); F end, r0)

println("Building H_wca  …")
H_wca  = fd_jac(r -> begin
    F = zeros(2,N); EngineRing.wca_forces!(F, r, p_wca); F end, r0)

println("Building H_act  …")
H_act  = fd_jac(r -> begin
    F = zeros(2,N); EngineRing.active_forces!(F, r, p_act, bead_types); F end, r0)

# ── Helper: compute f0_c and Fl_c at a given kappa ───────────────────────────
# Uses the passive Jacobian B = H_bond + kappa*H_bend to define the shape subspace.
# All shape modes of B have negative eigenvalues (bonds + bending stabilise everything).
# Rigid-body modes (3) have eigenvalue ~ 0.
# Always remove exactly 3 rigid-body modes (the 3 with smallest |lambda|).
# A relative threshold would also cut soft bending modes at low kappa.
function shape_basis(B_mat)
    eig = eigen(Symmetric(B_mat))
    lam = eig.values
    idx = sortperm(abs.(lam))[4:end]   # skip the 3 smallest-|lambda| (rigid body)
    return eig.vectors[:, sort(idx)], lam[sort(idx)]
end

function compute_f0c(kappa_val)
    B    = H_bond .+ kappa_val .* H_bend .+ H_wca
    VS, b = shape_basis(B)
    A    = VS' * H_act * VS
    M    = Diagonal(1.0 ./ b) * A
    lam  = eigvals(M)
    tol  = 1e-4 * maximum(abs.(lam))
    real_lam = real.(lam[abs.(imag.(lam)) .< tol])
    neg  = real_lam[real_lam .< 0]
    isempty(neg) && return NaN
    return -1.0 / minimum(neg)
end

# ── Main result at kappa_ref ──────────────────────────────────────────────────
println("\nBuilding passive Jacobian B = H_bond + $(kappa_ref)*H_bend + H_wca …")
B_ref = H_bond .+ kappa_ref .* H_bend .+ H_wca
VS_B, b_ref = shape_basis(B_ref)
A_ref = VS_B' * H_act * VS_B
M_ref  = Diagonal(1.0 ./ b_ref) * A_ref
lam_M  = eigvals(M_ref)

# A stationary bifurcation requires a REAL eigenvalue of M crossing -1/f0_c.
# Complex eigenvalues of M correspond to Hopf-like crossings and must not
# be used here.  Filter: keep only eigenvalues with negligible imaginary part.
im_tol     = 1e-4 * maximum(abs.(lam_M))
real_mask  = abs.(imag.(lam_M)) .< im_tol
real_lam_M = real.(lam_M[real_mask])

n_shape = length(b_ref)
@printf("\nShape modes: %d   Rigid-body removed: %d\n", n_shape, 2N - n_shape)
@printf("Eigenvalues of M: %d real,  %d complex\n",
        sum(real_mask), sum(.!real_mask))

println("\n── Generalised eigenvalue result (stationary bifurcation) ────────────")
if any(real_lam_M .< 0)
    lmin = minimum(real_lam_M[real_lam_M .< 0])
    f0c  = -1.0 / lmin
    Flc  = f0c / kappa_ref
    @printf("  Most negative real lambda_M = %+.6f\n", lmin)
    @printf("  f0_c  = -1 / lambda_min     = %.4f   (at kappa = %.1f)\n", f0c, kappa_ref)
    @printf("  Fl_c  = f0_c / kappa        = %.4f\n", Flc)
else
    println("  No negative real eigenvalue of M found → no stationary bifurcation.")
    println("  The instability (if any) is a Hopf bifurcation (complex eigenvalue).")
    f0c = NaN; Flc = NaN
end

# ── Fl_c vs kappa: verify approximate constancy ──────────────────────────────
println("\n── Fl_c vs kappa  (constant => Fl is the right control parameter) ────")
println("  kappa    f0_c      Fl_c")
println("  " * "-"^32)
for kap in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    f0_k = compute_f0c(kap)
    @printf("  %5.1f   %7.2f   %.4f\n", kap, f0_k, f0_k/kap)
end

# ── Per-mode breakdown at kappa_ref ──────────────────────────────────────────
# In the B eigenbasis: b_k < 0 is mode k's passive decay rate.
# a_kk = (VS[:,k])^T * H_act * (VS[:,k]) = active self-coupling of mode k.
# Single-mode threshold: f0_c^(k) = -b_k / a_kk  (positive when a_kk > 0).
a_diag    = real.(diag(A_ref))
f0_single = -b_ref ./ a_diag

destab = findall(k -> a_diag[k] > 0 && f0_single[k] > 0, 1:n_shape)
println("\n── Per-mode breakdown (modes destabilised by activity, κ=$kappa_ref) ─")
println("  rank   b_k (passive rate)   a_kk (active drive)   f0_c^(k)   Fl_c^(k)")
println("  " * "-"^70)
if isempty(destab)
    println("  (no mode has positive active self-coupling in this basis)")
else
    order = destab[sortperm(f0_single[destab])]
    for (rank, k) in enumerate(order)
        @printf("  %3d   %14.5f      %14.5f    %8.2f   %8.2f\n",
                rank, b_ref[k], a_diag[k], f0_single[k], f0_single[k]/kappa_ref)
    end
end

println("\n── Summary ───────────────────────────────────────────────────────────")
println("  f0_c comes from the balance: passive decay  =  active driving")
println("  b_k * (amplitude)  +  f0_c * a_kk * (amplitude)  =  0")
println("  => f0_c = -b_k / a_kk  for the critical mode")
println()
@printf("  At κ=%.1f: exact f0_c=%.2f, Fl_c=%.4f\n", kappa_ref, f0c, Flc)
println("  The near-constancy of Fl_c across kappa values confirms that")
println("  ks=$(ks) >> kappa and WCA stiffness ~ kappa, so f0_c ∝ kappa.")
println("  B = H_bond + kappa*H_bend + H_wca (all passive restoring forces).")

# ── Direct verification: check Re(lambda_max_shape) at predicted f0_c ────────
# Build rigid-body unit vectors (eigenvector-based filter, same as linear_stability.jl)
v_tx  = zeros(2N); v_tx[1:2:end]  .= 1/sqrt(N)
v_ty  = zeros(2N); v_ty[2:2:end]  .= 1/sqrt(N)
v_rot = zeros(2N)
for i in 1:N; v_rot[2i-1] = -r0[2,i]; v_rot[2i] = r0[1,i]; end
v_rot ./= norm(v_rot)
const RB_TOL_FO = 0.5
function is_rigid(v)
    vn = v / max(norm(v), 1e-14)
    return max(abs(dot(vn,v_tx)), abs(dot(vn,v_ty)), abs(dot(vn,v_rot))) > RB_TOL_FO
end

function max_shape_re(kap, f0_val)
    J    = H_bond .+ kap .* H_bend .+ H_wca .+ f0_val .* H_act
    ev, evec = eigen(J)
    best = -Inf
    for k in 1:2N
        is_rigid(real.(evec[:,k])) && continue
        re = real(ev[k])
        re > best && (best = re)
    end
    return best
end

println("\n── Verification: Re(lambda_max_shape) at key f0 values (kappa=$kappa_ref) ─")
println("  f0        Re(lambda_max_shape)   stable?")
println("  " * "-"^45)
for f0_test in [f0c * 0.5, f0c, f0c * 2.0, f0c * 3.5]
    re = max_shape_re(kappa_ref, f0_test)
    @printf("  %8.2f    %+12.6f         %s\n",
            f0_test, re, re < 0 ? "yes" : "NO — unstable")
end

println("\nDone.")
