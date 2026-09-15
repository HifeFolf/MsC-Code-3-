# linear_stability.jl
# Linear stability analysis of the active antipolar ring.
#
# The Jacobian of the overdamped equations of motion evaluated at the perfect
# circle is decomposed as:
#
#   J(κ, f₀) = μ × [ H_bond  +  κ · H_bend  +  f₀ · H_act ]
#
#   H_bond  — Jacobian of bond-spring forces  (symmetric, κ-independent)
#   H_bend  — Jacobian of bending forces, κ=1 (symmetric)
#   H_act   — Jacobian of active forces,  f₀=1 (NON-symmetric — non-conservative)
#
# The three base matrices are computed once by finite differences; the sweep
# over (κ, f₀) is then rapid linear combination + eigendecomposition.
#
# Non-symmetric J → eigenvalues can be complex:
#   Real crossing zero      → stationary bifurcation (ring buckles to new shape)
#   Complex pair crossing   → Hopf bifurcation (ring starts to oscillate / travel)
#
# Outputs
#   1. Re(λ_max) vs f₀ for fixed κ — shows instability threshold
#   2. Phase boundary f₀_c(κ) — compare with sweep phase diagram
#   3. Critical mode shape — where does the ring deform?
#   4. Fl = f₀/κ scaling check — is the flexure number the right control parameter?
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/linear_stability.jl
#   julia --project scripts/linear_stability.jl configs/default.toml

using TOML, LinearAlgebra, Printf, Statistics

const PROJECT_ROOT_LS = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_LS, "src", "EngineRing.jl"));       using .EngineRing
include(joinpath(PROJECT_ROOT_LS, "src", "InitialConditions.jl")); using .InitialConditions

const HAVE_MAKIE = try
    using CairoMakie; true
catch; @warn "CairoMakie not found — text output only."; false
end

# ── Load config ───────────────────────────────────────────────────────────────
config_path = length(ARGS) >= 1 ? ARGS[1] :
              joinpath(PROJECT_ROOT_LS, "configs", "default.toml")
cfg = TOML.parsefile(config_path)
ep  = cfg["params"]

N     = Int(ep["N"])
l     = ep["l"]
ks    = ep["ks"]
σ     = ep["sigma"]
ε     = ep["epsilon"]
θmax  = ep["theta_max"]
θbuf  = ep["theta_buf"]
kstop = ep["k_stop"]
μ     = ep["mu"]
κ_cfg = ep["kappa"]   # default κ from config (used for fixed-κ sweep)

bead_types  = InitialConditions.antipolar_bead_types(N)
passive_idx = findfirst(==(:passive), bead_types)

# Reference geometry: perfect circle (equilibrium of the elastic energy)
p_ref   = EngineRing.Params(N=N, l=l, ks=ks, κ=1.0, theta_max=θmax, theta_buf=θbuf,
                             k_stop=kstop, sigma=σ, epsilon=ε, topology=:ring,
                             f0=0.0, mu=μ, kBT=0.0, dt=ep["dt"])
state0  = InitialConditions.perfect_circle_ic(p_ref, bead_types)
r0      = copy(state0.r)

println("=" ^ 65)
println("Linear stability  |  N=$N  ks=$ks  κ_cfg=$κ_cfg  μ=$μ")
println("Passive bead: $(passive_idx === nothing ? "none" : "bead $passive_idx")")
println("=" ^ 65)

# ── Finite-difference Jacobian ────────────────────────────────────────────────
function fd_jacobian(force_fn, r0::Matrix{Float64}; δ=1e-5)
    dof = 2N
    rf  = vec(copy(r0))
    J   = zeros(dof, dof)
    for j in 1:dof
        rp = copy(rf); rp[j] += δ
        rm = copy(rf); rm[j] -= δ
        J[:, j] = vec(force_fn(reshape(rp, 2, N)) .- force_fn(reshape(rm, 2, N))) ./ (2δ)
    end
    return J
end

# ── Compute the three base Jacobians once at r0 ───────────────────────────────
println("\nBuilding base Jacobians …")

p_bond = EngineRing.Params(N=N, l=l, ks=ks,  κ=0.0, theta_max=θmax, theta_buf=θbuf,
                            k_stop=0.0, sigma=σ, epsilon=ε,  topology=:ring,
                            f0=0.0, mu=μ, kBT=0.0, dt=ep["dt"])
p_bend = EngineRing.Params(N=N, l=l, ks=0.0, κ=1.0, theta_max=θmax, theta_buf=θbuf,
                            k_stop=0.0, sigma=σ, epsilon=0.0, topology=:ring,
                            f0=0.0, mu=μ, kBT=0.0, dt=ep["dt"])
p_act  = EngineRing.Params(N=N, l=l, ks=0.0, κ=0.0, theta_max=θmax, theta_buf=θbuf,
                            k_stop=0.0, sigma=σ, epsilon=0.0, topology=:ring,
                            f0=1.0, mu=μ, kBT=0.0, dt=ep["dt"])

H_bond = fd_jacobian(r -> EngineRing.bond_forces(r, p_bond), r0)
H_bend = fd_jacobian(r -> begin F=zeros(2,N); EngineRing.bending_forces!(F,r,p_bend); F end, r0)
H_act  = fd_jacobian(r -> begin F=zeros(2,N); EngineRing.active_forces!(F,r,p_act,bead_types); F end, r0)

asym_bond = maximum(abs.(H_bond .- H_bond'))
asym_bend = maximum(abs.(H_bend .- H_bend'))
asym_act  = maximum(abs.(H_act  .- H_act' ))
@printf("  H_bond asymmetry : %.2e  (should be ~0)\n", asym_bond)
@printf("  H_bend asymmetry : %.2e  (should be ~0)\n", asym_bend)
@printf("  H_act  asymmetry : %.2e  (non-zero = non-conservative forces)\n", asym_act)

# ── Rigid-body mode vectors (used to filter trivial modes) ────────────────────
v_tx  = zeros(2N); v_tx[1:2:end] .= 1.0 / sqrt(N)
v_ty  = zeros(2N); v_ty[2:2:end] .= 1.0 / sqrt(N)
v_rot = zeros(2N)
for i in 1:N
    v_rot[2i-1] = -r0[2, i]
    v_rot[2i]   =  r0[1, i]
end
v_rot ./= norm(v_rot)

# Overlap threshold: eigenvectors with projection > RB_TOL onto any rigid-body
# direction are skipped.  0.5 catches near-rigid modes while keeping shape modes.
const RB_TOL = 0.5

function is_rigid_body(v::AbstractVector{<:Real})
    v_n = v / max(norm(v), 1e-14)
    return max(abs(dot(v_n, v_tx)), abs(dot(v_n, v_ty)), abs(dot(v_n, v_rot))) > RB_TOL
end

# ── Helper: build full Jacobian ───────────────────────────────────────────────
full_J(κ, f0) = μ .* (H_bond .+ κ .* H_bend .+ f0 .* H_act)

# Most unstable SHAPE eigenvalue (rigid-body modes filtered out)
function most_unstable_shape(κ, f0)
    J = full_J(κ, f0)
    evals, evecs = eigen(J)
    best_re = -Inf;  best_im = 0.0
    for k in 1:2N
        is_rigid_body(real.(evecs[:, k])) && continue
        real(evals[k]) > best_re || continue
        best_re = real(evals[k])
        best_im = imag(evals[k])
    end
    return best_re, best_im
end

# ── Helper: find f₀_c by bisection (shape modes only) ────────────────────────
function find_f0c(κ; f0_lo=0.1, f0_hi=500.0, tol=1e-3)
    λ_lo, _ = most_unstable_shape(κ, f0_lo)
    λ_hi, _ = most_unstable_shape(κ, f0_hi)
    (λ_lo >= 0) && return f0_lo
    (λ_hi <  0) && return NaN
    for _ in 1:50
        f0_mid = exp((log(f0_lo) + log(f0_hi)) / 2)   # bisect in log space
        λ_mid, _ = most_unstable_shape(κ, f0_mid)
        λ_mid < 0 ? (f0_lo = f0_mid) : (f0_hi = f0_mid)
        (f0_hi / f0_lo - 1) < tol && break
    end
    return sqrt(f0_lo * f0_hi)
end

# ── 1. Eigenvalue crossing: Re(λ_max) vs f₀ at fixed κ ───────────────────────
println("\n── Eigenvalue crossing (κ = $κ_cfg) ─────────────────────────────────")
f0_sweep = exp.(range(log(0.1), log(300.0); length=300))
λ_real   = Vector{Float64}(undef, length(f0_sweep))
λ_imag   = Vector{Float64}(undef, length(f0_sweep))
for (k, f0) in enumerate(f0_sweep)
    λ_real[k], λ_imag[k] = most_unstable_shape(κ_cfg, f0)
end

f0_crit = find_f0c(κ_cfg)
if isnan(f0_crit)
    println("  No instability in f₀ ∈ [0.01, 500] for κ=$κ_cfg")
else
    _, im_crit = most_unstable_shape(κ_cfg, f0_crit)
    biftype = abs(im_crit) > 1.0 ? "Hopf (oscillatory)" : "stationary (buckling)"
    @printf("  f₀_c = %.4f\n", f0_crit)
    @printf("  Fl_c = f₀_c / κ = %.4f\n", f0_crit / κ_cfg)
    @printf("  Im(λ) at threshold = %.4f  →  %s\n", im_crit, biftype)
end

# ── 2. Phase boundary f₀_c(κ) ────────────────────────────────────────────────
println("\n── Phase boundary ────────────────────────────────────────────────────")
κ_grid = exp.(range(log(1.0), log(50.0); length=60))
f0c_grid = [find_f0c(κ) for κ in κ_grid]
Fl_grid  = f0c_grid ./ κ_grid

valid = .!isnan.(f0c_grid)
if any(valid)
    Fl_v   = Fl_grid[valid]
    κ_v    = κ_grid[valid]
    f0c_v  = f0c_grid[valid]

    Fl_mean = mean(Fl_v);  Fl_std = std(Fl_v)
    @printf("  Fl_c = f₀_c/κ :  mean = %.4f,  std = %.4f  (%.1f%% variation)\n",
            Fl_mean, Fl_std, 100*Fl_std/Fl_mean)

    # Log-log slope: f₀_c ∝ κ^α
    lκ = log.(κ_v);  lf = log.(f0c_v)
    α  = (lf[end] - lf[1]) / (lκ[end] - lκ[1])
    @printf("  Log-log slope α (f₀_c ∝ κ^α) = %.4f\n", α)
    if abs(α - 1.0) < 0.05
        println("  → f₀_c ∝ κ¹  :  Fl = f₀/κ is the correct control parameter")
    else
        @printf("  → Departure from Fl scaling: bond stiffness ks/κ enters at low κ\n")
    end

    println("\n  κ        f₀_c     Fl_c")
    println("  " * "-"^35)
    for i in eachindex(κ_v)
        @printf("  %6.2f   %7.3f   %.4f\n", κ_v[i], f0c_v[i], Fl_v[i])
    end

    # Save boundary to CSV so phase diagram scripts can load and overlay it
    stab_dir = joinpath(PROJECT_ROOT_LS, "results", "linear_stability")
    mkpath(stab_dir)
    csv_path = joinpath(stab_dir, "phase_boundary_N$(N).csv")
    open(csv_path, "w") do io
        println(io, "kappa,f0c,Fl_c")
        for i in eachindex(κ_v)
            @printf(io, "%.8f,%.8f,%.8f\n", κ_v[i], f0c_v[i], Fl_v[i])
        end
    end
    println("\n  Boundary saved → $csv_path")
end

# ── 3. Critical mode shape ────────────────────────────────────────────────────
println("\n── Critical mode shape ───────────────────────────────────────────────")
f0_for_mode = isnan(f0_crit) ? ep["f0"] : f0_crit
J_crit  = full_J(κ_cfg, f0_for_mode)
evals_c, evecs_c = eigen(J_crit)
# Pick most unstable shape mode (skip rigid-body eigenvectors)
idx_c = argmax([is_rigid_body(real.(evecs_c[:, k])) ? -Inf : real(evals_c[k])
                for k in 1:2N])
v_c   = real.(evecs_c[:, idx_c])
disp_c  = reshape(v_c, 2, N)
amp_c   = [sqrt(disp_c[1,i]^2 + disp_c[2,i]^2) for i in 1:N]
amp_c ./= maximum(amp_c)

# Identify dominant wavenumber: project onto cos/sin Fourier modes
proj = Dict{Int,Float64}()
for m in 2:N÷2
    cos_mode = [cos(2π*m*(i-1)/N) for i in 1:N]
    sin_mode = [sin(2π*m*(i-1)/N) for i in 1:N]
    cos_mode /= norm(cos_mode); sin_mode /= norm(sin_mode)
    proj[m] = max(abs(dot(amp_c, cos_mode)), abs(dot(amp_c, sin_mode)))
end
dominant_m = argmax(proj)
@printf("  Dominant wavenumber: m=%d  (projection = %.3f)\n", dominant_m, proj[dominant_m])
@printf("  Mode shape: %s\n",
        dominant_m==2 ? "elliptical" : dominant_m==3 ? "triangular" :
        dominant_m==4 ? "square"     : "m=$dominant_m")

println("\n  bead  type          amplitude")
println("  " * "-"^38)
for i in 1:N
    mark = bead_types[i] == :passive ? "  ← passive" :
           (!isnothing(passive_idx) && abs(i - passive_idx) == 1) ? "  ← adj" : ""
    @printf("  %3d   %-12s  %.4f%s\n", i, string(bead_types[i]), amp_c[i], mark)
end

# ── Plots ─────────────────────────────────────────────────────────────────────
if HAVE_MAKIE && any(valid)
    out_dir = joinpath(PROJECT_ROOT_LS, "results", "linear_stability")
    mkpath(out_dir)
    tag = "N$(N)"

    # Plot 1: Re(λ_max) vs f₀
    fig1 = Figure(size=(700, 380))
    ax1  = Axis(fig1[1,1]; xlabel="f₀", ylabel="Re(λ_max)",
                xscale=log10, title="Instability threshold  |  κ=$κ_cfg,  $tag")
    lines!(ax1, f0_sweep, λ_real; color=:steelblue, linewidth=2)
    hlines!(ax1, [0.0]; color=:black, linestyle=:dash, linewidth=1.5)
    isnan(f0_crit) || vlines!(ax1, [f0_crit]; color=:crimson, linestyle=:dot, linewidth=2,
                              label="f₀_c = $(round(f0_crit; sigdigits=4))")
    isnan(f0_crit) || axislegend(ax1)
    save(joinpath(out_dir, "threshold_$(tag).png"), fig1)

    # Plot 2: Phase boundary
    fig2 = Figure(size=(650, 500))
    ax2  = Axis(fig2[1,1]; xlabel="κ", ylabel="f₀_c",
                xscale=log10, yscale=log10,
                title="Phase boundary (linear stability)  |  $tag")
    lines!(ax2,  κ_v, f0c_v;  color=:steelblue, linewidth=2, label="f₀_c(κ) from stability")
    scatter!(ax2, κ_v, f0c_v; color=:steelblue, markersize=6)
    lines!(ax2, κ_v, Fl_mean .* κ_v; color=:crimson, linestyle=:dash, linewidth=1.5,
           label="Fl = $(round(Fl_mean; sigdigits=3)) (linear)")
    axislegend(ax2; position=:lt)
    save(joinpath(out_dir, "phase_boundary_$(tag).png"), fig2)

    # Plot 3: Fl_c vs κ (should be flat if Fl is the right parameter)
    fig3 = Figure(size=(650, 380))
    ax3  = Axis(fig3[1,1]; xlabel="κ", ylabel="Fl_c = f₀_c / κ",
                xscale=log10, title="Flexure number at threshold  |  $tag")
    lines!(ax3,  κ_v, Fl_v; color=:steelblue, linewidth=2)
    scatter!(ax3, κ_v, Fl_v; color=:steelblue, markersize=6)
    hlines!(ax3, [Fl_mean]; color=:crimson, linestyle=:dash, linewidth=1.5,
            label="mean Fl_c = $(round(Fl_mean; sigdigits=4))")
    axislegend(ax3)
    save(joinpath(out_dir, "Fl_threshold_$(tag).png"), fig3)

    # Plot 4: Critical mode shape
    fig4 = Figure(size=(900, 420))
    ax4a = Axis(fig4[1,1]; xlabel="Bead index", ylabel="Displacement amplitude",
                title="Critical mode  (m=$dominant_m, f₀_c=$(round(f0_for_mode; sigdigits=4)), κ=$κ_cfg)")
    colors_bar = [b==:passive ? :orange : b==:active ? :royalblue : :crimson for b in bead_types]
    barplot!(ax4a, 1:N, amp_c; color=colors_bar)
    isnothing(passive_idx) || vlines!(ax4a, [passive_idx]; color=:orange, linestyle=:dash,
                                      linewidth=2, label="passive (bead $passive_idx)")
    isnothing(passive_idx) || axislegend(ax4a)

    ax4b = Axis(fig4[1,2]; aspect=DataAspect(), title="Deformation pattern")
    xs = [r0[1,:]; r0[1,1]]; ys = [r0[2,:]; r0[2,1]]
    lines!(ax4b, xs, ys; color=:lightgray, linewidth=1.5)
    sc = 0.5 * l / maximum(amp_c)
    for i in 1:N
        col = bead_types[i]==:passive ? :orange : bead_types[i]==:active ? :royalblue : :crimson
        scatter!(ax4b, [r0[1,i]], [r0[2,i]]; color=col, markersize=10)
        arrows!(ax4b, [r0[1,i]], [r0[2,i]], [sc*disp_c[1,i]], [sc*disp_c[2,i]];
                color=:black, arrowsize=8, linewidth=1.2)
    end
    save(joinpath(out_dir, "critical_mode_$(tag).png"), fig4)

    println("\nPlots saved to results/linear_stability/")
end

println("\nDone.")
