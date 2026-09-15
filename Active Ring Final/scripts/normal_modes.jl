# normal_modes.jl
# Linear normal-mode analysis of the passive ring (f0 = 0, kBT = 0).
#
# Method: numerical Hessian via central finite differences.
# The Hessian H (2N × 2N) is the matrix of second derivatives of the total
# elastic energy (bond + bending + WCA).  Its eigenvalues are mode stiffnesses;
# its eigenvectors are the spatial deformation patterns.
#
# Three global rigid-body modes (2 translations + 1 rotation) have zero
# stiffness and are filtered out.  The remaining 2N-3 modes are sorted from
# softest to stiffest and plotted.
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/normal_modes.jl                        # default.toml, circle IC
#   julia --project scripts/normal_modes.jl configs/default.toml antipolar_kinked
#
# Args:
#   arg 1 — config file   (default: configs/default.toml)
#   arg 2 — bead_layout   (default: "antipolar", alternative: "antipolar_kinked")

using TOML, LinearAlgebra, Printf

const PROJECT_ROOT_NM = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_NM, "src", "EngineRing.jl"));       using .EngineRing
include(joinpath(PROJECT_ROOT_NM, "src", "InitialConditions.jl")); using .InitialConditions

# ── Attempt to load CairoMakie; fall back gracefully ─────────────────────────
const HAVE_MAKIE = try
    import Pkg; Pkg.status("CairoMakie"; io = devnull)
    using CairoMakie
    true
catch
    @warn "CairoMakie not found — skipping plots, printing results only."
    false
end

# ── Parse args ────────────────────────────────────────────────────────────────
config_path = length(ARGS) >= 1 ? ARGS[1] :
              joinpath(PROJECT_ROOT_NM, "configs", "default.toml")
layout      = length(ARGS) >= 2 ? ARGS[2] : "antipolar"

cfg = TOML.parsefile(config_path)
ep  = cfg["params"]

# Force f0 = 0 and kBT = 0 regardless of config — this is a passive analysis
p = EngineRing.Params(
    N         = Int(ep["N"]),
    l         = ep["l"],
    ks        = ep["ks"],
    κ         = ep["kappa"],
    theta_max = ep["theta_max"],
    theta_buf = ep["theta_buf"],
    k_stop    = ep["k_stop"],
    sigma     = ep["sigma"],
    epsilon   = ep["epsilon"],
    topology  = Symbol(ep["topology"]),
    f0        = 0.0,    # passive: no propulsion
    mu        = ep["mu"],
    kBT       = 0.0,    # no noise
    dt        = ep["dt"],
)

bead_types = InitialConditions.antipolar_bead_types(p.N)
state      = InitialConditions.create_ic(layout, p, bead_types)

println("=" ^ 65)
println("Normal-mode analysis  |  N=$(p.N)  κ=$(p.κ)  ks=$(p.ks)  layout=$layout")
println("=" ^ 65)

# ── Total passive force (no active forces) ────────────────────────────────────
function passive_force(r_flat::Vector{Float64}, p::Params)
    r = reshape(copy(r_flat), 2, p.N)
    F = EngineRing.bond_forces(r, p)
    EngineRing.bending_forces!(F, r, p)
    EngineRing.wca_forces!(F, r, p)
    return vec(F)
end

# ── Numerical Hessian via central finite differences ─────────────────────────
# H[k, j] = -∂F_k/∂r_j  ≈  -(F_k(r+δe_j) - F_k(r-δe_j)) / (2δ)
function numerical_hessian(r0::Matrix{Float64}, p::Params; δ::Float64 = 1e-5)
    dof = 2 * p.N
    r_flat = vec(copy(r0))
    H = zeros(dof, dof)
    for j in 1:dof
        rp = copy(r_flat); rp[j] += δ
        rm = copy(r_flat); rm[j] -= δ
        Fp = passive_force(rp, p)
        Fm = passive_force(rm, p)
        H[:, j] = -(Fp .- Fm) ./ (2δ)
    end
    return Symmetric(H)   # symmetrise to suppress numerical noise
end

println("\nBuilding Hessian ($(2p.N)×$(2p.N)) …")
H = numerical_hessian(state.r, p)

# ── Eigendecomposition ────────────────────────────────────────────────────────
println("Diagonalising …")
eig = eigen(H)
λ = eig.values    # sorted ascending by LinearAlgebra.eigen for Symmetric
V = eig.vectors   # columns are eigenvectors

# ── Identify rigid-body modes (|λ| < threshold) ──────────────────────────────
rigid_tol = 1e-2 * maximum(abs.(λ))
rigid_mask = abs.(λ) .< rigid_tol
n_rigid    = sum(rigid_mask)
shape_mask = .!rigid_mask

println("\nEigenvalue summary:")
@printf("  Total DOF          : %d\n", 2p.N)
@printf("  Rigid-body modes   : %d  (|λ| < %.2e)\n", n_rigid, rigid_tol)
@printf("  Shape modes        : %d\n", sum(shape_mask))
@printf("  Softest shape mode : λ = %.6f\n", minimum(λ[shape_mask]))
@printf("  Stiffest mode      : λ = %.6f\n", maximum(λ))

# ── Print sorted shape-mode stiffnesses ──────────────────────────────────────
shape_idx = findall(shape_mask)   # indices into λ/V, already sorted
println("\nShape-mode stiffnesses (softest first):")
println("  rank   λ (stiffness)   dominant bead contribution")
println("  " * "-"^55)
for (rank, idx) in enumerate(shape_idx[1:min(10, end)])
    v = V[:, idx]
    # per-bead displacement amplitude: sqrt(vx² + vy²)
    amp = [sqrt(v[2i-1]^2 + v[2i]^2) for i in 1:p.N]
    peak_bead = argmax(amp)
    @printf("  %3d    %12.5f    bead %2d (type: %s, amp=%.4f)\n",
            rank, λ[idx], peak_bead, string(bead_types[peak_bead]), amp[peak_bead])
end

# ── Per-bead participation in the 3 softest shape modes ──────────────────────
n_soft = min(3, sum(shape_mask))
soft_modes = shape_idx[1:n_soft]

println("\nPer-bead displacement amplitude in the $(n_soft) softest shape modes:")
@printf("%-5s  %-12s", "bead", "type")
for k in 1:n_soft; @printf("  mode%-4d", k); end
println()
println("  " * "-"^(17 + 10*n_soft))
for i in 1:p.N
    @printf("%-5d  %-12s", i, string(bead_types[i]))
    for k in 1:n_soft
        v   = V[:, soft_modes[k]]
        amp = sqrt(v[2i-1]^2 + v[2i]^2)
        @printf("  %8.5f", amp)
    end
    marker = bead_types[i] == :passive ? "  ← passive" : ""
    println(marker)
end

# ── Plots ─────────────────────────────────────────────────────────────────────
if HAVE_MAKIE
    out_dir = joinpath(PROJECT_ROOT_NM, "results", "normal_modes")
    mkpath(out_dir)
    tag = "N$(p.N)_$(layout)"

    # 1. Eigenvalue spectrum
    fig1 = Figure(size = (700, 350))
    ax1  = Axis(fig1[1, 1],
                xlabel = "Mode index (sorted by stiffness)",
                ylabel = "Stiffness λ",
                title  = "Mode stiffness spectrum  |  $tag")
    scatter!(ax1, 1:sum(shape_mask), λ[shape_mask]; markersize = 6, color = :steelblue)
    hlines!(ax1, [0.0]; color = :black, linestyle = :dash, linewidth = 1)
    save(joinpath(out_dir, "spectrum_$(tag).png"), fig1)
    println("\nSaved: spectrum_$(tag).png")

    # 2. Per-bead amplitude for the 3 softest modes
    bead_idx = 1:p.N
    colors   = [:crimson, :royalblue, :seagreen]
    labels   = ["mode $k" for k in 1:n_soft]

    fig2 = Figure(size = (750, 380))
    ax2  = Axis(fig2[1, 1],
                xlabel = "Bead index",
                ylabel = "Displacement amplitude (normalised)",
                title  = "Softest shape modes  |  $tag")
    for k in 1:n_soft
        v   = V[:, soft_modes[k]]
        amp = [sqrt(v[2i-1]^2 + v[2i]^2) for i in 1:p.N]
        amp ./= maximum(amp)   # normalise to 1
        lines!(ax2, bead_idx, amp; color = colors[k], label = labels[k], linewidth = 2)
        scatter!(ax2, bead_idx, amp; color = colors[k], markersize = 7)
    end
    # Mark passive bead
    if !isnothing(findfirst(==(:passive), bead_types))
        pi = findfirst(==(:passive), bead_types)
        vlines!(ax2, [pi]; color = :orange, linestyle = :dash, linewidth = 1.5,
                label = "passive (bead $pi)")
    end
    axislegend(ax2; position = :rt)
    save(joinpath(out_dir, "soft_modes_$(tag).png"), fig2)
    println("Saved: soft_modes_$(tag).png")

    # 3. Deformation patterns overlaid on ring geometry
    fig3 = Figure(size = (900, 300 * n_soft))
    R0   = state.r
    for k in 1:n_soft
        v    = V[:, soft_modes[k]]
        disp = reshape(v, 2, p.N)
        scale = 0.5 * p.l / maximum(sqrt.(disp[1,:].^2 .+ disp[2,:].^2))
        ax = Axis(fig3[k, 1]; aspect = DataAspect(),
                  title = "Mode $k  (λ=$(round(λ[soft_modes[k]]; sigdigits=4)))  |  $tag")
        # Ring at equilibrium
        xs = [R0[1, :]; R0[1, 1]]
        ys = [R0[2, :]; R0[2, 1]]
        lines!(ax, xs, ys; color = :lightgray, linewidth = 1.5)
        # Displacement arrows
        for i in 1:p.N
            col = bead_types[i] == :passive ? :orange :
                  bead_types[i] == :active  ? :royalblue : :crimson
            scatter!(ax, [R0[1,i]], [R0[2,i]]; color = col, markersize = 10)
            arrows!(ax, [R0[1,i]], [R0[2,i]],
                        [scale*disp[1,i]], [scale*disp[2,i]];
                        color = :black, arrowsize = 8, linewidth = 1.2)
        end
    end
    save(joinpath(out_dir, "mode_shapes_$(tag).png"), fig3)
    println("Saved: mode_shapes_$(tag).png")
end

println("\nDone.")
