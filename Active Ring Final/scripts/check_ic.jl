# check_ic.jl
# Diagnostic: compare the circle and kinked ICs at t=0.
#
# For each IC this script reports, bead by bead:
#   - bond length to the next bead  (should equal l=1.0 for circle; deviates near passive for kinked)
#   - bending angle at each bead    (turning angle θ in degrees)
#   - per-force-component magnitudes: bond, bending, WCA, active
#   - total initial force magnitude
#
# Crucially it also checks:
#   - no unintended WCA overlaps (r < r_wca cutoff) except those expected from the kink
#   - active force magnitudes are exactly f0 for every :active / :active_rev bead
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/check_ic.jl
#   julia --project scripts/check_ic.jl configs/default.toml

using TOML, Printf, LinearAlgebra

const PROJECT_ROOT_CHK = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_CHK, "src", "EngineRing.jl"));       using .EngineRing
include(joinpath(PROJECT_ROOT_CHK, "src", "InitialConditions.jl")); using .InitialConditions

# ── Load params ──────────────────────────────────────────────────────────────
config_path = length(ARGS) >= 1 ? ARGS[1] :
              joinpath(PROJECT_ROOT_CHK, "configs", "default.toml")
cfg = TOML.parsefile(config_path)
ep  = cfg["params"]

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
    f0        = ep["f0"],
    mu        = ep["mu"],
    kBT       = 0.0,   # force check: zero noise
    dt        = ep["dt"],
)

bead_types = InitialConditions.antipolar_bead_types(p.N)
passive_idx = findfirst(==(:passive), bead_types)

println("=" ^ 70)
println("IC diagnostic  |  N=$(p.N)  l=$(p.l)  f0=$(p.f0)  κ=$(p.κ)  ks=$(p.ks)")
println("Passive bead index: $(passive_idx === nothing ? "none (N=$(p.N) is even)" : passive_idx)")
println("=" ^ 70)

r_wca_cut = 2.0^(1.0/6.0) * p.sigma   # WCA cutoff distance

# ── Helper: force decomposition at a given state ─────────────────────────────
function force_breakdown(state, p)
    r  = state.r
    bt = state.bead_types
    Fb = EngineRing.bond_forces(r, p)
    Fk = zeros(size(r)); EngineRing.bending_forces!(Fk, r, p)
    Fw = zeros(size(r)); EngineRing.wca_forces!(Fw, r, p)
    Fa = zeros(size(r)); EngineRing.active_forces!(Fa, r, p, bt)
    return Fb, Fk, Fw, Fa
end

# ── Helper: per-bond lengths ──────────────────────────────────────────────────
function bond_lengths(r, N)
    bl = Vector{Float64}(undef, N)
    for i in 1:N
        j = (i == N) ? 1 : i + 1
        d = r[:, j] - r[:, i]
        bl[i] = norm(d)
    end
    return bl
end

# ── Helper: bending angles at each bead (degrees) ────────────────────────────
function bending_angles_deg(r, N)
    θ = Vector{Float64}(undef, N)
    for i in 1:N
        im = (i == 1) ? N : i - 1
        ip = (i == N) ? 1 : i + 1
        u = r[:, i]  - r[:, im];  u /= max(norm(u), 1e-14)
        v = r[:, ip] - r[:, i];   v /= max(norm(v), 1e-14)
        θ[i] = acosd(clamp(dot(u, v), -1.0, 1.0))
    end
    return θ
end

# ── Helper: WCA overlap report ────────────────────────────────────────────────
function wca_overlaps(r, p)
    N   = p.N
    rc  = r_wca_cut
    overlaps = Tuple{Int,Int,Float64}[]
    for i in 1:N-1
        for j in i+1:N
            if p.topology == :open
                j - i == 1 && continue
            else
                min(j - i, N - (j - i)) == 1 && continue
            end
            d = norm(r[:, j] - r[:, i])
            d < rc && push!(overlaps, (i, j, d))
        end
    end
    return overlaps
end

# ── Run diagnostics for one IC ───────────────────────────────────────────────
function diagnose(label, state, p)
    r   = state.r
    bt  = state.bead_types
    N   = p.N
    Fb, Fk, Fw, Fa = force_breakdown(state, p)
    bl  = bond_lengths(r, N)
    θ   = bending_angles_deg(r, N)
    ovl = wca_overlaps(r, p)

    println("\n── $label ─────────────────────────────────────────────────────")
    @printf("%-5s %-10s %-8s %-8s %-8s %-8s %-8s %-8s\n",
            "bead", "type", "bond_l", "bend°", "|Fbond|", "|Fbend|", "|Fwca|", "|Fact|")
    println("-" ^ 70)
    for i in 1:N
        marker = (bt[i] == :passive) ? " ← passive" :
                 (passive_idx !== nothing && abs(i - passive_idx) <= 1) ? " ← adj" : ""
        @printf("%-5d %-10s %-8.5f %-8.2f %-8.4f %-8.4f %-8.4f %-8.4f%s\n",
                i, string(bt[i]),
                bl[i],
                θ[i],
                norm(Fb[:, i]),
                norm(Fk[:, i]),
                norm(Fw[:, i]),
                norm(Fa[:, i]),
                marker)
    end

    # Active force magnitude check
    println("\n  Active force magnitudes (should be exactly f0=$(p.f0) for active/active_rev):")
    all_ok = true
    for i in 1:N
        (bt[i] == :active || bt[i] == :active_rev) || continue
        mag = norm(Fa[:, i])
        ok  = abs(mag - p.f0) < 1e-10
        all_ok &= ok
        ok || @printf("  MISMATCH bead %d: |Fact| = %.6f  (expected %.6f)\n", i, mag, p.f0)
    end
    all_ok && println("  All active beads: OK")

    # Bond length deviations
    max_dev = maximum(abs.(bl .- p.l))
    println("\n  Max bond length deviation from l=$(p.l): $(round(max_dev; sigdigits=5))")
    for i in 1:N
        abs(bl[i] - p.l) > 1e-10 &&
            @printf("  Bond %d→%d : length = %.6f  (Δ = %+.6f)\n",
                    i, (i==N ? 1 : i+1), bl[i], bl[i]-p.l)
    end

    # WCA overlaps
    println("\n  WCA overlaps (non-bonded pairs with r < $(round(r_wca_cut; sigdigits=5))):")
    if isempty(ovl)
        println("  None — no unintended repulsions at t=0.")
    else
        for (i, j, d) in ovl
            @printf("  OVERLAP beads %d–%d : r = %.5f  (cutoff %.5f)\n", i, j, d, r_wca_cut)
        end
    end
end

# ── Run both ICs ─────────────────────────────────────────────────────────────
state_circle = InitialConditions.perfect_circle_ic(p, bead_types)
state_kinked = InitialConditions.kinked_circle_ic(p, bead_types)

diagnose("CIRCLE IC  (bead_layout = \"antipolar\")",         state_circle, p)
diagnose("KINKED IC  (bead_layout = \"antipolar_kinked\")",  state_kinked, p)

# ── Side-by-side summary ─────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

Fb_c, Fk_c, Fw_c, Fa_c = force_breakdown(state_circle, p)
Fb_k, Fk_k, Fw_k, Fa_k = force_breakdown(state_kinked, p)
Ftot_c = Fb_c .+ Fk_c .+ Fw_c .+ Fa_c
Ftot_k = Fb_k .+ Fk_k .+ Fw_k .+ Fa_k

println("\nPer-bead total force magnitude — circle vs kinked (differences only):")
any_diff = false
for i in 1:p.N
    fc = norm(Ftot_c[:, i])
    fk = norm(Ftot_k[:, i])
    abs(fc - fk) > 1e-8 || continue
    any_diff = true
    @printf("  bead %2d (%-10s): circle = %.5f  kinked = %.5f  Δ = %+.5f\n",
            i, string(bead_types[i]), fc, fk, fk - fc)
end
any_diff || println("  Identical everywhere (as expected for matching geometry).")

println("\nDone.")
