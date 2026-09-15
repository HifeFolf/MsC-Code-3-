# phase_diagram_local_curvature.jl
# Phase diagram using the mean local curvature over the passive bead and its
# HW=3 nearest neighbours on each side — a 7-bead window centred on the passive.
#
# Observable (mean signed turning angle per bead in window):
#
#   κ_loc = (1/M) Σ_{j=p-HW}^{p+HW} dψ_j    [degrees / bead]
#
# where M = 2·HW+1 = 7, dψ_j = signed change in bond direction at bead j,
# p = passive bead index.
#
# Key identity: the sum telescopes to just two bond-direction evaluations:
#
#   Σ_{j=p-HW}^{p+HW} dψ_j = ψ_out − ψ_in
#
#   ψ_out = direction of bond (p+HW)   → (p+HW+1)   [outgoing window boundary]
#   ψ_in  = direction of bond (p−HW−1) → (p−HW)     [incoming window boundary]
#
# Physical reference (N=17, HW=3, M=7):
#   κ_eq = 360/N ≈ 21.2 °/bead   (perfect circle — all beads equal)
#   κ_loc < κ_eq  →  passive region curves less than circle  →  inward-buckled
#   κ_loc > κ_eq  →  passive region curves more than circle  →  outward-buckled
#
# Diverging colormap RdBu centred at κ_eq:
#   red = inward-buckled,  white = circular,  blue = outward-buckled
#
# NOTE: Computed from traj.jld2 (not from observables.jld2 — this metric is not
#       stored there).  Loading trajectories for a full 1500-run sweep takes
#       a few minutes depending on disk speed.
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_local_curvature.jl
#   julia --project scripts/phase_diagram_local_curvature.jl Results_N17_kinked
#   julia --project scripts/phase_diagram_local_curvature.jl Results_N17_circle 1 750

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_LC = joinpath(@__DIR__, "..")
const HW          = 3                          # beads on each side of passive
const M           = 2 * HW + 1                 # window width = 7
const LAST_FRAMES = round(Int, 20.0 * 60)      # last 20 s at 60 fps = 1200

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_snapshot"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_LC, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Core computation ───────────────────────────────────────────────────────────
#
# Loads traj.jld2, iterates over the last LAST_FRAMES frames, and for each frame
# computes κ_loc via the telescoping identity (two bond evals per frame).
# Returns the time-averaged κ_loc in degrees per bead, or NaN on failure.
function mean_window_curvature_deg(traj_path, p, N)
    jldopen(traj_path, "r") do f
        pos   = f["positions"]                # 2 × N × n_frames
        n_tot = size(pos, 3)
        i_lo  = max(1, n_tot - LAST_FRAMES + 1)

        acc   = 0.0
        count = 0
        for fr in i_lo:n_tot
            r = view(pos, :, :, fr)

            # outgoing boundary bond: bead (p+HW) → bead (p+HW+1)
            bout1 = mod1(p + HW,     N)
            bout2 = mod1(p + HW + 1, N)
            ψ_out = atan(r[2, bout2] - r[2, bout1], r[1, bout2] - r[1, bout1])

            # incoming boundary bond: bead (p−HW−1) → bead (p−HW)
            bin1  = mod1(p - HW - 1, N)
            bin2  = mod1(p - HW,     N)
            ψ_in  = atan(r[2, bin2] - r[2, bin1], r[1, bin2] - r[1, bin1])

            # total turning over window (wrapped), converted to per-bead degrees
            δ = ψ_out - ψ_in
            δ -= 2π * round(δ / (2π))          # wrap to (−π, π]
            acc   += rad2deg(δ / M)
            count += 1
        end
        count > 0 ? acc / count : NaN
    end
end

# ── Scan runs ─────────────────────────────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

n_total = length(run_dirs)
println("Scanning $n_total runs from $subfolder  (window M=$M beads, ±$HW neighbours)…")
println("Using last 20 s ($LAST_FRAMES frames) per run.")
println("Loading from traj.jld2 — this may take a few minutes for large sweeps.\n")

κ_raw  = Float64[]
f0_raw = Float64[]
K_raw  = Float64[]
N_ring = Ref(17)

for (i, rd) in enumerate(run_dirs)
    run_path   = joinpath(results_dir, rd)
    traj_file  = joinpath(run_path, "traj.jld2")
    param_file = joinpath(run_path, "params.toml")
    (isfile(traj_file) && isfile(param_file)) || continue

    ep = TOML.parsefile(param_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])
    N  = Int(ep["N"])
    N_ring[] = N

    # passive bead index: only defined for odd N (antipolar layout)
    isodd(N) || continue
    p = (N - 1) ÷ 2 + 1

    Kloc = try
        mean_window_curvature_deg(traj_file, p, N)
    catch e
        @warn "  Skipping $rd: $e"
        NaN
    end
    isnan(Kloc) && continue

    push!(κ_raw, κ); push!(f0_raw, f₀); push!(K_raw, Kloc)
    i % 200 == 0 && @printf("  %d / %d runs processed\n", i, n_total)
end

println("Loaded $(length(K_raw)) runs.")
isempty(K_raw) && error("No data loaded — check that traj.jld2 files exist.")

# ── Build regular log-grid ────────────────────────────────────────────────────
kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f0_raw))
nκ = length(kappa_vals); nf = length(f0_vals)
@printf("Grid: %d κ values × %d f₀ values\n", nκ, nf)

K_lookup = Dict(zip(zip(κ_raw, f0_raw), K_raw))
K_matrix = fill(NaN, nκ, nf)
for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
    haskey(K_lookup, (κ, f₀)) && (K_matrix[ic, ir] = K_lookup[(κ, f₀)])
end

n_missing = count(isnan, K_matrix)
n_missing > 0 && @printf("  %d cells missing (shown as grey)\n", n_missing)

# ── Diverging colormap centred at equilibrium curvature ──────────────────────
N_val    = N_ring[]
K_eq     = 360.0 / N_val             # equilibrium °/bead for a perfect circle
K_finite = filter(!isnan, K_matrix)
K_lo, K_hi = isempty(K_finite) ? (K_eq - 20.0, K_eq + 20.0) :
             (minimum(K_finite), maximum(K_finite))
half_rng = max(K_eq - K_lo, K_hi - K_eq) * 1.05   # 5 % headroom
clo = K_eq - half_rng
chi = K_eq + half_rng

@printf("\nκ_local range in data: %.2f° … %.2f° / bead\n", K_lo, K_hi)
@printf("Equilibrium (360/%d): %.2f° / bead  (= %.3f rad/bead)\n",
        N_val, K_eq, deg2rad(K_eq))
@printf("Colormap range: %.2f° … %.2f°\n", clo, chi)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (950, 800))

Label(fig[0, 1],
      "Local curvature ($M-bead window around passive) — $subfolder   (N=$N_val, kBT=0.1)";
      fontsize = 14, tellwidth = false)

ax = Axis(fig[1, 1];
          xlabel = "Active force  f₀",
          ylabel = "Bending stiffness  κ",
          title  = "Mean κ_local  (last 20 s, ±$HW neighbours)   " *
                   "| κ_eq = $(round(K_eq; digits=1))° / bead   " *
                   "| red = inward-buckled,  blue = outward-buckled",
          xscale = log10, yscale = log10,
          xticks = LogTicks(WilkinsonTicks(6)),
          yticks = LogTicks(WilkinsonTicks(6)))

hm = heatmap!(ax, f0_vals, kappa_vals, K_matrix';
              colormap   = :RdBu,
              colorrange = (clo, chi),
              nan_color  = :lightgrey)

Colorbar(fig[1, 2], hm;
         label = "mean local curvature  κ_local  (° / bead)",
         width = 18)

# Equilibrium isoline (κ_local = 360/N)
contour!(ax, f0_vals, kappa_vals, K_matrix';
         levels    = [K_eq],
         color     = :black,
         linewidth = 1.5,
         linestyle = :dot)

# Linear stability boundary (optional — loaded from CSV if present)
for N_try in [N_val, 16, 17]
    csv_path = joinpath(PROJECT_ROOT_LC, "results", "linear_stability",
                        "phase_boundary_N$(N_try).csv")
    isfile(csv_path) || continue
    rows    = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
    κ_stab  = [parse(Float64, r[1]) for r in rows]
    f0_stab = [parse(Float64, r[2]) for r in rows]
    mask = (f0_stab .>= f0_vals[1]) .& (f0_stab .<= f0_vals[end]) .&
           (κ_stab  .>= kappa_vals[1]) .& (κ_stab  .<= kappa_vals[end])
    any(mask) || continue
    lines!(ax, f0_stab[mask], κ_stab[mask];
           color = :white, linewidth = 2.5, linestyle = :dash,
           label = "Linear stability")
    axislegend(ax; position = :lt, backgroundcolor = (:black, 0.4),
               labelcolor = :white)
    println("  Stability boundary overlaid (N=$N_try).")
    break
end

out_path = joinpath(results_dir, "phase_diagram_local_curvature.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
