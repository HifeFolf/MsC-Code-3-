# phase_diagram_passive_angle.jl
# Phase diagram whose heatmap colour is the mean turning angle at the passive
# junction bead (Kconn, in degrees), averaged over the last 20 s of each run.
#
# Physical reference:
#   +360/N ≈ +21.2° (N=17)  — equilibrium (perfect circle, all beads equal)
#   more positive             — passive bead buckles outward
#   more negative / zero      — passive bead buckles inward (kinked/collapsed)
#
# The diverging colormap is centred at the equilibrium value 360/N so that
# red = inward-buckled, white = circular, blue = outward-buckled.
#
# Panel 2 shows std(Kconn) on a LOG colour scale, which separates the thermal
# floor (≈static) from genuine junction motion — a linear scale buries both in
# the same dark colour.  Panel 3 is a categorical phase map that classifies each
# cell into inward-buckled / circular / outward-buckled / dynamic using the
# S_DYN and M_TOL thresholds below (tune them to your data).
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_passive_angle.jl
#   julia --project scripts/phase_diagram_passive_angle.jl Results_N17_kinked
#   julia --project scripts/phase_diagram_passive_angle.jl Results_N17_circle 1 1500

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_PA = joinpath(@__DIR__, "..")
const LAST_FRAMES     = round(Int, 20.0 * 60)   # last 20 s at 60 fps

# ── Phase-classification thresholds (all in degrees — tune to your data) ──────
# A cell is "dynamic" if the junction angle fluctuates more than S_DYN over the
# window; otherwise it is a static state classified by how far its mean turning
# angle sits from the equilibrium value 360/N (±M_TOL = circular).
const S_DYN   = 15.0    # std above this ⇒ dynamic (oscillating/rotating/chaotic)
const M_TOL   = 10.0    # |mean − 360/N| within this ⇒ "circular" (near equilibrium)
const S_FLOOR = 1.0     # lower clip for the log std colour scale (thermal floor)

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_PA, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan runs ─────────────────────────────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder…")
println("Using last 20 s ($LAST_FRAMES frames) per run.")

κ_raw  = Float64[]
f0_raw = Float64[]
K_raw  = Float64[]   # mean Kconn in degrees
Ks_raw = Float64[]   # std of Kconn in degrees (static vs dynamic diagnostic)
N_ring = Ref(17)     # read from first successful params.toml

for rd in run_dirs
    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])
    N_ring[] = Int(ep["N"])

    Kv = jldopen(obs_file, "r") do f
        haskey(f, "Kconn") ? f["Kconn"] : nothing
    end
    Kv === nothing && continue

    n   = length(Kv)
    idx = max(1, n - LAST_FRAMES + 1):n
    push!(κ_raw,  κ)
    push!(f0_raw, f₀)
    push!(K_raw,  rad2deg(mean(Kv[idx])))
    push!(Ks_raw, length(idx) > 1 ? rad2deg(std(Kv[idx])) : 0.0)
end

println("Loaded $(length(K_raw)) runs with Kconn data.")
isempty(K_raw) && error("No Kconn data found — was this sweep run with run_sweep.jl?")

# ── Build regular log-grid ────────────────────────────────────────────────────
kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f0_raw))
nκ = length(kappa_vals); nf = length(f0_vals)
@printf("Grid: %d κ values × %d f₀ values\n", nκ, nf)

K_lookup  = Dict(zip(zip(κ_raw, f0_raw), K_raw))
Ks_lookup = Dict(zip(zip(κ_raw, f0_raw), Ks_raw))
K_matrix  = fill(NaN, nκ, nf)
Ks_matrix = fill(NaN, nκ, nf)
for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
    haskey(K_lookup,  (κ, f₀)) && (K_matrix[ic, ir]  = K_lookup[(κ, f₀)])
    haskey(Ks_lookup, (κ, f₀)) && (Ks_matrix[ic, ir] = Ks_lookup[(κ, f₀)])
end

n_missing = count(isnan, K_matrix)
n_missing > 0 && @printf("  %d cells missing (shown as white)\n", n_missing)

# ── Diverging colormap centred at equilibrium angle 360/N ────────────────────
N_val     = N_ring[]
K_eq      = 360.0 / N_val          # equilibrium turning angle in degrees
K_finite  = filter(!isnan, K_matrix)
K_lo, K_hi = isempty(K_finite) ? (K_eq - 20.0, K_eq + 20.0) :
             (minimum(K_finite), maximum(K_finite))

# Symmetric range around K_eq so the white/neutral point is centred correctly
half_range = max(K_eq - K_lo, K_hi - K_eq) * 1.05   # 5 % headroom
clo = K_eq - half_range
chi = K_eq + half_range

@printf("Kconn range in data: %.1f° … %.1f°\n", K_lo, K_hi)
@printf("Equilibrium (360/%d): %.2f°\n", N_val, K_eq)
@printf("Colormap range: %.1f° … %.1f°\n", clo, chi)

# std colour range: thermal floor → max observed fluctuation (log-scaled)
Ks_finite = filter(!isnan, Ks_matrix)
Ks_hi     = isempty(Ks_finite) ? 1.0 : maximum(Ks_finite)
Ks_hi     = max(Ks_hi, S_FLOOR * 10)            # guard against all-static data
if !isempty(Ks_finite)
    qs = quantile(Ks_finite, [0.1, 0.5, 0.9])
    @printf("Kconn std: min %.2f°  p10 %.2f°  median %.2f°  p90 %.2f°  max %.2f°\n",
            minimum(Ks_finite), qs[1], qs[2], qs[3], maximum(Ks_finite))
    @printf("  (S_DYN = %.1f° → %d of %d cells classified dynamic)\n",
            S_DYN, count(>(S_DYN), Ks_finite), length(Ks_finite))
end
# Clamp to the thermal floor so the log scale never sees a zero/▽ value.
Ks_disp = map(x -> isnan(x) ? NaN : max(x, S_FLOOR), Ks_matrix)

# ── Categorical phase classification ─────────────────────────────────────────
# 1 = inward-buckled (static)   2 = circular (static)
# 3 = outward-buckled (static)  4 = dynamic
function classify_phase(m, s)
    isnan(m) && return NaN
    s > S_DYN              && return 4.0          # dynamics override the mean
    m > K_eq + M_TOL       && return 3.0          # outward
    m < K_eq - M_TOL       && return 1.0          # inward
    return 2.0                                    # near equilibrium ⇒ circular
end
phase_matrix = [classify_phase(K_matrix[i, j], Ks_matrix[i, j])
                for i in 1:nκ, j in 1:nf]
phase_colors = [:darkorange, :gainsboro, :steelblue, :crimson]
phase_labels = ["Inward-buckled (static)", "Circular (static)",
                "Outward-buckled (static)", "Dynamic"]
for (code, lab) in zip(1:4, phase_labels)
    @printf("  phase %d  %-26s %d cells\n",
            code, lab, count(==(Float64(code)), filter(!isnan, vec(phase_matrix))))
end

# ── Linear stability boundary (optional, shared by all panels) ───────────────
function load_stability_boundary()
    for N_try in [N_val, 16, 17]
        csv_path = joinpath(PROJECT_ROOT_PA, "results", "linear_stability",
                            "phase_boundary_N$(N_try).csv")
        isfile(csv_path) || continue
        rows    = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
        κ_stab  = [parse(Float64, r[1]) for r in rows]
        f0_stab = [parse(Float64, r[2]) for r in rows]
        in_range = (f0_stab .>= f0_vals[1]) .& (f0_stab .<= f0_vals[end]) .&
                   (κ_stab  .>= kappa_vals[1]) .& (κ_stab  .<= kappa_vals[end])
        any(in_range) || continue
        println("  Stability boundary loaded (N=$N_try).")
        return (f0_stab[in_range], κ_stab[in_range])
    end
    return (nothing, nothing)
end
stab_f0, stab_κ = load_stability_boundary()

function overlay_stability!(axis)
    stab_κ === nothing && return
    lines!(axis, stab_f0, stab_κ;
           color = :white, linewidth = 2.5, linestyle = :dash,
           label = "Linear stability")
    axislegend(axis; position = :lt, backgroundcolor = (:black, 0.4),
               labelcolor = :white)
end

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (950, 2150))

Label(fig[0, 1:2],
      "Passive-bead turning angle — $subfolder   (N=$N_val, kBT=0.1)";
      fontsize = 14, tellwidth = false)

# ── Panel 1: mean Kconn (inward / outward buckling) ──────────────────────────
ax = Axis(fig[1, 1];
          xlabel = "Active force  f₀",
          ylabel = "Bending stiffness  κ",
          title  = "Mean Kconn  (last 20 s)   " *
                   "| equilibrium = $(round(K_eq; digits=1))°   " *
                   "| red = inward-buckled,  blue = outward-buckled",
          xscale = log10, yscale = log10,
          xticks = LogTicks(WilkinsonTicks(6)),
          yticks = LogTicks(WilkinsonTicks(6)))

hm = heatmap!(ax, f0_vals, kappa_vals, K_matrix';
              colormap   = :RdBu,
              colorrange = (clo, chi),
              nan_color  = :lightgrey)

Colorbar(fig[1, 2], hm;
         label = "mean turning angle at passive bead  (degrees)",
         width = 18)

# Equilibrium contour (Kconn = 360/N)
contour!(ax, f0_vals, kappa_vals, K_matrix';
         levels    = [K_eq],
         color     = :black,
         linewidth = 1.5,
         linestyle = :dot)

overlay_stability!(ax)

# ── Panel 2: std of Kconn — LOG scale (separates thermal floor from motion) ───
ax2 = Axis(fig[2, 1];
           xlabel = "Active force  f₀",
           ylabel = "Bending stiffness  κ",
           title  = "std(Kconn)  (last 20 s, log scale)   " *
                    "| dark = static (thermal floor),  bright = dynamic",
           xscale = log10, yscale = log10,
           xticks = LogTicks(WilkinsonTicks(6)),
           yticks = LogTicks(WilkinsonTicks(6)))

hm2 = heatmap!(ax2, f0_vals, kappa_vals, Ks_disp';
               colormap   = :viridis,
               colorscale = log10,
               colorrange = (S_FLOOR, Ks_hi),
               nan_color  = :lightgrey)

Colorbar(fig[2, 2], hm2;
         label = "std of turning angle at passive bead  (degrees, log)",
         width = 18)

overlay_stability!(ax2)

# ── Panel 3: categorical phase map (crisp phase boundaries) ───────────────────
ax3 = Axis(fig[3, 1];
           xlabel = "Active force  f₀",
           ylabel = "Bending stiffness  κ",
           title  = "Phase classification   " *
                    "(S_DYN = $(round(Int, S_DYN))°,  circular = 360/N ± $(round(Int, M_TOL))°)",
           xscale = log10, yscale = log10,
           xticks = LogTicks(WilkinsonTicks(6)),
           yticks = LogTicks(WilkinsonTicks(6)))

heatmap!(ax3, f0_vals, kappa_vals, phase_matrix';
         colormap   = cgrad(phase_colors; categorical = true),
         colorrange = (0.5, 4.5),
         nan_color  = :white)

Legend(fig[3, 2],
       [PolyElement(color = c) for c in phase_colors],
       phase_labels, "Phase";
       framevisible = true)

overlay_stability!(ax3)

out_path = joinpath(results_dir, "phase_diagram_passive_angle.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
