# phase_diagram_log.jl
# Phase diagram for a log-spaced κ × f₀ sweep (e.g. Results_log).
#
# Both axes are logarithmic. Data are binned onto a regular log grid and
# shown as a heatmap. Works for any sweep folder with log-spaced κ and f₀.
#
# Color: mean asphericity b averaged over the last LAST_SECONDS of each run.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/phase_diagram_log.jl
#   julia --project scripts/phase_diagram_log.jl Results_log
#   julia --project scripts/phase_diagram_log.jl Results_log 1 1521
#
# Arguments:
#   arg 1 — results subfolder   (default: Results_log)
#   arg 2 — first run ID        (default: 1)
#   arg 3 — last  run ID        (default: all)
#
# Output:  results/<subfolder>/phase_diagram_log.png

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_PL = joinpath(@__DIR__, "..")
const LAST_SECONDS    = 20.0
const FRAMERATE       = 60
const LAST_FRAMES     = round(Int, LAST_SECONDS * FRAMERATE)   # 1200

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_snapshot"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_PL, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan runs ─────────────────────────────────────────────────────────────────

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

n_total = length(run_dirs)
println("Scanning $n_total runs from $subfolder (IDs $id_start – $id_end)…")
println("Using last $(round(Int, LAST_SECONDS))s ($LAST_FRAMES frames) per run.")

κ_raw  = Float64[]
f0_raw = Float64[]
b_raw  = Float64[]

for rd in run_dirs
    local run_path, obs_file, params_file, ep, κ, f₀, b_vec, n, idx

    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    b_vec = jldopen(obs_file, "r") do f
        haskey(f, "b") ? f["b"] : nothing
    end
    b_vec === nothing && continue

    n   = length(b_vec)
    idx = max(1, n - LAST_FRAMES + 1):n
    push!(κ_raw,  κ)
    push!(f0_raw, f₀)
    push!(b_raw,  mean(b_vec[idx]))
end

println("Loaded $(length(b_raw)) runs with valid observables.")
isempty(b_raw) && error("No data loaded.")

# ── Build regular log κ × log f₀ grid ────────────────────────────────────────

kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f0_raw))
nκ = length(kappa_vals)
nf = length(f0_vals)
@printf("Grid: %d κ values  ×  %d f₀ values\n", nκ, nf)

b_lookup = Dict(zip(zip(κ_raw, f0_raw), b_raw))
b_matrix = fill(NaN, nκ, nf)
for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
    key = (κ, f₀)
    haskey(b_lookup, key) && (b_matrix[ic, ir] = b_lookup[key])
end

n_missing = count(isnan, b_matrix)
n_missing > 0 && @printf("  %d cells missing (will appear white)\n", n_missing)

# ── Figure ────────────────────────────────────────────────────────────────────

fig = Figure(size = (950, 800))

Label(fig[0, 1],
      "Phase diagram (log axes) — $subfolder   (kBT = 0.1,  N = 17)";
      fontsize = 14, tellwidth = false)

ax = Axis(fig[1, 1];
    xlabel  = "Active force  f₀",
    ylabel  = "Bending stiffness  κ",
    title   = "Mean asphericity  b  (last $(round(Int, LAST_SECONDS))s = $LAST_FRAMES frames)",
    xscale  = log10,
    yscale  = log10,
    xticks  = LogTicks(WilkinsonTicks(6)),
    yticks  = LogTicks(WilkinsonTicks(6)),
)

b_finite   = filter(!isnan, b_matrix)
cmax       = isempty(b_finite) ? 1.0 : min(1.0, maximum(b_finite))

hm = heatmap!(ax, f0_vals, kappa_vals, b_matrix';
              colormap   = :viridis,
              colorrange = (0.0, cmax),
              nan_color  = :white)

Colorbar(fig[1, 2], hm;
         label  = "mean asphericity  b  (0 = circular,  1 = rod-like)",
         width  = 18)

# ── Linear stability boundary overlay (optional) ─────────────────────────────
added_boundary = false
for N_try in [17, 16]
    csv_path = joinpath(PROJECT_ROOT_PL, "results", "linear_stability",
                        "phase_boundary_N$(N_try).csv")
    isfile(csv_path) || continue
    fields  = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
    κ_stab  = [parse(Float64, f[1]) for f in fields]
    f0_stab = [parse(Float64, f[2]) for f in fields]
    in_range = (f0_stab .>= minimum(f0_vals)) .& (f0_stab .<= maximum(f0_vals)) .&
               (κ_stab  .>= minimum(kappa_vals)) .& (κ_stab  .<= maximum(kappa_vals))
    any(in_range) || continue
    lines!(ax, f0_stab[in_range], κ_stab[in_range];
           color = :white, linewidth = 2.5, linestyle = :dash,
           label = "Linear stability (N=$N_try)")
    println("  Overlaid linear stability boundary from $csv_path")
    added_boundary = true
end
added_boundary && axislegend(ax; position = :lt, backgroundcolor = (:black, 0.4),
                              labelcolor = :white)

out_path = joinpath(results_dir, "phase_diagram_log.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
