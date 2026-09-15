

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_PD1 = joinpath(@__DIR__, "..")
const LAST_SECONDS     = 20.0  # use only the last N video-seconds for the b average
const FRAMERATE        = 60    # frames per video second (matches run config)

subfolder  = length(ARGS) >= 1 ? ARGS[1] : "Results_05"
id_start   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_PD1, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan all run directories ──────────────────────────────────────────────────

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder (IDs $id_start – $id_end)…")

# Collect raw (κ, f₀, b) triples
κ_raw = Float64[]
f_raw = Float64[]
b_raw = Float64[]

for rd in run_dirs
    run_path = joinpath(results_dir, rd)
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

    n           = length(b_vec)
    last_frames = round(Int, LAST_SECONDS * FRAMERATE)   # 10.0 s × 60 fps = 600 frames
    idx         = max(1, n - last_frames + 1):n
    b_ss        = mean(b_vec[idx])

    push!(κ_raw, κ)
    push!(f_raw, f₀)
    push!(b_raw, b_ss)
end

println("Loaded $(length(b_raw)) runs with valid observables.")

# ── Build regular κ × f₀ grid ────────────────────────────────────────────────

kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f_raw))
nκ = length(kappa_vals)
nf = length(f0_vals)

@printf("Grid: %d κ values  ×  %d f₀ values  =  %d cells\n", nκ, nf, nκ * nf)

# Build lookup: (κ, f₀) → b
b_lookup = Dict(zip(zip(κ_raw, f_raw), b_raw))

# Fill 2-D matrix (rows = f₀ index, cols = κ index — CairoMakie heatmap is [x, y])
b_matrix = fill(NaN, nκ, nf)
for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
    key = (κ, f₀)
    haskey(b_lookup, key) && (b_matrix[ic, ir] = b_lookup[key])
end

n_missing = count(isnan, b_matrix)
n_missing > 0 && @printf("  %d cells missing (will appear white)\n", n_missing)

# ── Figure ────────────────────────────────────────────────────────────────────

fig = Figure(size = (900, 750))

Label(fig[0, 1],
      "Phase diagram — $subfolder   (kBT = 0.1 fixed,  N = 16)";
      fontsize = 14, tellwidth = false)

ax = Axis(fig[1, 1];
    xlabel        = "Active force  f₀",
    ylabel        = "Bending stiffness  κ",
    title         = "Mean asphericity  b  (last $(round(Int, LAST_SECONDS))s = $(round(Int, LAST_SECONDS*FRAMERATE)) frames)",
    xticks        = LinearTicks(8),
    yticks        = LinearTicks(8),
)

hm = heatmap!(ax, f0_vals, kappa_vals, b_matrix';
              colormap   = :viridis,
              colorrange = (0.0, min(1.0, maximum(filter(!isnan, b_matrix)))),
              nan_color  = :white)

Colorbar(fig[1, 2], hm;
         label  = "mean asphericity  b  (0 = circular, 1 = rod-like)",
         width  = 18)

# ── Linear stability boundary overlay (optional) ─────────────────────────────
added_boundary = false
for N_try in [17, 16]
    csv_path = joinpath(PROJECT_ROOT_PD1, "results", "linear_stability",
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

out_path = joinpath(results_dir, "phase_diagram_01.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
