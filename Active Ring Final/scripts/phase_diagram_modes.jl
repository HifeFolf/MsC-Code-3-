

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT  = joinpath(@__DIR__, "..")
const LAST_SECONDS  = 20.0
const FRAMERATE     = 60
const LAST_FRAMES   = round(Int, LAST_SECONDS * FRAMERATE)   # 1200

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_kinked2"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan all runs ─────────────────────────────────────────────────────────────

println("Loading runs from $subfolder (IDs $id_start – $id_end)…")
println("Computing dominant mode (A₁–A₄) from last $(round(Int, LAST_SECONDS))s ($(LAST_FRAMES) frames) of each run.")
println()

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

κ_vals   = Float64[]
f0_vals  = Float64[]
dom_mode = Int[]      # 1=A1, 2=A2, 3=A3, 4=A4

n_total  = length(run_dirs)
n_loaded = Ref(0)

for (ri, rd) in enumerate(run_dirs)
    ri % 500 == 0 && @printf("  %d / %d\n", ri, n_total)

    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    means = jldopen(obs_file, "r") do f
        # Load whichever modes are available; fall back gracefully for old data
        A1 = haskey(f, "A1") ? mean(f["A1"][max(1, length(f["A1"])-LAST_FRAMES+1):end]) : -Inf
        A2 = haskey(f, "A2") ? mean(f["A2"][max(1, length(f["A2"])-LAST_FRAMES+1):end]) : -Inf
        A3 = haskey(f, "A3") ? mean(f["A3"][max(1, length(f["A3"])-LAST_FRAMES+1):end]) : -Inf
        A4 = haskey(f, "A4") ? mean(f["A4"][max(1, length(f["A4"])-LAST_FRAMES+1):end]) : -Inf
        (A1, A2, A3, A4)
    end
    all(m -> m == -Inf, means) && continue

    dom = argmax(collect(means))   # 1=A1, 2=A2, 3=A3, 4=A4

    push!(κ_vals,   κ)
    push!(f0_vals,  f₀)
    push!(dom_mode, dom)
    n_loaded[] += 1
end

println("\nLoaded $(n_loaded[]) / $n_total runs.")

# ── Figure ────────────────────────────────────────────────────────────────────

const MODE_COLORS = [
    RGBf(0.85, 0.15, 0.15),   # A1 — red      (polar / tear-drop)
    RGBf(0.18, 0.70, 0.30),   # A2 — green    (elliptical)
    RGBf(0.95, 0.50, 0.10),   # A3 — orange   (triangular)
    RGBf(0.15, 0.45, 0.85),   # A4 — blue     (square)
]
const MODE_LABELS = [
    "A₁ — polar / tear-drop (one-lobe)",
    "A₂ — elliptical (two-lobe)",
    "A₃ — triangular (three-lobe)",
    "A₄ — square (four-lobe)",
]

point_colors = [MODE_COLORS[dom_mode[i]] for i in eachindex(dom_mode)]

# Only include legend entries for modes that actually appear
present_modes = sort(unique(dom_mode))

fig = Figure(size = (950, 800))

Label(fig[0, 1],
      "Dominant contour Fourier mode (A₁–A₄) — $subfolder   (last $(round(Int, LAST_SECONDS))s,  N = 16)";
      fontsize = 13, tellwidth = false)

ax = Axis(fig[1, 1];
    xlabel = "Active force  f₀",
    ylabel = "Bending stiffness  κ",
    title  = "Cell colour = mode with highest mean amplitude",
    xticks = LinearTicks(8),
    yticks = LinearTicks(8),
)

scatter!(ax, f0_vals, κ_vals;
         color       = point_colors,
         markersize  = 5,
         strokewidth = 0)

# Legend — only show modes that appear in the data
legend_elements = [MarkerElement(color = MODE_COLORS[m], marker = :circle,
                                 markersize = 14) for m in present_modes]
Legend(fig[1, 2], legend_elements, MODE_LABELS[present_modes];
       framevisible = true, labelsize = 13)

out_path = joinpath(results_dir, "phase_diagram_modes.png")
save(out_path, fig, px_per_unit = 2)
println("Saved: $out_path")
