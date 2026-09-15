

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_MP = joinpath(@__DIR__, "..")
const LAST_SECONDS    = 20.0
const FRAMERATE       = 60
const LAST_FRAMES     = round(Int, LAST_SECONDS * FRAMERATE)   # 1200

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_05"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_MP, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan runs ─────────────────────────────────────────────────────────────────

println("Loading runs from $subfolder (IDs $id_start – $id_end)…")
println("Comparing P₂ vs P₃ (position modes) from last $(round(Int, LAST_SECONDS))s ($LAST_FRAMES frames) per run.")
println()

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

κ_vals   = Float64[]
f0_vals  = Float64[]
dom_mode = Int[]      # 2 = P2 dominant, 3 = P3 dominant

n_total  = length(run_dirs)
n_loaded = Ref(0)

for (ri, rd) in enumerate(run_dirs)
    ri % 500 == 0 && @printf("  %d / %d\n", ri, n_total)

    local run_path, obs_file, params_file, ep, κ, f₀, P2m, P3m

    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    P2m, P3m = jldopen(obs_file, "r") do f
        P2 = haskey(f, "P2") ? mean(f["P2"][max(1, length(f["P2"]) - LAST_FRAMES + 1):end]) : nothing
        P3 = haskey(f, "P3") ? mean(f["P3"][max(1, length(f["P3"]) - LAST_FRAMES + 1):end]) : nothing
        (P2, P3)
    end
    (P2m === nothing || P3m === nothing) && continue

    push!(κ_vals,   κ)
    push!(f0_vals,  f₀)
    push!(dom_mode, P2m >= P3m ? 2 : 3)
    n_loaded[] += 1
end

println("\nLoaded $(n_loaded[]) / $n_total runs.")
isempty(dom_mode) && error("No data found — run the sweep with the updated engine first (P2, P3 keys required).")

# ── Figure ────────────────────────────────────────────────────────────────────

const MODE_COLORS = Dict(
    2 => RGBf(0.18, 0.70, 0.30),   # P2 — green
    3 => RGBf(0.95, 0.50, 0.10),   # P3 — orange
)
const MODE_LABELS = Dict(
    2 => "P₂ — elliptical deformation",
    3 => "P₃ — triangular deformation",
)

point_colors  = [MODE_COLORS[dom_mode[i]] for i in eachindex(dom_mode)]
present_modes = sort(unique(dom_mode))

fig = Figure(size = (950, 800))

Label(fig[0, 1],
      "Dominant position Fourier mode (P₂ vs P₃) — $subfolder   (last $(round(Int, LAST_SECONDS))s,  N = 17)";
      fontsize = 13, tellwidth = false)

ax = Axis(fig[1, 1];
    xlabel = "Active force  f₀",
    ylabel = "Bending stiffness  κ",
    title  = "Cell colour = P₂ (elliptical) vs P₃ (triangular)",
    xticks = LinearTicks(8),
    yticks = LinearTicks(8),
)

scatter!(ax, f0_vals, κ_vals;
         color       = point_colors,
         markersize  = 5,
         strokewidth = 0)

legend_elements = [MarkerElement(color = MODE_COLORS[m], marker = :circle,
                                 markersize = 14) for m in present_modes]
Legend(fig[1, 2], legend_elements, [MODE_LABELS[m] for m in present_modes];
       framevisible = true, labelsize = 13)

out_path = joinpath(results_dir, "phase_diagram_modes_pos.png")
save(out_path, fig, px_per_unit = 2)
println("Saved: $out_path")
