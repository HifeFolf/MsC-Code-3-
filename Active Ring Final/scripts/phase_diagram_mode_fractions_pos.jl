
using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_POS = joinpath(@__DIR__, "..")
const LAST_SECONDS     = 20.0
const FRAMERATE        = 60
const LAST_FRAMES      = round(Int, LAST_SECONDS * FRAMERATE)   # 600

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_05"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_POS, "results", subfolder)
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
println("Position-based mode fractions fₘ = Pₘ/(P₁+…+P₆), last $(round(Int,LAST_SECONDS))s per run.")
println()

κ_raw  = Float64[]
f0_raw = Float64[]
f1_raw = Float64[]
f2_raw = Float64[]
f3_raw = Float64[]
f4_raw = Float64[]
f5_raw = Float64[]
f6_raw = Float64[]

n_loaded = Ref(0)

for (ri, rd) in enumerate(run_dirs)
    ri % 500 == 0 && @printf("  %d / %d\n", ri, n_total)

    local run_path, obs_file, params_file, ep, κ, f₀
    local P1v, P2v, P3v, P4v, P5v, P6v, n, idx
    local P1m, P2m, P3m, P4m, P5m, P6m, total

    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    P1v, P2v, P3v, P4v, P5v, P6v = jldopen(obs_file, "r") do f
        (haskey(f, "P1") ? f["P1"] : nothing,
         haskey(f, "P2") ? f["P2"] : nothing,
         haskey(f, "P3") ? f["P3"] : nothing,
         haskey(f, "P4") ? f["P4"] : nothing,
         haskey(f, "P5") ? f["P5"] : nothing,
         haskey(f, "P6") ? f["P6"] : nothing)
    end
    (P1v === nothing || P2v === nothing) && continue

    n   = length(P1v)
    idx = max(1, n - LAST_FRAMES + 1):n

    P1m = mean(P1v[idx])
    P2m = mean(P2v[idx])
    P3m = P3v !== nothing ? mean(P3v[idx]) : 0.0
    P4m = P4v !== nothing ? mean(P4v[idx]) : 0.0
    P5m = P5v !== nothing ? mean(P5v[idx]) : 0.0
    P6m = P6v !== nothing ? mean(P6v[idx]) : 0.0

    total = P1m + P2m + P3m + P4m + P5m + P6m
    total < 1e-14 && continue

    push!(κ_raw,  κ);  push!(f0_raw, f₀)
    push!(f1_raw, P1m / total)
    push!(f2_raw, P2m / total)
    push!(f3_raw, P3m / total)
    push!(f4_raw, P4m / total)
    push!(f5_raw, P5m / total)
    push!(f6_raw, P6m / total)
    n_loaded[] += 1
end

println("\nLoaded $(n_loaded[]) / $n_total runs.")
isempty(f1_raw) && error("No data loaded — run the sweep with the updated engine first.")

# ── Build regular κ × f₀ grid ────────────────────────────────────────────────

kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f0_raw))
nκ = length(kappa_vals)
nf = length(f0_vals)
@printf("Grid: %d κ values  ×  %d f₀ values\n", nκ, nf)

function build_matrix(frac_raw)
    lookup = Dict(zip(zip(κ_raw, f0_raw), frac_raw))
    mat = fill(NaN, nκ, nf)
    for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
        key = (κ, f₀)
        haskey(lookup, key) && (mat[ic, ir] = lookup[key])
    end
    return mat
end

matrices = [build_matrix(r) for r in (f1_raw, f2_raw, f3_raw, f4_raw, f5_raw, f6_raw)]

# ── Figure: 2×3 heatmaps ─────────────────────────────────────────────────────

const MODE_TITLES_POS = [
    "f₁ = P₁/ΣP   —   ring radius (background)",
    "f₂ = P₂/ΣP   —   elliptical deformation",
    "f₃ = P₃/ΣP   —   triangular deformation",
    "f₄ = P₄/ΣP   —   square deformation",
    "f₅ = P₅/ΣP   —   pentagonal deformation",
    "f₆ = P₆/ΣP   —   hexagonal deformation",
]
const MODE_CMAPS = [:Reds_9, :Greens_9, :Oranges_9, :Blues_9, :Purples_9, :GnBu_9]

panel_positions = [(1,1), (1,3), (1,5), (2,1), (2,3), (2,5)]

fig = Figure(size = (1950, 1100))

Label(fig[0, 1:6],
      "Position-based Fourier mode fractions   fₘ = Pₘ/(P₁+…+P₆) — $subfolder   (last $(round(Int,LAST_SECONDS))s)";
      fontsize = 13, tellwidth = false)

for (k, (row, col)) in enumerate(panel_positions)
    ax = Axis(fig[row, col];
        xlabel = "Active force  f₀",
        ylabel = "Bending stiffness  κ",
        title  = MODE_TITLES_POS[k],
        xticks = LinearTicks(6),
        yticks = LinearTicks(6),
    )
    hm = heatmap!(ax, f0_vals, kappa_vals, matrices[k]';
                  colormap   = MODE_CMAPS[k],
                  colorrange = (0.0, 1.0),
                  nan_color  = :lightgrey)
    Colorbar(fig[row, col+1], hm;
             label  = "fraction of total mode amplitude",
             width  = 16)
end

rowgap!(fig.layout, 1, 20)
for c in (1, 3, 5); colgap!(fig.layout, c, 8); end

out_path = joinpath(results_dir, "phase_diagram_mode_fractions_pos.png")
save(out_path, fig, px_per_unit = 2)
println("Saved: $out_path")
