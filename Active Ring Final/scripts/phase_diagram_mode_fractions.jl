
using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_MF = joinpath(@__DIR__, "..")
const LAST_SECONDS    = 20.0
const FRAMERATE       = 60
const LAST_FRAMES     = round(Int, LAST_SECONDS * FRAMERATE)   # 600

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_05"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_MF, "results", subfolder)
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
println("Mode fractions fₘ = Aₘ/(A₁+…+A₆), last $(round(Int,LAST_SECONDS))s per run.")
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
    local A1v, A2v, A3v, A4v, A5v, A6v, n, idx
    local A1m, A2m, A3m, A4m, A5m, A6m, total

    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    A1v, A2v, A3v, A4v, A5v, A6v = jldopen(obs_file, "r") do f
        (haskey(f, "A1") ? f["A1"] : nothing,
         haskey(f, "A2") ? f["A2"] : nothing,
         haskey(f, "A3") ? f["A3"] : nothing,
         haskey(f, "A4") ? f["A4"] : nothing,
         haskey(f, "A5") ? f["A5"] : nothing,
         haskey(f, "A6") ? f["A6"] : nothing)
    end
    A2v === nothing && continue   # A2 is the minimum required

    n   = length(A2v)
    idx = max(1, n - LAST_FRAMES + 1):n

    # Fall back to 0 for modes absent in old runs
    A1m = A1v !== nothing ? mean(A1v[idx]) : 0.0
    A2m = mean(A2v[idx])
    A3m = A3v !== nothing ? mean(A3v[idx]) : 0.0
    A4m = A4v !== nothing ? mean(A4v[idx]) : 0.0
    A5m = A5v !== nothing ? mean(A5v[idx]) : 0.0
    A6m = A6v !== nothing ? mean(A6v[idx]) : 0.0

    total = A1m + A2m + A3m + A4m + A5m + A6m
    total < 1e-14 && continue

    push!(κ_raw,  κ);  push!(f0_raw, f₀)
    push!(f1_raw, A1m / total)
    push!(f2_raw, A2m / total)
    push!(f3_raw, A3m / total)
    push!(f4_raw, A4m / total)
    push!(f5_raw, A5m / total)
    push!(f6_raw, A6m / total)
    n_loaded[] += 1
end

println("\nLoaded $(n_loaded[]) / $n_total runs.")
isempty(f2_raw) && error("No data loaded.")

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

# ── Figure: 2×3 heatmaps ──────────────────────────────────────────────────────

const MODE_TITLES = [
    "f₁ = A₁/ΣA   —   polar / tear-drop",
    "f₂ = A₂/ΣA   —   elliptical",
    "f₃ = A₃/ΣA   —   triangular",
    "f₄ = A₄/ΣA   —   square",
    "f₅ = A₅/ΣA   —   pentagonal",
    "f₆ = A₆/ΣA   —   hexagonal",
]
const MODE_COLORS = [:Reds_9, :Greens_9, :Oranges_9, :Blues_9, :Purples_9, :GnBu_9]

# Layout: 2 rows × 3 panels, each panel has a colorbar in the next column
# Columns: 1=hm, 2=cb, 3=hm, 4=cb, 5=hm, 6=cb
panel_positions = [(1,1), (1,3), (1,5), (2,1), (2,3), (2,5)]

fig = Figure(size = (1950, 1100))

Label(fig[0, 1:6],
      "Angle-based Fourier mode fractions   fₘ = Aₘ/(A₁+…+A₆) — $subfolder   (last $(round(Int,LAST_SECONDS))s)";
      fontsize = 14, tellwidth = false)

for (k, (row, col)) in enumerate(panel_positions)
    ax = Axis(fig[row, col];
        xlabel = "Active force  f₀",
        ylabel = "Bending stiffness  κ",
        title  = MODE_TITLES[k],
        xticks = LinearTicks(6),
        yticks = LinearTicks(6),
    )
    hm = heatmap!(ax, f0_vals, kappa_vals, matrices[k]';
                  colormap   = MODE_COLORS[k],
                  colorrange = (0.0, 1.0),
                  nan_color  = :lightgrey)
    Colorbar(fig[row, col+1], hm;
             label  = "fraction of total mode amplitude",
             width  = 16)
end

rowgap!(fig.layout, 1, 20)
for c in (1, 3, 5); colgap!(fig.layout, c, 8); end

out_path = joinpath(results_dir, "phase_diagram_mode_fractions.png")
save(out_path, fig, px_per_unit = 2)
println("Saved: $out_path")
