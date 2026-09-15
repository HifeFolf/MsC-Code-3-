

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT = joinpath(@__DIR__, "..")
const SKIP_SECONDS = 5.0    # discard this many video-seconds from the start (transient)
const FRAMERATE    = 60     # frames per video second (matches run config)
const N_BEADS      = 16
const NM1          = N_BEADS - 1   # = 15

subfolder  = length(ARGS) >= 1 ? ARGS[1] : "Results_01"
id_start   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan all runs ─────────────────────────────────────────────────────────────

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder (IDs $id_start – $id_end)…")

χ_raw = Float64[]
Φ_raw = Float64[]
b_raw = Float64[]

for rd in run_dirs
    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    κ == 0.0 && continue   # Φ undefined

    b_vec = jldopen(obs_file, "r") do f
        haskey(f, "b") ? f["b"] : nothing
    end
    b_vec === nothing && continue

    n           = length(b_vec)
    skip_frames = round(Int, SKIP_SECONDS * FRAMERATE)   # 5.0 s × 60 fps = 300 frames
    idx         = (skip_frames + 1):n
    length(idx) < 1 && continue
    b_ss = mean(b_vec[idx])

    χ = κ / NM1
    Φ = f₀ * NM1 / κ

    push!(χ_raw, χ)
    push!(Φ_raw, Φ)
    push!(b_raw, b_ss)
end

println("Loaded $(length(b_raw)) runs (κ=0 excluded).")

# ── Bin onto regular (χ, log Φ) grid ─────────────────────────────────────────

# χ axis: linear,  Φ axis: log (but Φ=0 for f₀=0 — put those in a separate row)
const N_CHI_BINS = 99     # one bin per unique κ value (100 κ values, κ=0 excluded → 99)
const N_PHI_BINS = 120    # log-spaced bins for Φ

# Separate f₀=0 runs (Φ=0) from active runs
active  = Φ_raw .> 0.0
χ_act   = χ_raw[active];   Φ_act = Φ_raw[active];   b_act = b_raw[active]
χ_pas   = χ_raw[.!active];  b_pas = b_raw[.!active]

# Grid edges
chi_edges = range(minimum(χ_raw), maximum(χ_raw), length = N_CHI_BINS + 1)
phi_edges = exp.(range(log(minimum(Φ_act)), log(maximum(Φ_act)),
                       length = N_PHI_BINS + 1))

chi_centers = [(chi_edges[i] + chi_edges[i+1]) / 2 for i in 1:N_CHI_BINS]
phi_centers = [sqrt(phi_edges[i] * phi_edges[i+1])  for i in 1:N_PHI_BINS]   # log midpoint

# Fill heatmap matrix: rows = Φ bin, cols = χ bin
b_matrix = fill(NaN, N_CHI_BINS, N_PHI_BINS)
counts   = zeros(Int, N_CHI_BINS, N_PHI_BINS)
b_accum  = zeros(N_CHI_BINS, N_PHI_BINS)

for k in eachindex(χ_act)
    ic = searchsortedlast(chi_edges, χ_act[k])
    ip = searchsortedlast(phi_edges, Φ_act[k])
    ic = clamp(ic, 1, N_CHI_BINS)
    ip = clamp(ip, 1, N_PHI_BINS)
    counts[ic, ip]  += 1
    b_accum[ic, ip] += b_act[k]
end

for ic in 1:N_CHI_BINS, ip in 1:N_PHI_BINS
    counts[ic, ip] > 0 && (b_matrix[ic, ip] = b_accum[ic, ip] / counts[ic, ip])
end

n_filled  = count(!isnan, b_matrix)
n_missing = count(isnan,  b_matrix)
@printf("Heatmap cells: %d filled,  %d empty\n", n_filled, n_missing)

# ── Figure ────────────────────────────────────────────────────────────────────

fig = Figure(size = (950, 800))

Label(fig[0, 1],
      "Phase diagram — Results_01   χ = κ/(N−1),   Φ = f₀(N−1)/κ   (N=16)";
      fontsize = 13, tellwidth = false)

ax = Axis(fig[1, 1];
    xlabel       = "Dimensionless stiffness   χ = κ / (N−1)",
    ylabel       = "Dimensionless activity   Φ = f₀(N−1) / κ",
    title        = "Mean asphericity  b  (skipping first $(round(Int, SKIP_SECONDS))s = $(round(Int, SKIP_SECONDS*FRAMERATE)) frames)",
    yscale       = log10,
    yticks       = LogTicks(WilkinsonTicks(6)),
    xticks       = LinearTicks(7),
)

b_finite = filter(!isnan, b_matrix)
cmax = isempty(b_finite) ? 1.0 : min(1.0, maximum(b_finite))

hm = heatmap!(ax, chi_centers, phi_centers, b_matrix;
              colormap   = :viridis,
              colorrange = (0.0, cmax),
              nan_color  = :white)

Colorbar(fig[1, 2], hm;
         label = "mean asphericity  b  (0 = circular,  1 = rod-like)",
         width = 18)

out_path = joinpath(results_dir, "phase_diagram_collapsed.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
