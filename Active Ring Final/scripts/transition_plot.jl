# transition_plot.jl
# Plot regime fractions as a function of f₀/κ, using all existing sweep runs.
#
# For each run, the 150-snapshot vote fractions are computed and plotted
# against f₀/κ.  Runs are binned by f₀/κ to produce smooth curves showing
# how the mode percentages change across the phase boundaries.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/transition_plot.jl
#
# Output:  results/transition_curves.png

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_TP = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_TP, "src", "EngineRing.jl"))
using .EngineRing

include(joinpath(PROJECT_ROOT_TP, "src", "IOUtils.jl"))
using .IOUtils

# ── Snapshot classifier (same as phase_diagram.jl) ───────────────────────────

const CENTROIDS = Dict(
    :circular   => (0.004, 0.022, 0.017),
    :curled     => (0.062, 0.475, 0.306),
    :elongated  => (0.522, 0.293, 0.065),
    :triangular => (0.213, 0.180, 0.081),
)

const N_SNAPSHOTS = 150

function classify_snapshot(Rg, b, A2, A3)
    Rg < 2.0 && A2 > 0.25 && return :curled
    b < 0.05 && A2 < 0.08 && return :circular
    b > 0.25 && A2 > 0.15 && Rg > 2.0 && return :elongated
    0.05 < b < 0.30 && A3 > 0.04 && 0.08 < A2 < 0.25 && return :triangular
    best = :circular; best_d = Inf
    for (regime, (cb, cA2, cA3)) in CENTROIDS
        d = (b - cb)^2 + (A2 - cA2)^2 + (A3 - cA3)^2
        d < best_d && (best_d = d; best = regime)
    end
    return best
end

function snapshot_fractions(obs::Dict{String, Vector{Float64}})
    n       = length(obs["Rg"])
    indices = round.(Int, range(1, n, length = N_SNAPSHOTS))
    votes   = Dict(:circular => 0, :curled => 0, :elongated => 0, :triangular => 0)
    for i in indices
        r = classify_snapshot(obs["Rg"][i], obs["b"][i], obs["A2"][i], obs["A3"][i])
        votes[r] += 1
    end
    return Dict(k => v / N_SNAPSHOTS for (k, v) in votes)
end

# ── Load all sweep runs ──────────────────────────────────────────────────────

results_dir = joinpath(PROJECT_ROOT_TP, "results")
run_dirs    = filter(d -> (m = match(r"^run_(\d+)$", d); m !== nothing && 500 <= parse(Int, m[1]) <= 1100),
                     readdir(results_dir, join=false))
sort!(run_dirs)

# Per-run data
ratios     = Float64[]        # f0/κ for each run
frac_circ  = Float64[]
frac_curl  = Float64[]
frac_elong = Float64[]
frac_tri   = Float64[]

println("Loading $(length(run_dirs)) runs…\n")

for rd in run_dirs
    rp = joinpath(results_dir, rd)
    isfile(joinpath(rp, "observables.jld2")) || continue
    isfile(joinpath(rp, "params.toml"))      || continue

    p, _  = IOUtils.load_params(rp)
    _, obs = IOUtils.load_observables(rp)
    haskey(obs, "A2") || continue

    # Skip κ = 0 (ratio undefined) and very small κ
    p.κ < 0.5 && continue

    ratio = p.f0 / p.κ
    fracs = snapshot_fractions(obs)

    push!(ratios,     ratio)
    push!(frac_circ,  fracs[:circular])
    push!(frac_curl,  fracs[:curled])
    push!(frac_elong, fracs[:elongated])
    push!(frac_tri,   fracs[:triangular])
end

println("Loaded $(length(ratios)) runs with κ ≥ 0.5")

# ── Binning helper ────────────────────────────────────────────────────────────

function bin_fractions(ratios, fc, fcu, fe, ft; edges)
    nb = length(edges) - 1
    centers = [(edges[i] + edges[i+1]) / 2 for i in 1:nb]
    bc  = zeros(nb); bcu = zeros(nb); be = zeros(nb); bt = zeros(nb)
    cnt = zeros(Int, nb)
    edge_vec = collect(edges)
    for j in eachindex(ratios)
        bi = searchsortedlast(edge_vec, ratios[j])
        bi = clamp(bi, 1, nb)
        bc[bi]  += fc[j];  bcu[bi] += fcu[j]
        be[bi]  += fe[j];  bt[bi]  += ft[j]
        cnt[bi] += 1
    end
    for i in 1:nb
        cnt[i] > 0 && (bc[i] /= cnt[i]; bcu[i] /= cnt[i]; be[i] /= cnt[i]; bt[i] /= cnt[i])
    end
    m = cnt .> 0
    return centers[m], bc[m], bcu[m], be[m], bt[m]
end

order = sortperm(ratios)

# ── Two-panel figure ─────────────────────────────────────────────────────────

fig = Figure(size = (1400, 600))

# --- Panel A: Zoomed view f₀/κ ∈ [0, 5] with fine bins ---
ax1 = Axis(fig[1, 1];
    xlabel   = "f₀ / κ",
    ylabel   = "Fraction of snapshots",
    title    = "A)  Transition region  (f₀/κ = 0 – 5)",
    yticks   = 0:0.1:1.0,
)
ylims!(ax1, -0.02, 1.02)
xlims!(ax1, 0.0, 5.0)

zoom_mask = ratios .<= 5.0
zm_r  = ratios[zoom_mask]
zm_fc = frac_circ[zoom_mask]; zm_fcu = frac_curl[zoom_mask]
zm_fe = frac_elong[zoom_mask]; zm_ft = frac_tri[zoom_mask]

bc_z, yc_z, ycu_z, ye_z, yt_z = bin_fractions(zm_r, zm_fc, zm_fcu, zm_fe, zm_ft;
                                                edges = range(0.0, 5.0, length = 51))

# Scatter
zo = sortperm(zm_r)
scatter!(ax1, zm_r[zo], zm_fc[zo];  color = (:royalblue,   0.15), markersize = 5)
scatter!(ax1, zm_r[zo], zm_fcu[zo]; color = (:darkorange,  0.15), markersize = 5)
scatter!(ax1, zm_r[zo], zm_fe[zo];  color = (:crimson,     0.15), markersize = 5)
scatter!(ax1, zm_r[zo], zm_ft[zo];  color = (:forestgreen, 0.15), markersize = 5)

# Binned curves
lines!(ax1, bc_z, yc_z;  color = :royalblue,  linewidth = 3, label = "Circular")
lines!(ax1, bc_z, ycu_z; color = :darkorange, linewidth = 3, label = "Curled")
lines!(ax1, bc_z, ye_z;  color = :crimson,    linewidth = 3, label = "Elongated")
lines!(ax1, bc_z, yt_z;  color = :forestgreen, linewidth = 3, label = "Triangular")

vlines!(ax1, [0.7]; color = :gray40, linestyle = :dash, linewidth = 1.5, label = "Circ→Tri  ≈ 0.7")
vlines!(ax1, [1.4]; color = :gray60, linestyle = :dot,  linewidth = 1.5, label = "Tri→Elong ≈ 1.4")

axislegend(ax1; position = :rt)

# --- Panel B: Full range, log scale ---
ax2 = Axis(fig[1, 2];
    xlabel   = "f₀ / κ  (log scale)",
    ylabel   = "Fraction of snapshots",
    title    = "B)  Full range",
    xscale   = log10,
    yticks   = 0:0.1:1.0,
    xticks   = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
)
ylims!(ax2, -0.02, 1.02)

# Use all data, binned on log-spaced edges
log_edges = 10 .^ range(log10(max(minimum(ratios), 0.01)), log10(maximum(ratios)), length = 51)
bc_f, yc_f, ycu_f, ye_f, yt_f = bin_fractions(ratios, frac_circ, frac_curl, frac_elong, frac_tri;
                                                edges = log_edges)

scatter!(ax2, ratios[order], frac_circ[order];  color = (:royalblue,   0.12), markersize = 4)
scatter!(ax2, ratios[order], frac_curl[order];  color = (:darkorange,  0.12), markersize = 4)
scatter!(ax2, ratios[order], frac_elong[order]; color = (:crimson,     0.12), markersize = 4)
scatter!(ax2, ratios[order], frac_tri[order];   color = (:forestgreen, 0.12), markersize = 4)

lines!(ax2, bc_f, yc_f;  color = :royalblue,  linewidth = 3, label = "Circular")
lines!(ax2, bc_f, ycu_f; color = :darkorange, linewidth = 3, label = "Curled")
lines!(ax2, bc_f, ye_f;  color = :crimson,    linewidth = 3, label = "Elongated")
lines!(ax2, bc_f, yt_f;  color = :forestgreen, linewidth = 3, label = "Triangular")

vlines!(ax2, [0.7]; color = :gray40, linestyle = :dash, linewidth = 1.5)
vlines!(ax2, [1.4]; color = :gray60, linestyle = :dot,  linewidth = 1.5)

axislegend(ax2; position = :rc)

out_path = joinpath(results_dir, "transition_curves.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
