# phase_diagram.jl
# Classify sweep runs into shape regimes and plot a (f0, κ) phase diagram.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/phase_diagram.jl
#
# Output:  results/phase_diagram.png

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_PD = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_PD, "src", "EngineRing.jl"))
using .EngineRing

include(joinpath(PROJECT_ROOT_PD, "src", "IOUtils.jl"))
using .IOUtils

# ── Regime classification criteria ───────────────────────────────────────────
#
# Derived from reference runs (steady-state means ± generous margins):
#
#   Circular   (run 505):  b ≈ 0.004,  A2 ≈ 0.02,  A3 ≈ 0.02,  Rg ≈ 2.56
#   Curled     (runs 515, 529): b ≈ 0.06,  A2 ≈ 0.48,  Rg ≈ 1.53
#   Elongated  (runs 530, 545): b ≈ 0.52,  A2 ≈ 0.29,  A3 < 0.10
#   Triangular (run 540):  b ≈ 0.21,  A2 ≈ 0.18,  A3 ≈ 0.08,  Rg ≈ 2.49

const N_SNAPSHOTS = 150   # number of evenly-spaced snapshots to classify

# Reference centroids for nearest-centroid fallback
const CENTROIDS = Dict(
    :circular   => (0.004, 0.022, 0.017),
    :curled     => (0.062, 0.475, 0.306),
    :elongated  => (0.522, 0.293, 0.065),
    :triangular => (0.213, 0.180, 0.081),
)

"""
Classify a single snapshot by its instantaneous observable values.
Returns one of :circular, :curled, :elongated, :triangular.
"""
function classify_snapshot(Rg, b, A2, A3)
    # 1. Curled: compact ring with elevated contour mode-2
    Rg < 2.0 && A2 > 0.25 && return :curled

    # 2. Circular: nearly round, minimal deformation
    b < 0.05 && A2 < 0.08 && return :circular

    # 3. Elongated: high asphericity, extended ring
    b > 0.25 && A2 > 0.15 && Rg > 2.0 && return :elongated

    # 4. Triangular: moderate asphericity with mode-3 activity
    0.05 < b < 0.30 && A3 > 0.04 && 0.08 < A2 < 0.25 && return :triangular

    # Fallback: nearest centroid in (b, A2, A3) space
    best = :circular
    best_dist = Inf
    for (regime, (cb, cA2, cA3)) in CENTROIDS
        d = (b - cb)^2 + (A2 - cA2)^2 + (A3 - cA3)^2
        if d < best_dist
            best_dist = d
            best      = regime
        end
    end
    return best
end

"""
Classify a full run by majority vote over N_SNAPSHOTS evenly-spaced frames.
Returns (dominant_regime, vote_counts::Dict).
"""
function classify_regime(obs::Dict{String, Vector{Float64}})
    n       = length(obs["Rg"])
    indices = round.(Int, range(1, n, length = N_SNAPSHOTS))

    votes = Dict(:circular => 0, :curled => 0, :elongated => 0, :triangular => 0)

    for i in indices
        regime = classify_snapshot(obs["Rg"][i], obs["b"][i], obs["A2"][i], obs["A3"][i])
        votes[regime] = get(votes, regime, 0) + 1
    end

    dominant = argmax(votes)
    return dominant, votes
end

# ── Scan all sweep runs ──────────────────────────────────────────────────────
results_dir = joinpath(PROJECT_ROOT_PD, "results")
run_dirs    = filter(d -> (m = match(r"^run_(\d+)$", d); m !== nothing && 500 <= parse(Int, m[1]) <= 1100),
                     readdir(results_dir, join=false))
sort!(run_dirs)

f0_vals    = Float64[]
kappa_vals = Float64[]
regimes    = Symbol[]

println("Classifying $(length(run_dirs)) runs…\n")

for rd in run_dirs
    run_path = joinpath(results_dir, rd)

    # Skip incomplete runs
    isfile(joinpath(run_path, "observables.jld2")) || continue
    isfile(joinpath(run_path, "params.toml"))      || continue

    p, _  = IOUtils.load_params(run_path)
    _, obs = IOUtils.load_observables(run_path)

    # Need A2, A3, Kloc
    haskey(obs, "A2") || continue

    regime, votes = classify_regime(obs)

    push!(f0_vals,    p.f0)
    push!(kappa_vals, p.κ)
    push!(regimes,    regime)

    n = length(obs["Rg"])
    @printf("  %-10s  f0=%5.1f  κ=%5.1f   C=%3d  Cu=%3d  E=%3d  T=%3d\n",
            regime, p.f0, p.κ,
            votes[:circular], votes[:curled], votes[:elongated], votes[:triangular])
end

# ── Plot phase diagram ──────────────────────────────────────────────────────
regime_style = Dict(
    :circular    => (color = :royalblue,    marker = :circle,   label = "Circular"),
    :curled      => (color = :darkorange,   marker = :star5,    label = "Curled"),
    :elongated   => (color = :crimson,      marker = :rect,     label = "Elongated"),
    :triangular  => (color = :forestgreen,  marker = :utriangle, label = "Triangular"),
)

fig = Figure(size = (800, 650))
ax  = Axis(fig[1, 1];
    xlabel   = "Active force  f₀",
    ylabel   = "Bending stiffness  κ",
    title    = "Phase diagram  (N=16,  kBT=0.1)",
    aspect   = DataAspect(),
)

# Plot each regime
for regime in [:circular, :curled, :elongated, :triangular]
    mask = regimes .== regime
    any(mask) || continue
    s = regime_style[regime]
    scatter!(ax, f0_vals[mask], kappa_vals[mask];
             color      = s.color,
             marker     = s.marker,
             markersize = 18,
             label      = s.label)
end

# Handle any unclassified points
mask_unc = regimes .== :unclassified
if any(mask_unc)
    scatter!(ax, f0_vals[mask_unc], kappa_vals[mask_unc];
             color = :gray50, marker = :diamond, markersize = 16,
             label = "Unclassified")
end

axislegend(ax; position = :rt)

out_path = joinpath(results_dir, "phase_diagram.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
