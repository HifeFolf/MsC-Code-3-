# phase_diagram_antipolar.jl
# Sweep f0 and κ for the anti-polar ring.
# 100 parameter points (10×10 grid over f0, κ ∈ [1, 25]).
# Each run is ~25 s of simulation; 150 snapshots are taken evenly.
# Every snapshot is classified individually into a regime.
# All classified snapshots (up to 15 000) are plotted as individual
# points on the (f0, κ) phase diagram, coloured by regime.
#
# Usage:  julia phase_diagram_antipolar.jl

using Printf
using CairoMakie
using Random

include(joinpath(@__DIR__, "..", "Simulation", "Engine_ring.jl"))
using .EngineRing

# ─── Gyration tensor observables ────────────────────────────────────────────
function gyration_vals(r::Matrix{Float64})
    N   = size(r, 2)
    xcm = sum(r[1, :]) / N
    ycm = sum(r[2, :]) / N
    Gxx = sum((r[1, i] - xcm)^2 for i in 1:N) / N
    Gyy = sum((r[2, i] - ycm)^2 for i in 1:N) / N
    Gxy = sum((r[1, i] - xcm) * (r[2, i] - ycm) for i in 1:N) / N
    mid  = (Gxx + Gyy) / 2.0
    half = sqrt(((Gxx - Gyy) / 2.0)^2 + Gxy^2)
    λ1   = mid + half
    λ2   = mid - half
    rg2  = λ1 + λ2
    b    = rg2 > 1e-30 ? (λ1 - λ2)^2 / rg2^2 : 0.0
    return rg2, λ1, λ2, b
end

# ─── Whole-ring periodic wrap ───────────────────────────────────────────────
function wrap_ring!(r::Matrix{Float64}, L::Float64)
    hw = L / 2.0
    min_x = minimum(r[1, :]); max_x = maximum(r[1, :])
    min_y = minimum(r[2, :]); max_y = maximum(r[2, :])
    min_x >  hw && (r[1, :] .-= L)
    max_x < -hw && (r[1, :] .+= L)
    min_y >  hw && (r[2, :] .-= L)
    max_y < -hw && (r[2, :] .+= L)
end

# ─── Regime classification (per snapshot) ───────────────────────────────────
# Codes: 1=strict curled, 2=broad curled,
#        3=strict tri,    4=broad tri,
#        5=strict circ,   6=broad circ,  0=none
function classify_frame(rg2, λ1, λ2, b)
    # Classification based on Rg² only

    # Strict curled
    2.4 <= rg2 <= 3.4 && return 1
    # Strict rounded-tri
    5.4 <= rg2 <= 6.3 && return 3
    # Strict circular
    6.2 <= rg2 <= 6.6 && return 5

    # Broad curled
    2.3 <= rg2 <= 3.6 && return 2
    # Broad rounded-tri
    5.2 <= rg2 <= 6.5 && return 4
    # Broad circular
    6.1 <= rg2 <= 6.7 && return 6

    return 0
end

# ─── Run one (f0, κ): ~25 s simulation, 150 evenly spaced snapshots ────────
function run_point(f0_val::Float64, κ_val::Float64;
                   n_samples::Int = 150,
                   total_steps::Int = 150_000)

    p = EngineRing.Params(f0 = f0_val, κ = κ_val)
    N = p.N
    l = p.l
    L = p.L

    bead_types = vcat(fill(:active, N ÷ 2), fill(:active_rev, N - N ÷ 2))

    R  = l / (2 * sin(π / N))
    r0 = zeros(2, N)
    for i in 1:N
        φ = 2π * (i - 1) / N
        r0[1, i] = R * cos(φ)
        r0[2, i] = R * sin(φ)
    end

    state = EngineRing.State(copy(r0), copy(bead_types))
    sample_every = total_steps ÷ n_samples

    # Collect the regime code for each snapshot
    codes = Int[]

    for s in 1:total_steps
        EngineRing.step!(state, p)
        if s % sample_every == 0
            wrap_ring!(state.r, L)
            rg2, λ1, λ2, b = gyration_vals(state.r)
            push!(codes, classify_frame(rg2, λ1, λ2, b))
        end
    end

    return codes
end

# ─── Main ───────────────────────────────────────────────────────────────────
function main()
    f0_vals = collect(range(1.0, 25.0, length=10))
    κ_vals  = collect(range(1.0, 25.0, length=10))
    n_points = length(f0_vals) * length(κ_vals)

    println("═══════════════════════════════════════════════════════")
    println("  10×10 grid over f0, κ ∈ [1, 25]  ($n_points runs)")
    println("  150 snapshots per run → up to $(n_points * 150) phase-map points")
    println("═══════════════════════════════════════════════════════")

    # Collect all classified snapshots: (f0, κ, code)
    all_f0   = Float64[]
    all_κ    = Float64[]
    all_code = Int[]

    count = 0
    for f0 in f0_vals, κ in κ_vals
        count += 1
        @printf("[%d/%d]  f0=%.2f  κ=%.2f  … ", count, n_points, f0, κ)
        codes = run_point(f0, κ)
        n_classified = sum(c > 0 for c in codes)
        @printf("%d / %d snapshots classified\n", n_classified, length(codes))

        for c in codes
            if c > 0
                push!(all_f0, f0)
                push!(all_κ, κ)
                push!(all_code, c)
            end
        end
    end

    println("\nTotal classified snapshots: $(length(all_code)) / $(n_points * 150)")

    # ── Plot ────────────────────────────────────────────────────────────────
    # Colour per code:
    #   1 = strict curled (dark blue)     2 = broad curled (light blue)
    #   3 = strict tri    (dark red)      4 = broad tri    (light red)
    #   5 = strict circ   (dark green)    6 = broad circ   (light green)
    code_color = Dict(
        1 => RGBAf(0.05, 0.20, 0.70, 0.5),
        2 => RGBAf(0.40, 0.60, 0.95, 0.5),
        3 => RGBAf(0.75, 0.05, 0.05, 0.5),
        4 => RGBAf(1.00, 0.45, 0.45, 0.5),
        5 => RGBAf(0.05, 0.55, 0.05, 0.5),
        6 => RGBAf(0.45, 0.80, 0.45, 0.5),
    )

    # Add jitter so overlapping points are visible
    df = f0_vals[2] - f0_vals[1]  # grid spacing
    dk = κ_vals[2]  - κ_vals[1]
    jitter_scale = 0.3  # fraction of grid spacing

    rng = MersenneTwister(42)
    jittered_f0 = all_f0 .+ jitter_scale .* df .* (rand(rng, length(all_f0)) .- 0.5)
    jittered_κ  = all_κ  .+ jitter_scale .* dk .* (rand(rng, length(all_κ))  .- 0.5)
    colors = [code_color[c] for c in all_code]

    fig = Figure(size = (800, 650))
    ax  = Axis(fig[1, 1];
        xlabel = "f₀  (propulsion force)",
        ylabel = "κ  (bending stiffness)",
        title  = "Anti-polar ring — per-snapshot phase diagram  (150 × 100 runs)",
    )

    scatter!(ax, jittered_f0, jittered_κ;
        color = colors, markersize = 4, strokewidth = 0)

    # Legend
    legend_entries = [
        MarkerElement(color = RGBf(0.05, 0.20, 0.70), marker = :circle, markersize = 10) => "Curled (strict)",
        MarkerElement(color = RGBf(0.40, 0.60, 0.95), marker = :circle, markersize = 10) => "Curled (broad)",
        MarkerElement(color = RGBf(0.75, 0.05, 0.05), marker = :circle, markersize = 10) => "Rounded-tri (strict)",
        MarkerElement(color = RGBf(1.00, 0.45, 0.45), marker = :circle, markersize = 10) => "Rounded-tri (broad)",
        MarkerElement(color = RGBf(0.05, 0.55, 0.05), marker = :circle, markersize = 10) => "Circular (strict)",
        MarkerElement(color = RGBf(0.45, 0.80, 0.45), marker = :circle, markersize = 10) => "Circular (broad)",
    ]
    Legend(fig[1, 2],
        [e[1] for e in legend_entries],
        [e[2] for e in legend_entries];
        framevisible = true, labelsize = 11)

    outpath = joinpath(@__DIR__, "phase_diagram_f0_1-25_k_1-25.png")
    save(outpath, fig; px_per_unit = 2)
    println("Saved: $outpath")
end

Base.invokelatest(main)
