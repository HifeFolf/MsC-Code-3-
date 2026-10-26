# plot_phase_diagram.jl
# Phase diagram of a clamped active filament in the (f0, κ) plane.
#
# THRESHOLD STRATEGY:
#   σ_y: passive baseline → "straight" means σ_y ≤ 2× thermal noise at that κ
#   ACF peak & winding rate: DATA-DRIVEN — histogram the observable across the
#       full grid, find the valley between the two highest peaks → threshold
#   Sensitivity shown via ±30% perturbation of all thresholds
#
# Regimes:
#   0 = Straight         (indistinguishable from passive thermal fluctuations)
#   1 = Buckled static   (deflected beyond thermal, but no oscillation)
#   2 = Periodic beating  (clear ACF peak → limit cycle)
#   3 = Rotating          (continuous winding around clamp)
#   4 = Irregular         (active but disordered)
#
# Output:
#   Page 1: phase diagram + 3 continuous order-parameter maps
#   Page 2: observable histograms + passive baseline + threshold sensitivity

using GLMakie
using Statistics
using LinearAlgebra
using Serialization

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

# ═══════════════════════════════════════════════════════════════
#  Observable measurement functions
# ═══════════════════════════════════════════════════════════════

function measure_tip_amplitude(tip_y::Vector{Float64}, L::Float64)
    return std(tip_y) / L
end

function compute_acf(signal::Vector{Float64}, max_lag::Int)
    y = signal .- mean(signal)
    n = length(y)
    max_lag = min(max_lag, n ÷ 2)
    acf = zeros(max_lag + 1)
    for lag in 0:max_lag
        acf[lag+1] = sum(y[1:end-lag] .* y[1+lag:end])
    end
    if acf[1] < 1e-14
        return zeros(max_lag + 1)
    end
    acf ./= acf[1]
    return acf
end

function measure_beating_frequency(tip_y::Vector{Float64}, dt_save::Float64)
    max_lag = min(length(tip_y) ÷ 2, 2000)
    if max_lag < 10
        return (NaN, 0.0)
    end
    acf = compute_acf(tip_y, max_lag)
    zc = findfirst(acf .< 0)
    if zc === nothing || zc <= 2
        return (NaN, 0.0)
    end
    search_end = min(length(acf), zc + max_lag ÷ 2)
    if search_end <= zc
        return (NaN, 0.0)
    end
    peak_offset = argmax(acf[zc:search_end])
    peak = zc + peak_offset - 1
    acf_peak_height = acf[peak]
    if acf_peak_height < 0.05
        return (NaN, acf_peak_height)
    end
    period = peak * dt_save
    ω = 2π / period
    return (ω, acf_peak_height)
end

function measure_winding_rate(tip_x::Vector{Float64}, tip_y::Vector{Float64},
                              dt_save::Float64)
    n = length(tip_x)
    if n < 2
        return 0.0
    end
    cumulative = 0.0
    for i in 2:n
        dθ = atan(tip_y[i], tip_x[i]) - atan(tip_y[i-1], tip_x[i-1])
        if dθ > π;      dθ -= 2π
        elseif dθ < -π; dθ += 2π
        end
        cumulative += dθ
    end
    total_time = (n - 1) * dt_save
    return cumulative / total_time
end

# ═══════════════════════════════════════════════════════════════
#  Data-driven threshold: find valley between two histogram peaks
# ═══════════════════════════════════════════════════════════════

"""
    find_valley_threshold(data; nbins=50, fallback=NaN) → (threshold, peaks, valley_idx)

Histogram `data`, smooth, find the two highest peaks, return the bin center
at the minimum between them (the valley).  If no bimodality is detected,
returns `fallback`.
"""
function find_valley_threshold(data::Vector{Float64};
        nbins::Int=50, fallback::Float64=NaN)

    isempty(data) && return (fallback, Float64[], 0)

    lo, hi = extrema(data)
    if lo ≈ hi
        return (fallback, Float64[], 0)
    end

    edges = range(lo, hi, length=nbins+1)
    centers = [(edges[i] + edges[i+1]) / 2 for i in 1:nbins]
    counts = zeros(Int, nbins)
    for v in data
        idx = clamp(searchsortedlast(collect(edges), v), 1, nbins)
        counts[idx] += 1
    end

    # smooth with a 3-bin moving average to suppress noise
    smoothed = zeros(Float64, nbins)
    for i in 1:nbins
        lo_i = max(1, i-1)
        hi_i = min(nbins, i+1)
        smoothed[i] = mean(counts[lo_i:hi_i])
    end

    # find all local maxima
    peak_idxs = Int[]
    for i in 2:nbins-1
        if smoothed[i] > smoothed[i-1] && smoothed[i] >= smoothed[i+1]
            push!(peak_idxs, i)
        end
    end
    # also check endpoints
    if nbins >= 2 && smoothed[1] >= smoothed[2]
        pushfirst!(peak_idxs, 1)
    end
    if nbins >= 2 && smoothed[end] >= smoothed[end-1]
        push!(peak_idxs, nbins)
    end

    if length(peak_idxs) < 2
        return (fallback, centers[peak_idxs], 0)
    end

    # sort peaks by height, take two highest
    sorted = sort(peak_idxs, by=i -> smoothed[i], rev=true)
    p1, p2 = minmax(sorted[1], sorted[2])

    # find valley = minimum of smoothed counts between the two peaks
    valley_region = p1:p2
    valley_offset = argmin(smoothed[valley_region])
    valley_idx = p1 + valley_offset - 1
    threshold = centers[valley_idx]

    return (threshold, centers, valley_idx)
end

# ═══════════════════════════════════════════════════════════════
#  Classification (thresholds are arguments, not hardcoded)
# ═══════════════════════════════════════════════════════════════

function classify_regime(σ_y_norm::Float64, ω::Float64,
                         acf_peak::Float64, winding_rate::Float64,
                         σ_thresh::Float64, acf_thresh::Float64,
                         winding_thresh::Float64)
    if σ_y_norm < σ_thresh
        return 0  # STRAIGHT
    end
    if abs(winding_rate) > winding_thresh
        return 3  # ROTATING
    end
    if !isnan(ω) && acf_peak > acf_thresh
        return 2  # PERIODIC
    end
    if isnan(ω) && acf_peak < 0.1
        return 1  # BUCKLED STATIC
    end
    return 4      # IRREGULAR
end

function classify_all!(regimes::Matrix{Int},
                       sigma_y::Matrix{Float64}, omega::Matrix{Float64},
                       acf_peak::Matrix{Float64}, winding::Matrix{Float64},
                       σ_baselines::Vector{Float64},
                       σ_mult::Float64, acf_thresh::Float64,
                       winding_thresh::Float64)
    nκ, nf = size(regimes)
    for j in 1:nκ
        σ_thresh = σ_mult * σ_baselines[j]
        for i in 1:nf
            regimes[j, i] = classify_regime(
                sigma_y[j, i], omega[j, i],
                acf_peak[j, i], winding[j, i],
                σ_thresh, acf_thresh, winding_thresh)
        end
    end
end

# ═══════════════════════════════════════════════════════════════
#  Simulation
# ═══════════════════════════════════════════════════════════════

function adaptive_sim_time(f0::Float64; base::Float64=300.0)
    if f0 < 0.5;     return base * 3.0
    elseif f0 < 2.0;  return base * 2.0
    else;              return base
    end
end

function run_single_point(f0::Float64, κ::Float64;
        N::Int=20, sim_time::Float64=300.0,
        substeps::Int=200, burn_frac::Float64=0.3)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l
    L = (N - 1) * l
    dt_save = substeps * p.dt
    frames  = round(Int, sim_time / dt_save)

    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    clamp1 = [0.0, 0.0]
    clamp2 = [l,   0.0]

    tip_x = Vector{Float64}(undef, frames)
    tip_y = Vector{Float64}(undef, frames)

    for f in 1:frames
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            state.r[:, 1] .= clamp1
            state.r[:, 2] .= clamp2
        end
        tip_x[f] = state.r[1, N]
        tip_y[f] = state.r[2, N]
    end

    burn = round(Int, burn_frac * frames)
    tx = tip_x[burn+1:end]
    ty = tip_y[burn+1:end]

    σ_y = measure_tip_amplitude(ty, L)
    ω, acf_pk = measure_beating_frequency(ty, dt_save)
    winding = measure_winding_rate(tx, ty, dt_save)

    # return observables + raw data for diagnostics
    return (sigma_y=σ_y, omega=ω, acf_peak=acf_pk, winding_rate=winding,
            tip_x=tx, tip_y=ty, dt_save=dt_save,
            final_r=copy(state.r), L=L)
end

# ═══════════════════════════════════════════════════════════════
#  Per-run diagnostic plot (saved to folder for visual inspection)
# ═══════════════════════════════════════════════════════════════

function save_run_diagnostic(result, f0::Float64, κ::Float64, outdir::String)
    fig = Figure(size=(900, 400))

    # Panel 1: chain shape (final frame)
    r = result.final_r
    N = size(r, 2)
    ax1 = Axis(fig[1, 1];
        xlabel = "x", ylabel = "y",
        title  = "f0=$(round(f0, digits=3))  κ=$(round(κ, digits=2))" *
                 "  σ=$(round(result.sigma_y, digits=4))" *
                 "  acf=$(round(result.acf_peak, digits=3))" *
                 "  |w|=$(round(abs(result.winding_rate), digits=3))",
        titlesize = 11,
        aspect = DataAspect(),
    )
    lines!(ax1, r[1, :], r[2, :]; linewidth = 2, color = :black)
    scatter!(ax1, [r[1, 1]], [r[2, 1]]; markersize = 10, color = :red,
             label = "clamp (bead 1)")
    scatter!(ax1, [r[1, N]], [r[2, N]]; markersize = 8, color = :dodgerblue,
             label = "free tip (bead $N)")
    axislegend(ax1; position = :lt, labelsize = 9)

    # Panel 2: tip_y(t) time series
    n = length(result.tip_y)
    times = (0:n-1) .* result.dt_save
    ax2 = Axis(fig[1, 2];
        xlabel = "time", ylabel = "tip y / L",
        title  = "Tip trajectory (post burn-in)",
        titlesize = 11,
    )
    lines!(ax2, times, result.tip_y ./ result.L;
        linewidth = 0.5, color = :black)

    mkpath(outdir)
    fname = "f0_$(round(f0, digits=3))_kappa_$(round(κ, digits=2)).png"
    save(joinpath(outdir, fname), fig; px_per_unit=2)
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  Passive baseline: run f0=0 at each κ to get thermal σ_y
# ═══════════════════════════════════════════════════════════════

function run_passive_baselines(κ_values::Vector{Float64};
        N::Int=20, sim_time::Float64=300.0,
        substeps::Int=200, burn_frac::Float64=0.3)
    println("\n  PHASE 0: Passive baselines (f0 = 0)")
    σ_baselines = Float64[]
    for (j, κ) in enumerate(κ_values)
        print("    [$j/$(length(κ_values))]  κ=$(round(κ, digits=2)) ... ")
        result = run_single_point(0.0, κ;
            N=N, sim_time=sim_time, substeps=substeps, burn_frac=burn_frac)
        push!(σ_baselines, result.sigma_y)
        println("σ_y/L = $(round(result.sigma_y, digits=5))")
    end
    return σ_baselines
end

# ═══════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ═══════════════════════════════════════════════════════════════

function save_checkpoint(path, f0_vals, κ_vals,
                         sigma_y, omega, acf_peak, winding, σ_baselines)
    serialize(path, Dict(
        "f0_values"    => f0_vals,
        "kappa_values" => κ_vals,
        "sigma_y"      => sigma_y,
        "omega"        => omega,
        "acf_peak"     => acf_peak,
        "winding"      => winding,
        "baselines"    => σ_baselines,
    ))
end

# ═══════════════════════════════════════════════════════════════
#  Phase diagram sweep (stores raw observables, no classification)
# ═══════════════════════════════════════════════════════════════

const REGIME_NAMES = ["STRAIGHT", "BUCKLED", "PERIODIC", "ROTATING", "IRREGULAR"]

function run_phase_diagram(f0_values::Vector{Float64}, κ_values::Vector{Float64},
        σ_baselines::Vector{Float64};
        N::Int=20, substeps::Int=200, burn_frac::Float64=0.3,
        checkpoint_path::String="",
        diagnostics_dir::String="")

    nf = length(f0_values)
    nκ = length(κ_values)
    total = nf * nκ

    # raw observable matrices [i_κ, i_f0]
    sigma_y   = fill(NaN, nκ, nf)
    omega     = fill(NaN, nκ, nf)
    acf_peak  = fill(NaN, nκ, nf)
    winding   = fill(NaN, nκ, nf)
    computed  = fill(false, nκ, nf)

    # load checkpoint
    if !isempty(checkpoint_path) && isfile(checkpoint_path)
        println("  Loading checkpoint: ", checkpoint_path)
        cp = deserialize(checkpoint_path)
        sigma_y   = cp["sigma_y"]
        omega     = cp["omega"]
        acf_peak  = cp["acf_peak"]
        winding   = cp["winding"]
        computed  = .!isnan.(sigma_y)
    end

    count = 0
    for (j, κ) in enumerate(κ_values)
        for (i, f0) in enumerate(f0_values)
            count += 1
            if computed[j, i]
                println("  [$count/$total]  κ=$(round(κ, digits=2)), " *
                        "f0=$(round(f0, digits=3))  (cached)")
                continue
            end

            sim_time = adaptive_sim_time(f0)
            print("  [$count/$total]  κ=$(round(κ, digits=2)), " *
                  "f0=$(round(f0, digits=3)),  T=$(round(Int, sim_time)) ... ")

            result = run_single_point(f0, κ;
                N=N, sim_time=sim_time, substeps=substeps, burn_frac=burn_frac)

            sigma_y[j, i]  = result.sigma_y
            omega[j, i]    = result.omega
            acf_peak[j, i] = result.acf_peak
            winding[j, i]  = result.winding_rate

            println("σ_y=$(round(result.sigma_y, digits=4))" *
                    "  acf=$(round(result.acf_peak, digits=3))" *
                    "  |w|=$(round(abs(result.winding_rate), digits=3))")

            # save per-run diagnostic plot
            if !isempty(diagnostics_dir)
                save_run_diagnostic(result, f0, κ, diagnostics_dir)
            end

            if !isempty(checkpoint_path) && count % 10 == 0
                save_checkpoint(checkpoint_path, f0_values, κ_values,
                    sigma_y, omega, acf_peak, winding, σ_baselines)
                println("    [checkpoint saved]")
            end
        end
    end

    if !isempty(checkpoint_path)
        save_checkpoint(checkpoint_path, f0_values, κ_values,
            sigma_y, omega, acf_peak, winding, σ_baselines)
    end

    return (sigma_y=sigma_y, omega=omega, acf_peak=acf_peak, winding=winding)
end

# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: Phase diagram + continuous order parameters
# ═══════════════════════════════════════════════════════════════

function plot_main_diagram(f0_values, κ_values, regimes, raw)

    fig = Figure(size=(1400, 1100))

    regime_colors = [:gray80, :steelblue, :forestgreen, :darkorange, :crimson]
    regime_labels = ["Straight", "Buckled", "Periodic", "Rotating", "Irregular"]

    all_f0 = Float64[]; all_κ = Float64[]
    for κ in κ_values, f0 in f0_values
        push!(all_f0, f0); push!(all_κ, κ)
    end

    # ── Panel 1: Regime map ──────────────────────────────────
    ax1 = Axis(fig[1, 1];
        xlabel = "f₀  (active force per bond)",
        ylabel = "κ  (bending stiffness)",
        title  = "Phase diagram  (N=20, kBT=1)",
        xscale = log10, yscale = log10,
    )
    for (j, κ) in enumerate(κ_values)
        for (i, f0) in enumerate(f0_values)
            r = regimes[j, i]
            if r >= 0
                scatter!(ax1, [f0], [κ];
                    color = regime_colors[r+1],
                    markersize = 20, marker = :rect)
            end
        end
    end
    for (k, label) in enumerate(regime_labels)
        scatter!(ax1, [NaN], [NaN];
            color = regime_colors[k], markersize = 12,
            marker = :rect, label = label)
    end
    axislegend(ax1; position = :rb)

    # ── Panel 2: σ_y / L  (with passive baseline overlay) ───
    ax2 = Axis(fig[1, 2];
        xlabel = "f₀", ylabel = "κ",
        title  = "σ_y / L   (+ passive 2× baseline line)",
        xscale = log10, yscale = log10,
    )
    σ_flat = [raw.sigma_y[j, i]
              for (j, _) in enumerate(κ_values)
              for (i, _) in enumerate(f0_values)]
    σ_clean = replace(σ_flat, NaN => 0.0)
    sc2 = scatter!(ax2, all_f0, all_κ;
        color = σ_clean, colormap = :viridis,
        markersize = 20, marker = :rect)
    Colorbar(fig[1, 3], sc2; label = "σ_y / L")

    # ── Panel 3: ACF peak ───────────────────────────────────
    ax3 = Axis(fig[2, 1];
        xlabel = "f₀", ylabel = "κ",
        title  = "ACF peak  (periodicity measure)",
        xscale = log10, yscale = log10,
    )
    acf_flat = [raw.acf_peak[j, i]
                for (j, _) in enumerate(κ_values)
                for (i, _) in enumerate(f0_values)]
    acf_clean = replace(acf_flat, NaN => 0.0)
    sc3 = scatter!(ax3, all_f0, all_κ;
        color = acf_clean, colormap = :plasma,
        markersize = 20, marker = :rect)
    Colorbar(fig[2, 2], sc3; label = "ACF peak")

    # ── Panel 4: |winding rate| ─────────────────────────────
    ax4 = Axis(fig[2, 3];
        xlabel = "f₀", ylabel = "κ",
        title  = "Winding rate  |dθ/dt|",
        xscale = log10, yscale = log10,
    )
    wind_flat = [abs(raw.winding[j, i])
                 for (j, _) in enumerate(κ_values)
                 for (i, _) in enumerate(f0_values)]
    wind_clean = replace(wind_flat, NaN => 0.0)
    sc4 = scatter!(ax4, all_f0, all_κ;
        color = wind_clean, colormap = :inferno,
        markersize = 20, marker = :rect)
    Colorbar(fig[2, 4], sc4; label = "|winding rate|")

    return fig
end

# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: Threshold justification
#    Row 1: observable histograms (bimodality check)
#    Row 2: passive baseline + threshold sensitivity
# ═══════════════════════════════════════════════════════════════

function plot_threshold_justification(f0_values, κ_values, raw, σ_baselines,
                                      σ_mult, acf_thresh, winding_thresh)
    nκ = length(κ_values)
    nf = length(f0_values)

    fig = Figure(size=(1600, 1100))

    Label(fig[0, 1:3], "Threshold justification",
          fontsize = 22, tellwidth = false)

    # ── Collect all observables as flat vectors ──────────────
    all_σ  = filter(!isnan, vec(raw.sigma_y))
    all_acf = filter(!isnan, vec(raw.acf_peak))
    all_w   = filter(!isnan, abs.(vec(raw.winding)))

    # also compute σ_y / baseline_σ ratio for each point
    σ_ratio = Float64[]
    for j in 1:nκ
        bl = σ_baselines[j]
        if bl < 1e-10; bl = 1e-10; end
        for i in 1:nf
            v = raw.sigma_y[j, i]
            if !isnan(v)
                push!(σ_ratio, v / bl)
            end
        end
    end

    # ── Row 1, Col 1: histogram of σ_y / baseline ───────────
    ax1 = Axis(fig[1, 1];
        xlabel = "σ_y / σ_y^passive",
        ylabel = "count",
        title  = "Tip amplitude ratio",
    )
    hist!(ax1, σ_ratio; bins = 40, color = (:dodgerblue, 0.6))
    vlines!(ax1, [σ_mult]; color = :red, linewidth = 2,
            linestyle = :dash, label = "$(σ_mult)× passive (physical)")
    axislegend(ax1; position = :rt)

    # ── Row 1, Col 2: histogram of ACF peak ──────────────────
    ax2 = Axis(fig[1, 2];
        xlabel = "ACF peak height",
        ylabel = "count",
        title  = "ACF peak  —  valley detection",
    )
    hist!(ax2, all_acf; bins = 40, color = (:forestgreen, 0.6))
    vlines!(ax2, [acf_thresh]; color = :red, linewidth = 2,
            linestyle = :dash,
            label = "valley = $(round(acf_thresh, digits=3))")
    axislegend(ax2; position = :rt)

    # ── Row 1, Col 3: histogram of |winding rate| ───────────
    ax3 = Axis(fig[1, 3];
        xlabel = "|winding rate|  (rad/time)",
        ylabel = "count",
        title  = "Winding rate  —  valley detection",
    )
    hist!(ax3, all_w; bins = 40, color = (:darkorange, 0.6))
    vlines!(ax3, [winding_thresh]; color = :red, linewidth = 2,
            linestyle = :dash,
            label = "valley = $(round(winding_thresh, digits=3))")
    axislegend(ax3; position = :rt)

    # ── Row 2, Col 1: passive baseline σ_y(κ) ───────────────
    ax4 = Axis(fig[2, 1];
        xlabel = "κ  (bending stiffness)",
        ylabel = "σ_y / L",
        title  = "Passive baseline (f0=0)  +  $(σ_mult)× threshold",
        xscale = log10, yscale = log10,
    )
    scatter!(ax4, κ_values, σ_baselines;
        markersize = 10, color = :black, label = "passive σ_y/L")
    lines!(ax4, κ_values, σ_mult .* σ_baselines;
        color = :red, linewidth = 2, linestyle = :dash,
        label = "$(σ_mult)× passive (\"straight\" boundary)")
    # overlay theory: for a clamped WLC, σ_y ~ sqrt(kBT L / κ) / L = 1/sqrt(κ L)
    N = 20; L = N - 1
    κ_fine = 10.0 .^ range(log10(κ_values[1]), log10(κ_values[end]), length=100)
    σ_theory = [1.0 / sqrt(κ * L) / L for κ in κ_fine]
    lines!(ax4, κ_fine, σ_theory;
        color = :gray50, linewidth = 1.5, linestyle = :dot,
        label = "theory ~ 1/√(κL)/L")
    axislegend(ax4; position = :rt)

    # ── Row 2, Cols 2–3: Threshold sensitivity ──────────────
    # Classify at nominal, ±30% thresholds, show all three regime maps
    multipliers = [0.7, 1.0, 1.3]
    titles = ["-30% thresholds", "Nominal thresholds", "+30% thresholds"]

    regime_colors = [:gray80, :steelblue, :forestgreen, :darkorange, :crimson]
    regime_labels = ["Straight", "Buckled", "Periodic", "Rotating", "Irregular"]

    for (k, mult) in enumerate(multipliers)
        ax = Axis(fig[2, k+1];
            xlabel = "f₀", ylabel = "κ",
            title  = titles[k],
            xscale = log10, yscale = log10,
        )

        perturbed_σ_mult = σ_mult * mult
        perturbed_acf    = acf_thresh * mult
        perturbed_wind   = winding_thresh * mult

        regimes_tmp = fill(-1, nκ, nf)
        classify_all!(regimes_tmp, raw.sigma_y, raw.omega,
                      raw.acf_peak, raw.winding, σ_baselines,
                      perturbed_σ_mult, perturbed_acf, perturbed_wind)

        for (j, κ) in enumerate(κ_values)
            for (i, f0) in enumerate(f0_values)
                r = regimes_tmp[j, i]
                if r >= 0
                    scatter!(ax, [f0], [κ];
                        color = regime_colors[r+1],
                        markersize = 16, marker = :rect)
                end
            end
        end

        if k == 2  # legend only on middle panel
            for (m, label) in enumerate(regime_labels)
                scatter!(ax, [NaN], [NaN];
                    color = regime_colors[m], markersize = 10,
                    marker = :rect, label = label)
            end
            axislegend(ax; position = :rb, labelsize = 10)
        end
    end

    return fig
end

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

function main()
    N = 20

    # f0 range: from below buckling threshold to strongly active
    f0_values = collect(10.0 .^ range(log10(0.1), log10(200.0), length=12))
    # κ range: soft to stiff
    κ_values  = collect(10.0 .^ range(log10(2.0), log10(200.0), length=10))

    checkpoint_path = joinpath(@__DIR__, "phase_diagram_checkpoint.jls")

    println("=" ^ 70)
    println("  PHASE DIAGRAM: CLAMPED ACTIVE FILAMENT")
    println("  N=$N,  kBT=1,  grid: $(length(f0_values)) f0 × $(length(κ_values)) κ" *
            " = $(length(f0_values) * length(κ_values)) simulations")
    println("  f0 range: $(round(f0_values[1], digits=3)) → $(round(f0_values[end], digits=1))")
    println("  κ  range: $(round(κ_values[1], digits=2)) → $(round(κ_values[end], digits=1))")
    println("=" ^ 70)

    # ── Phase 0: Passive baselines ───────────────────────────
    σ_baselines = run_passive_baselines(κ_values; N=N)

    println("\n  Passive baselines:")
    for (j, κ) in enumerate(κ_values)
        println("    κ=$(round(κ, digits=2))  →  σ_y/L = $(round(σ_baselines[j], digits=5))")
    end

    # ── Phase 1: Active grid ─────────────────────────────────
    println("\n  PHASE 1: Active simulations")
    diagnostics_dir = joinpath(@__DIR__, "phase_diagram_runs")
    raw = run_phase_diagram(f0_values, κ_values, σ_baselines;
        N=N, checkpoint_path=checkpoint_path,
        diagnostics_dir=diagnostics_dir)

    # ── Phase 2: Data-driven thresholds ────────────────────────
    σ_mult = 2.0   # σ_y: physically grounded at 2× passive baseline

    # ACF peak: find valley in distribution
    all_acf = filter(!isnan, vec(raw.acf_peak))
    acf_thresh_auto, _, _ = find_valley_threshold(all_acf; nbins=50, fallback=0.3)
    acf_thresh = isnan(acf_thresh_auto) ? 0.3 : acf_thresh_auto

    # |winding rate|: find valley in distribution
    all_w = filter(!isnan, abs.(vec(raw.winding)))
    wind_thresh_auto, _, _ = find_valley_threshold(all_w; nbins=50, fallback=0.5)
    winding_thresh = isnan(wind_thresh_auto) ? 0.5 : wind_thresh_auto

    println("\n  DATA-DRIVEN THRESHOLDS:")
    println("    σ_y threshold: $(σ_mult) × passive baseline (per-κ, physically grounded)")
    println("    ACF peak:      $(round(acf_thresh, digits=3))  (valley in distribution)")
    println("    |winding rate|: $(round(winding_thresh, digits=3))  (valley in distribution)")

    regimes = fill(-1, length(κ_values), length(f0_values))
    classify_all!(regimes, raw.sigma_y, raw.omega,
                  raw.acf_peak, raw.winding, σ_baselines,
                  σ_mult, acf_thresh, winding_thresh)

    println("\n  REGIME SUMMARY:")
    for k in 0:4
        n = count(regimes .== k)
        println("    $(REGIME_NAMES[k+1]): $n points")
    end

    # ── Figure 1: Phase diagram ──────────────────────────────
    fig1 = plot_main_diagram(f0_values, κ_values, regimes, raw)
    display(fig1)
    path1 = joinpath(@__DIR__, "phase_diagram.png")
    save(path1, fig1; px_per_unit=2)
    println("\nSaved diagram: ", path1)

    # ── Figure 2: Threshold justification ────────────────────
    fig2 = plot_threshold_justification(f0_values, κ_values, raw, σ_baselines,
                                        σ_mult, acf_thresh, winding_thresh)
    display(fig2)
    path2 = joinpath(@__DIR__, "phase_diagram_thresholds.png")
    save(path2, fig2; px_per_unit=2)
    println("Saved justification: ", path2)
end

main()
