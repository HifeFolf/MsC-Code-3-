# plot_beating_frequency.jl
# Test the Isele-Holder / Chelakkot scaling law:  ω_b ∝ f0^(4/3)
#
# At fixed N and κ (i.e. fixed bending stiffness and drag), sweep f0
# and measure the beating frequency of the clamped filament from the
# autocorrelation of the tip y-coordinate.
#
# Expected: log-log slope of 4/3 ≈ 1.333
#
# Note on time scales (Isele-Holder eqn 17):
#   τ = (2/3π)⁴ · γ_l · L⁴ / κ
# In Brownian units (μ=1 → γ=1, l=1): τ ≈ 0.002 · L⁴/κ
# For N=20, κ=19: τ ≈ 0.002 · 19⁴/19 ≈ 14.6
#
# The beating period scales as T ∝ f0^(-4/3), so low f0 → long periods.
# Simulations must be long enough to capture several full beats.

using GLMakie
using Statistics

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

"""
    measure_beating_frequency(tip_y, dt_save)

Extract beating frequency from tip y-coordinate via autocorrelation.
Returns ω in rad/time, or NaN if no clear oscillation is found.
"""
function measure_beating_frequency(tip_y::Vector{Float64}, dt_save::Float64)
    n = length(tip_y)
    y = tip_y .- mean(tip_y)

    max_lag = min(n ÷ 2, 2000)
    if max_lag < 10
        return NaN
    end

    # autocorrelation
    acf = zeros(max_lag + 1)
    for lag in 0:max_lag
        acf[lag+1] = sum(y[1:end-lag] .* y[1+lag:end])
    end
    if acf[1] < 1e-14
        return NaN
    end
    acf ./= acf[1]

    # find first zero crossing
    zc = findfirst(acf .< 0)
    if zc === nothing || zc <= 2
        return NaN
    end

    # find first peak after zero crossing → period
    search_end = min(length(acf), zc + max_lag ÷ 2)
    if search_end <= zc
        return NaN
    end
    peak_offset = argmax(acf[zc:search_end])
    peak = zc + peak_offset - 1

    if acf[peak] < 0.1
        return NaN
    end

    period = peak * dt_save
    return 2π / period
end

"""
    run_clamped_simulation(; κ, f0, N, sim_time, substeps, burn_frac)

Run a clamped chain at given f0 for `sim_time` time units.
Returns the tip y-coordinate time series (after burn-in).
"""
function run_clamped_simulation(; κ::Float64=19.0, f0::Float64=0.0,
        N::Int=20,
        sim_time::Float64=300.0, substeps::Int=200,
        burn_frac::Float64=0.3)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l

    dt_save = substeps * p.dt
    frames  = round(Int, sim_time / dt_save)

    # straight initial chain along +x
    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    clamp1 = [0.0, 0.0]
    clamp2 = [l,   0.0]

    tip_y = Vector{Float64}(undef, frames)

    for f in 1:frames
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            state.r[:, 1] .= clamp1
            state.r[:, 2] .= clamp2
        end
        tip_y[f] = state.r[2, N]
    end

    # discard burn-in
    burn = round(Int, burn_frac * frames)
    return tip_y[burn+1:end], dt_save, p
end

function main()
    N = 20
    κ = 19.0
    L = N - 1

    # f0 values: must be well above the clamped buckling threshold
    f0_values = [1.5, 3.0, 5.0, 15.0, 25.0, 50.0]

    ω_measured = Float64[]
    f0_valid   = Float64[]

    println("="^65)
    println("  BEATING FREQUENCY SCALING TEST")
    println("  N=$N, κ=$κ, L=$L")
    println("  Expected: ω ∝ f0^(4/3)")
    println("  Simulation: 300 time units per point, 30% burn-in")
    println("="^65)

    for f0 in f0_values
        print("  f0 = $f0 ... ")

        tip_y, dt_save, _ = run_clamped_simulation(; κ=κ, f0=f0, N=N)
        ω = measure_beating_frequency(tip_y, dt_save)

        if isnan(ω)
            println("no oscillation detected")
        else
            push!(ω_measured, ω)
            push!(f0_valid, f0)
            T = 2π / ω
            println("ω = $(round(ω, digits=4)),  T = $(round(T, digits=3))")
        end
    end

    if length(f0_valid) < 3
        println("\nNot enough valid measurements for power-law fit.")
        return
    end

    # --- power-law fit: log(ω) = α·log(f0) + c ---
    log_f0 = log.(f0_valid)
    log_ω  = log.(ω_measured)
    mean_lf = mean(log_f0)
    mean_lω = mean(log_ω)
    α = sum((log_f0 .- mean_lf) .* (log_ω .- mean_lω)) / sum((log_f0 .- mean_lf).^2)
    c = mean_lω - α * mean_lf

    expected_α = 4 / 3
    rel_error = abs(α - expected_α) / expected_α

    println("\n  Measured exponent:  α = $(round(α, digits=3))")
    println("  Expected exponent: 4/3 = $(round(expected_α, digits=3))")
    println("  Relative error:    $(round(rel_error * 100, digits=1)) %")

    # --- figure ---
    fig = Figure(size=(700, 550))

    ax = Axis(fig[1, 1];
        xlabel = "f₀  (active force per bond)",
        ylabel = "ω_b  (beating frequency)",
        title  = "Beating frequency scaling  (N=$N, κ=$κ)",
        xscale = log10,
        yscale = log10,
    )

    # simulation data
    scatter!(ax, f0_valid, ω_measured;
        markersize = 12, color = :dodgerblue,
        label = "simulation")

    # fit line
    f0_fit = range(minimum(f0_valid), maximum(f0_valid), length=100)
    ω_fit = exp.(α .* log.(f0_fit) .+ c)
    lines!(ax, f0_fit, ω_fit;
        linewidth = 2, linestyle = :dash, color = :forestgreen,
        label = "fit: ω ∝ f0^$(round(α, digits=2))")

    # expected 4/3 line (same intercept)
    ω_43 = exp.(expected_α .* log.(f0_fit) .+ c)
    lines!(ax, f0_fit, ω_43;
        linewidth = 2, linestyle = :dot, color = :crimson,
        label = "theory: ω ∝ f0^(4/3)")

    axislegend(ax; position = :lt)

    display(fig)

    output_path = joinpath(@__DIR__, "beating_frequency.png")
    save(output_path, fig)
    println("\nSaved to: ", output_path)
end

main()
