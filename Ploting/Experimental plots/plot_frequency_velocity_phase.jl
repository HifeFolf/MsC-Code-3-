# plot_frequency_velocity_phase.jl
# Two panels reproducing Isele-Holder style diagnostics:
#
#   Panel 1 (H): Beating frequency f vs COM swimming speed U
#     for a FREE active filament at various f0 values.
#     Expected: roughly linear f ~ U (both scale with f0).
#
#   Panel 2 (I): Phase portrait (Omega, Theta) for CLAMPED filament
#     showing limit-cycle orbits. Theta = tip tangent angle,
#     Omega = dTheta/dt.

using GLMakie
using Statistics

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

# ═══════════════════════════════════════════════════════════════
#  Panel H: Beating frequency and COM speed from free chain
# ═══════════════════════════════════════════════════════════════

"""
Measure beating frequency from autocorrelation of a signal.
Returns frequency in rad/time, or NaN if no oscillation found.
"""
function measure_frequency(signal::Vector{Float64}, dt::Float64)
    n = length(signal)
    y = signal .- mean(signal)
    max_lag = min(n ÷ 2, 2000)
    max_lag < 10 && return NaN

    acf = zeros(max_lag + 1)
    for lag in 0:max_lag
        acf[lag+1] = sum(y[1:end-lag] .* y[1+lag:end])
    end
    acf[1] < 1e-14 && return NaN
    acf ./= acf[1]

    zc = findfirst(acf .< 0)
    (zc === nothing || zc <= 2) && return NaN

    search_end = min(length(acf), zc + max_lag ÷ 2)
    search_end <= zc && return NaN

    peak_offset = argmax(acf[zc:search_end])
    peak = zc + peak_offset - 1
    acf[peak] < 0.1 && return NaN

    period = peak * dt
    return 2π / period
end

"""
Run a free active chain. Return:
  - COM speed (mean displacement per unit time after burn-in)
  - Beating frequency (from ACF of bond tangent angle at midpoint)
"""
function run_free_chain_observables(; κ::Float64=19.0, f0::Float64=1.0,
        N::Int=20, sim_time::Float64=500.0,
        substeps::Int=200, burn_frac::Float64=0.2)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l
    dt_save = substeps * p.dt
    frames = round(Int, sim_time / dt_save)
    burn = round(Int, burn_frac * frames)

    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    # burn-in
    for _ in 1:burn
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
        end
    end

    # record COM position + midpoint tangent angle
    n_rec = frames - burn
    com = zeros(2, n_rec)
    θ_mid = zeros(n_rec)  # tangent angle of bond near chain midpoint
    mid_bond = N ÷ 2

    for k in 1:n_rec
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
        end
        com[1, k] = sum(state.r[1, :]) / N
        com[2, k] = sum(state.r[2, :]) / N
        dx = state.r[1, mid_bond+1] - state.r[1, mid_bond]
        dy = state.r[2, mid_bond+1] - state.r[2, mid_bond]
        θ_mid[k] = atan(dy, dx)
    end

    # COM speed: total displacement / total time
    total_dx = com[1, end] - com[1, 1]
    total_dy = com[2, end] - com[2, 1]
    total_dist = sqrt(total_dx^2 + total_dy^2)
    total_time = (n_rec - 1) * dt_save
    U = total_dist / total_time

    # Beating frequency from midpoint tangent angle oscillation
    # Unwrap the angle first to avoid 2π jumps messing up the ACF
    θ_unwrapped = copy(θ_mid)
    for i in 2:length(θ_unwrapped)
        d = θ_unwrapped[i] - θ_unwrapped[i-1]
        if d > π;      θ_unwrapped[i] -= 2π
        elseif d < -π;  θ_unwrapped[i] += 2π
        end
    end
    # Remove linear trend (heading rotation) to isolate oscillation
    t_axis = (0:n_rec-1) .* dt_save
    slope = (θ_unwrapped[end] - θ_unwrapped[1]) / (t_axis[end] - t_axis[1])
    θ_detrended = θ_unwrapped .- slope .* t_axis

    ω = measure_frequency(θ_detrended, dt_save)
    freq = isnan(ω) ? NaN : ω / (2π)

    return U, freq
end

# ═══════════════════════════════════════════════════════════════
#  Panel I: Phase portrait from clamped chain
# ═══════════════════════════════════════════════════════════════

"""
Run a clamped chain and record the tip tangent angle Theta(t).
Returns (Theta, Omega, times) where Omega = dTheta/dt.
"""
function run_clamped_phase_portrait(; κ::Float64=19.0, f0::Float64=10.0,
        N::Int=20, sim_time::Float64=100.0,
        substeps::Int=200, burn_frac::Float64=0.3)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l
    dt_save = substeps * p.dt
    frames = round(Int, sim_time / dt_save)

    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    clamp1 = [0.0, 0.0]
    clamp2 = [l,   0.0]

    θ_all = Vector{Float64}(undef, frames)

    for f in 1:frames
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            state.r[:, 1] .= clamp1
            state.r[:, 2] .= clamp2
        end
        # Tip tangent angle: angle of last bond
        dx = state.r[1, N] - state.r[1, N-1]
        dy = state.r[2, N] - state.r[2, N-1]
        θ_all[f] = atan(dy, dx)
    end

    # discard burn-in
    burn = round(Int, burn_frac * frames)
    θ = θ_all[burn+1:end]
    times = (0:length(θ)-1) .* dt_save

    # unwrap angle
    for i in 2:length(θ)
        d = θ[i] - θ[i-1]
        if d > π;      θ[i] -= 2π
        elseif d < -π;  θ[i] += 2π
        end
    end

    # compute Omega = dTheta/dt via central differences
    n = length(θ)
    Ω = zeros(n)
    Ω[1] = (θ[2] - θ[1]) / dt_save
    Ω[end] = (θ[end] - θ[end-1]) / dt_save
    for i in 2:n-1
        Ω[i] = (θ[i+1] - θ[i-1]) / (2 * dt_save)
    end

    return θ, Ω, times
end

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

function main()
    N = 20
    κ = 19.0

    # ── Panel H: f vs U ────────────────────────────────────────
    f0_sweep = [2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0]

    U_vals = Float64[]
    f_vals = Float64[]

    println("Panel H: Beating frequency vs COM speed (free chain)")
    for f0 in f0_sweep
        print("  f0=$f0 ... ")
        U, freq = run_free_chain_observables(; κ=κ, f0=f0, N=N)
        println("U=$(round(U, digits=4)),  f=$(isnan(freq) ? "NaN" : round(freq, digits=4))")
        if !isnan(freq) && U > 0
            push!(U_vals, U)
            push!(f_vals, freq)
        end
    end

    # ── Panel I: Phase portrait ────────────────────────────────
    f0_phase = [5.0, 15.0, 40.0]
    phase_colors = [:forestgreen, :darkorange, :crimson]

    println("\nPanel I: Phase portraits (clamped chain)")

    # ── Figure ─────────────────────────────────────────────────
    fig = Figure(size=(1100, 500))

    # Panel H
    ax1 = Axis(fig[1, 1];
        xlabel = "U  (COM speed)",
        ylabel = "f  (beating frequency)",
        title  = "Frequency vs speed  (free chain, N=$N, κ=$κ)",
    )

    if length(U_vals) >= 2
        scatter!(ax1, U_vals, f_vals;
            markersize = 12, color = :black, label = "simulation")

        # linear fit: f = a * U
        a = sum(U_vals .* f_vals) / sum(U_vals .^ 2)
        U_fit = range(0, maximum(U_vals) * 1.1, length=100)
        lines!(ax1, U_fit, a .* U_fit;
            linewidth = 2, color = :red, linestyle = :dash,
            label = "f = $(round(a, digits=3)) U")
        axislegend(ax1; position = :lt)
    else
        println("  Not enough valid (U, f) pairs for Panel H.")
    end

    # Panel I
    ax2 = Axis(fig[1, 2];
        xlabel = "Ω  (rad/time)",
        ylabel = "Θ  (rad)",
        title  = "Phase portrait  (clamped, N=$N, κ=$κ)",
    )

    for (k, f0) in enumerate(f0_phase)
        println("  f0=$f0 ...")
        θ, Ω, _ = run_clamped_phase_portrait(; κ=κ, f0=f0, N=N)
        lines!(ax2, Ω, θ;
            linewidth = 0.8, color = (phase_colors[k], 0.7),
            label = "f0=$f0")
    end
    axislegend(ax2; position = :rt)

    display(fig)

    output_path = joinpath(@__DIR__, "frequency_velocity_phase.png")
    save(output_path, fig; px_per_unit=2)
    println("\nSaved to: ", output_path)
end

main()
