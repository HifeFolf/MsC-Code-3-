# plot_msd_chain.jl
# Mean Squared Displacement for the active chain in two regimes:
#
#   Panel 1: FREE CHAIN ("railway motion")
#     MSD of the centre of mass.  Expected:
#       short τ → ballistic (∝ τ²) from directed self-propulsion
#       long  τ → diffusive (∝ τ)  from rotational diffusion randomising heading
#
#   Panel 2: CLAMPED CHAIN (tip MSD)
#     MSD of the free tip (bead N) relative to the clamp.  Expected:
#       periodic beating → MSD oscillates and saturates
#       irregular/high f0 → MSD grows then saturates at ~L²
#
# Each panel shows multiple f0 values + a passive reference.

using GLMakie
using Statistics

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

"""Time-averaged MSD from a 2×T trajectory."""
function tamsd(traj::Matrix{Float64}, dt_frame::Float64)
    T = size(traj, 2)
    max_lag = T ÷ 2
    msd = zeros(max_lag)
    @inbounds for lag in 1:max_lag
        n = T - lag
        s = 0.0
        for t in 1:n
            dx = traj[1, t+lag] - traj[1, t]
            dy = traj[2, t+lag] - traj[2, t]
            s += dx*dx + dy*dy
        end
        msd[lag] = s / n
    end
    return collect((1:max_lag) .* dt_frame), msd
end

# ═══════════════════════════════════════════════════════════════
#  Free chain: COM trajectory
# ═══════════════════════════════════════════════════════════════

function run_free_com(; κ::Float64=19.0, f0::Float64=0.0,
        N::Int=20, sim_time::Float64=500.0,
        substeps::Int=200, burn_frac::Float64=0.1)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l
    dt_save = substeps * p.dt
    frames  = round(Int, sim_time / dt_save)
    burn    = round(Int, burn_frac * frames)

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

    # record COM
    n_rec = frames - burn
    com = zeros(2, n_rec)
    for k in 1:n_rec
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
        end
        com[1, k] = sum(state.r[1, :]) / N
        com[2, k] = sum(state.r[2, :]) / N
    end

    return com, dt_save
end

# ═══════════════════════════════════════════════════════════════
#  Clamped chain: tip trajectory
# ═══════════════════════════════════════════════════════════════

function run_clamped_tip(; κ::Float64=19.0, f0::Float64=0.0,
        N::Int=20, sim_time::Float64=500.0,
        substeps::Int=200, burn_frac::Float64=0.1)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l
    dt_save = substeps * p.dt
    frames  = round(Int, sim_time / dt_save)
    burn    = round(Int, burn_frac * frames)

    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    clamp1 = [0.0, 0.0]
    clamp2 = [l,   0.0]

    # burn-in
    for _ in 1:burn
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            state.r[:, 1] .= clamp1
            state.r[:, 2] .= clamp2
        end
    end

    # record tip position
    n_rec = frames - burn
    tip = zeros(2, n_rec)
    for k in 1:n_rec
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            state.r[:, 1] .= clamp1
            state.r[:, 2] .= clamp2
        end
        tip[1, k] = state.r[1, N]
        tip[2, k] = state.r[2, N]
    end

    return tip, dt_save
end

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

function main()
    N = 20
    κ = 19.0
    sim_time = 500.0

    f0_values = [0.0, 1.5, 5.0, 25.0]
    colors    = [:gray50, :dodgerblue, :forestgreen, :crimson]

    fig = Figure(size=(1300, 550))

    # ── Panel 1: Free chain COM MSD ──────────────────────────
    ax1 = Axis(fig[1, 1];
        xlabel = "lag time  τ",
        ylabel = "MSD  ⟨|Δr_com|²⟩",
        title  = "Free chain COM  (N=$N, κ=$κ)",
        xscale = log10, yscale = log10,
    )

    for (k, f0) in enumerate(f0_values)
        label = f0 == 0 ? "passive" : "f0=$f0"
        println("Free chain: f0=$f0 ...")
        com, dt = run_free_com(; κ=κ, f0=f0, N=N, sim_time=sim_time)
        τ, msd = tamsd(com, dt)
        lines!(ax1, τ, msd; linewidth=2, color=colors[k], label=label)
    end

    axislegend(ax1; position=:lt)

    # ── Panel 2: Clamped chain tip MSD ───────────────────────
    ax2 = Axis(fig[1, 2];
        xlabel = "lag time  τ",
        ylabel = "MSD  ⟨|Δr_tip|²⟩",
        title  = "Clamped chain tip  (N=$N, κ=$κ)",
        xscale = log10, yscale = log10,
    )

    for (k, f0) in enumerate(f0_values)
        label = f0 == 0 ? "passive" : "f0=$f0"
        println("Clamped chain: f0=$f0 ...")
        tip, dt = run_clamped_tip(; κ=κ, f0=f0, N=N, sim_time=sim_time)
        τ, msd = tamsd(tip, dt)
        lines!(ax2, τ, msd; linewidth=2, color=colors[k], label=label)
    end

    # saturation reference: L² = (N-1)²
    L = N - 1
    hlines!(ax2, [L^2]; color=:black, linestyle=:dot, linewidth=1,
            label="L² = $(L^2)")
    axislegend(ax2; position=:lt)

    # add power-law guides to Panel 1
    # (do this after all data is plotted so limits are known)
    τ_lo = 1e-2; τ_hi = 1e2
    A_ref = 1.0
    lines!(ax1, [τ_lo, τ_hi], A_ref .* [τ_lo, τ_hi];
        color=:black, linestyle=:dash, linewidth=1, label="∝ τ¹")
    lines!(ax1, [τ_lo, τ_hi], A_ref .* ([τ_lo, τ_hi] .^ 2);
        color=:black, linestyle=:dot, linewidth=1, label="∝ τ²")

    display(fig)

    output_path = joinpath(@__DIR__, "msd_chain.png")
    save(output_path, fig; px_per_unit=2)
    println("\nSaved to: ", output_path)
end

main()
