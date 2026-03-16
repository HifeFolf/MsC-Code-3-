# plot_diffusion.jl
# Validation N1: single bead diffusion (no forces).
#
# One bead, no springs, no active force, thermal noise only.
# Overdamped Langevin:  dr = √(2μk_BT dt) ξ
#
# In 2D the MSD should be:
#   MSD(t) = ⟨|r(t) - r(0)|²⟩ = 4Dt,   D = μk_BT
#
# So the slope  d(MSD)/dt = 4μk_BT  validates noise amplitude and units.
#
# Graph: MSD vs time with fitted slope and theoretical line.

using GLMakie
using Statistics

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function main()
    n_traj   = 500        # number of independent trajectories (ensemble)
    nsteps   = 100_000    # steps per trajectory
    save_every = 100      # save positions every this many steps

    # Single bead, no activity, no springs, no bending, no WCA
    p = Engine.Params(N=1, f0=0.0, ks=0.0, κ=0.0, epsilon=0.0)
    mu  = p.mu
    kBT = p.kBT
    dt  = p.dt

    D_theory = mu * kBT                  # diffusion coefficient
    slope_theory = 4 * D_theory          # d(MSD)/dt in 2D

    n_save = div(nsteps, save_every)
    times  = collect(1:n_save) .* (save_every * dt)

    # Accumulate MSD over ensemble
    msd = zeros(n_save)

    for traj in 1:n_traj
        state = Engine.State(zeros(2, 1))
        r0 = copy(state.r)

        for s in 1:nsteps
            Base.invokelatest(Engine.step!, state, p)

            if s % save_every == 0
                idx = div(s, save_every)
                dx = state.r[1, 1] - r0[1, 1]
                dy = state.r[2, 1] - r0[2, 1]
                msd[idx] += dx * dx + dy * dy
            end
        end
    end
    msd ./= n_traj

    # --- linear fit: MSD = a * t + b ---
    t_fit = times
    mean_t   = mean(t_fit)
    mean_msd = mean(msd)
    slope_fit = sum((t_fit .- mean_t) .* (msd .- mean_msd)) / sum((t_fit .- mean_t).^2)
    intercept = mean_msd - slope_fit * mean_t
    D_fit = slope_fit / 4

    # --- figure ---
    fig = Figure(size=(700, 500))

    ax = Axis(fig[1, 1];
        xlabel = "Time  t",
        ylabel = "MSD(t) = ⟨|r(t) − r(0)|²⟩",
        title  = "Single bead diffusion  (Validation N1)",
    )

    scatter!(ax, times, msd;
        markersize = 4, color = :dodgerblue, label = "simulation  ($n_traj trajectories)")

    lines!(ax, times, slope_theory .* times;
        color = :crimson, linestyle = :dash, linewidth = 2,
        label = "theory: 4μk_BT · t = $(round(slope_theory, digits=4)) t")

    lines!(ax, times, slope_fit .* times .+ intercept;
        color = :forestgreen, linestyle = :dot, linewidth = 2,
        label = "fit: slope = $(round(slope_fit, digits=4))")

    axislegend(ax; position = :lt)

    display(fig)

    output_path = joinpath(@__DIR__, "diffusion.png")
    save(output_path, fig)
    println("Saved to: ", output_path)
    println()
    println("Theory:   D = μk_BT = $D_theory")
    println("          slope = 4D = $slope_theory")
    println("Fit:      D = $(round(D_fit, digits=6))")
    println("          slope = $(round(slope_fit, digits=6))")
    println("Error:    $(round(abs(slope_fit - slope_theory) / slope_theory * 100, digits=2)) %")
end

main()
