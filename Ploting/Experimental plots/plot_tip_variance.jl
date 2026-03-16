# plot_tip_variance.jl
# Ensemble-averaged tip displacement variance ⟨Δr²_tip(t)⟩ for a clamped
# active filament, plotted on linear axes.
#
# Multiple independent trajectories start from the same straight initial
# condition, so the oscillatory beating structure is preserved in the average.
#
# Sweep over f0 values → each curve shows the characteristic beating period
# and growing decorrelation envelope (cf. Isele-Holder Fig. D).

using GLMakie
using Statistics

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

"""
    run_ensemble_tip_variance(; κ, f0, N, n_traj, sim_time, substeps)

Launch `n_traj` independent clamped simulations from a straight chain,
record tip position at each frame, return ensemble-averaged ⟨Δr²(t)⟩
and the time axis.
"""
function run_ensemble_tip_variance(; κ::Float64=19.0, f0::Float64=0.0,
        N::Int=20, n_traj::Int=50,
        sim_time::Float64=20.0, substeps::Int=200)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l
    dt_save = substeps * p.dt
    frames  = round(Int, sim_time / dt_save)

    # straight initial chain
    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end

    clamp1 = [0.0, 0.0]
    clamp2 = [l,   0.0]

    # initial tip position (same for all trajectories)
    tip0 = r0[:, N]

    # accumulate ⟨Δr²(t)⟩
    dr2 = zeros(frames)

    for m in 1:n_traj
        state = Engine.State(copy(r0))
        for f in 1:frames
            for _ in 1:substeps
                Base.invokelatest(Engine.step!, state, p)
                state.r[:, 1] .= clamp1
                state.r[:, 2] .= clamp2
            end
            dx = state.r[1, N] - tip0[1]
            dy = state.r[2, N] - tip0[2]
            dr2[f] += dx*dx + dy*dy
        end
    end

    dr2 ./= n_traj
    times = (1:frames) .* dt_save
    return collect(times), dr2
end

function main()
    N = 20
    κ = 19.0
    n_traj = 50
    sim_time = 20.0

    f0_values = [1.5, 3.0, 5.0, 15.0, 25.0, 50.0]

    # colour ramp from cool to warm
    n = length(f0_values)
    colors = [RGBf(0.2 + 0.7*k/n, 0.6 - 0.4*k/n, 0.9 - 0.8*k/n) for k in 1:n]

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1];
        xlabel = "t",
        ylabel = "⟨Δr²_tip(t)⟩",
        title  = "Clamped tip variance  (N=$N, κ=$κ, $n_traj trajectories)",
    )

    for (k, f0) in enumerate(f0_values)
        println("f0=$f0  ($n_traj trajectories, T=$sim_time) ...")
        t, dr2 = run_ensemble_tip_variance(;
            κ=κ, f0=f0, N=N, n_traj=n_traj, sim_time=sim_time)
        lines!(ax, t, dr2; linewidth=2, color=colors[k],
               label="f0=$f0")
    end

    axislegend(ax; position=:lt)
    display(fig)

    output_path = joinpath(@__DIR__, "tip_variance.png")
    save(output_path, fig; px_per_unit=2)
    println("\nSaved to: ", output_path)
end

main()
