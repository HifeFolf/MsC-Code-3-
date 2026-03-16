# plot_terminal_velocity.jl
# Validation 1: free bead terminal velocity.
#
# One bead, constant external force F = f₀ x̂, no springs, no noise.
# Overdamped Langevin:  dr/dt = μ F
# Expected terminal velocity (immediate):  v∞ = μ f₀
#
# Because overdamped dynamics have no inertia, the bead reaches v∞
# from the very first step. The plot should show:
#   Panel 1 — speed flat at v∞ = μ f₀  (no transient)
#   Panel 2 — position linear: x(t) = v∞ t
#
# f₀, μ, dt all come from Engine defaults.

using GLMakie

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function main()
    nsteps = 5_000     # number of steps (acts as "frames" for this script)

    p   = Engine.Params()    # single source of truth
    f0  = p.f0
    mu  = p.mu
    dt  = p.dt

    v_inf = mu * f0   # theoretical terminal velocity

    # --- run: bare Euler step  r ← r + μ F dt  (Engine.jl's step!) ---
    x = 0.0
    t = 0.0

    times  = Vector{Float64}(undef, nsteps)
    speeds = Vector{Float64}(undef, nsteps)
    xs     = Vector{Float64}(undef, nsteps)

    for k in 1:nsteps
        dx  = mu * f0 * dt
        x  += dx
        t  += dt
        times[k]  = t
        speeds[k] = dx / dt
        xs[k]     = x
    end

    # --- figure ---
    fig = Figure(size=(950, 420))

    ax1 = Axis(fig[1, 1];
        xlabel = "Time  t",
        ylabel = "Speed  v(t)",
        title  = "Terminal velocity  (Validation 1)",
    )
    lines!(ax1, times, speeds;
        linewidth = 2.5, color = :dodgerblue, label = "simulation")
    hlines!(ax1, [v_inf];
        color = :crimson, linestyle = :dash, linewidth = 1.5,
        label = "v∞ = μ f₀ = $v_inf")
    ylims!(ax1, 0.0, 1.5 * v_inf)
    axislegend(ax1; position = :rb)

    ax2 = Axis(fig[1, 2];
        xlabel = "Time  t",
        ylabel = "x(t)",
        title  = "Position vs time  (slope = v∞)",
    )
    lines!(ax2, times, xs;
        linewidth = 2.5, color = :forestgreen, label = "simulation")
    lines!(ax2, times, v_inf .* times;
        color = :crimson, linestyle = :dash, linewidth = 1.5,
        label = "x = v∞ · t")
    axislegend(ax2; position = :lt)

    display(fig)

    output_path = joinpath(@__DIR__, "terminal_velocity.png")
    save(output_path, fig)
    println("Saved to: ", output_path)
    println("Theoretical  v∞ = μ f₀ = $(v_inf)")
    println("Simulated    v  = $(speeds[end])   (should match exactly)")
end

main()
