# plot_limit_cycle.jl
# Clamped beating limit cycle.
# Clamp bead 1 (position + orientation), run with activity,
# plot mean curvature X(t) = Σ θᵢ(t) and phase portrait (X, Ẋ).
# All physics parameters come from Engine defaults.

using GLMakie
using LinearAlgebra

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function signed_angle(r::Matrix{Float64}, i::Int)
    ax = r[1,i] - r[1,i-1];  ay = r[2,i] - r[2,i-1]
    bx = r[1,i+1] - r[1,i];  by = r[2,i+1] - r[2,i]
    return atan(ax*by - ay*bx, ax*bx + ay*by)
end

function total_curvature(r::Matrix{Float64}, N::Int)
    X = 0.0
    for i in 2:N-1
        X += signed_angle(r, i)
    end
    return X
end

function main()
    frames   = 2000
    substeps = 100

    p = Engine.Params()       # single source of truth
    N = p.N
    l = p.l

    # --- initial straight chain along x ---
    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    clamp_r1 = r0[:, 1]
    clamp_r2 = r0[:, 2]

    Δt_frame = substeps * p.dt
    times = Float64[]
    Xvals = Float64[]
    t = 0.0

    push!(times, t)
    push!(Xvals, total_curvature(state.r, N))

    # --- run simulation ---
    for _ in 1:frames
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            state.r[:, 1] .= clamp_r1
            state.r[:, 2] .= clamp_r2
        end
        t += Δt_frame
        push!(times, t)
        push!(Xvals, total_curvature(state.r, N))
    end

    Xdot    = diff(Xvals) ./ Δt_frame
    X_phase = Xvals[1:end-1]

    # --- figure: 3 panels ---
    fig = Figure(size=(1400, 500))

    ax1 = Axis(fig[1, 1];
        xlabel="Time",
        ylabel="X(t) = Σ θᵢ",
        title="Mean curvature vs time",
    )
    lines!(ax1, times, Xvals; linewidth=1.5, color=:dodgerblue)

    ax2 = Axis(fig[1, 2];
        xlabel="Time",
        ylabel="Ẋ(t)",
        title="Curvature rate vs time",
    )
    lines!(ax2, times[1:end-1], Xdot; linewidth=1.5, color=:crimson)

    ax3 = Axis(fig[1, 3];
        xlabel="X(t)",
        ylabel="Ẋ(t)",
        title="Phase portrait",
        aspect=DataAspect(),
    )
    lines!(ax3, X_phase, Xdot; linewidth=1.5, color=:black, alpha=0.3)
    n_tail = length(X_phase) ÷ 4
    lines!(ax3, X_phase[end-n_tail:end], Xdot[end-n_tail:end];
        linewidth=2, color=:crimson, label="last 25%")
    axislegend(ax3; position=:rt)

    display(fig)

    output_path = joinpath(@__DIR__, "limit_cycle.png")
    save(output_path, fig)
    println("Saved plot to: ", output_path)
end

main()
