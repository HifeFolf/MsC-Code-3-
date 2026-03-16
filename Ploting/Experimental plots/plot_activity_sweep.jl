# plot_activity_sweep.jl
# Sweep f0 and overlay X(t) = Σ θᵢ(t) for each activity level.
# Same clamped setup as plot_limit_cycle.jl.
# All physics parameters come from Engine defaults; only f0 varies per run.

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

function run_clamped(f0::Float64, frames::Int, substeps::Int)
    p = Engine.Params(f0=f0)   # only f0 differs from Engine defaults
    N = p.N
    l = p.l

    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    clamp_r1 = r0[:, 1]
    clamp_r2 = r0[:, 2]

    Δt_frame = substeps * p.dt
    times = Float64[0.0]
    Xvals = Float64[total_curvature(state.r, N)]
    t = 0.0

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

    return times, Xvals
end

function main()
    frames   = 2000
    substeps = 100

    f0_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors    = [:dodgerblue, :forestgreen, :goldenrod, :darkorange, :crimson]

    fig = Figure(size=(900, 500))
    ax  = Axis(fig[1, 1];
        xlabel="Time",
        ylabel="X(t) = Σ θᵢ",
        title="Mean curvature vs activity",
    )

    for (k, f0) in enumerate(f0_values)
        println("Running f0 = $f0 ...")
        times, Xvals = run_clamped(f0, frames, substeps)
        lines!(ax, times, Xvals;
            linewidth=1.5,
            color=colors[k],
            label="f₀ = $f0",
        )
    end

    axislegend(ax; position=:lt)
    display(fig)

    output_path = joinpath(@__DIR__, "activity_sweep.png")
    save(output_path, fig)
    println("Saved plot to: ", output_path)
end

main()
