using GLMakie

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function main()
    frames   = 1200
    substeps = 100

    p = Engine.Params()       
    N = p.N
    l = p.l

    
    r0 = zeros(2, N)
    zig_angle = π/3
    for i in 2:N
        θ = iseven(i) ? zig_angle : -zig_angle
        r0[1, i] = r0[1, i-1] + l * cos(θ)
        r0[2, i] = r0[2, i-1] + l * sin(θ)
    end
    state = Engine.State(copy(r0))

    # --- observables ---
    xs = Observable(state.r[1, :])
    ys = Observable(state.r[2, :])

    # --- figure ---
    fig = Figure(size=(800, 600))
    ax  = Axis(fig[1, 1];
        xlabel="x", ylabel="y",
        title="Active Chain (Level 0)",
        aspect=DataAspect(),
    )
    half_w = N * l / 2 + 2
    half_h = N * l / 2

    lines!(ax, xs, ys; linewidth=2, color=:dodgerblue)
    scatter!(ax, xs, ys; markersize=12, color=:dodgerblue)

    display(fig)

    
    output_path = joinpath(@__DIR__, "active_chain8.mp4")
    record(fig, output_path, 1:frames; framerate=60) do _
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
        end
        xs[] = state.r[1, :]
        ys[] = state.r[2, :]
        cx = sum(xs[]) / N
        cy = sum(ys[]) / N
        if isfinite(cx) && isfinite(cy)
            xlims!(ax, cx - half_w, cx + half_w)
            ylims!(ax, cy - half_h, cy + half_h)
        end
    end
    println("Saved animation to: ", output_path)
end

main()
