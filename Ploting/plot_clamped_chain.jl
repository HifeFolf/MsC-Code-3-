

using GLMakie

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function main()
    frames   = 1200
    substeps = 100

    p = Engine.Params()
    N = p.N
    l = p.l

    # --- initial straight chain along +x ---
    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    # clamped positions (bead 1 at origin, bead 2 at (l, 0))
    clamp1 = [0.0, 0.0]
    clamp2 = [l,   0.0]

    # --- observables ---
    xs = Observable(state.r[1, :])
    ys = Observable(state.r[2, :])

    # --- figure ---
    fig = Figure(size=(800, 600))
    ax  = Axis(fig[1, 1];
        xlabel="x", ylabel="y",
        title="Clamped Active Chain  (κ=$(p.κ), f0=$(p.f0))",
        aspect=DataAspect(),
    )

    # fixed view centred on the clamp
    L = (N - 1) * l
    xlims!(ax, -L * 0.3, L * 1.1)
    ylims!(ax, -L * 0.7, L * 0.7)

    # draw clamp anchor
    scatter!(ax, [clamp1[1]], [clamp1[2]]; markersize=18, color=:red, marker=:xcross)

    lines!(ax, xs, ys; linewidth=2, color=:dodgerblue)
    scatter!(ax, xs, ys; markersize=10, color=:dodgerblue)

    display(fig)

    # --- record ---
    output_path = joinpath(@__DIR__, "clamped_chain3.mp4")
    record(fig, output_path, 1:frames; framerate=60) do _
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)

            # enforce clamp after each substep
            state.r[:, 1] .= clamp1
            state.r[:, 2] .= clamp2
        end
        xs[] = state.r[1, :]
        ys[] = state.r[2, :]
    end
    println("Saved animation to: ", output_path)
end

main()
