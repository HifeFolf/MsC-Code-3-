# plot_active_ring.jl
# Live Makie animation of a heterogeneous active ring polymer inside a square
# box with a periodic grid of WCA pylons.
#
# Bead pattern (repeating 3×, total N = 15):
#   3 active  (:active)  — self-propulsion along local tangent
#   1 loaded  (:loaded)  — constant external load force (downward by default)
#   1 passive (:passive) — no force, purely mechanical
#
# Colour legend:  dodgerblue = active,  crimson = loaded,  gray60 = passive

using GLMakie

include(joinpath(@__DIR__, "..", "Simulation", "Engine_ring.jl"))

function pylon_positions(p)
    hw     = p.L / 2.0
    d      = p.pylon_spacing
    n_grid = floor(Int, hw / d)
    xs, ys = Float64[], Float64[]
    for gi in -n_grid:n_grid, gj in -n_grid:n_grid
        gi == 0 && gj == 0 && continue
        push!(xs, gi * d)
        push!(ys, gj * d)
    end
    return xs, ys
end

function main()
    frames   = 5000
    substeps = 100

    p = EngineRing.Params()   # N=15, L=30, pylon_spacing=6, …
    N = p.N
    l = p.l

    # --- bead type pattern: [A,A,A, L, P] × 3 ---
    block      = [:active, :active, :active, :loaded, :passive]
    bead_types = repeat(block, 3)   # length 15

    # --- initial configuration: regular polygon at centre ---
    R  = l / (2 * sin(π / N))
    r0 = zeros(2, N)
    for i in 1:N
        φ = 2π * (i - 1) / N
        r0[1, i] = R * cos(φ)
        r0[2, i] = R * sin(φ)
    end
    state = EngineRing.State(copy(r0), copy(bead_types))

    # --- per-bead colours (fixed) ---
    colour_map   = Dict(:active => :dodgerblue, :loaded => :crimson, :passive => :gray60)
    bead_colours = [colour_map[t] for t in bead_types]

    # --- observables for the ring ---
    xs     = Observable(vcat(state.r[1, :], state.r[1, 1]))
    ys     = Observable(vcat(state.r[2, :], state.r[2, 1]))
    xs_pts = Observable(state.r[1, :])
    ys_pts = Observable(state.r[2, :])

    # --- figure ---
    fig = Figure(size=(720, 720))
    ax  = Axis(fig[1, 1];
        xlabel  = "x", ylabel = "y",
        title   = "Active Ring in Box  (blue=active, red=loaded, gray=passive)",
        aspect  = DataAspect(),
    )

    hw  = p.L / 2.0
    pad = 1.0
    xlims!(ax, -hw - pad, hw + pad)
    ylims!(ax, -hw - pad, hw + pad)

    # square boundary
    box_xs = [-hw,  hw, hw, -hw, -hw]
    box_ys = [-hw, -hw, hw,  hw, -hw]
    lines!(ax, box_xs, box_ys; linewidth=2.5, color=:black)

    # pylons
    pxs, pys = pylon_positions(p)
    scatter!(ax, pxs, pys; markersize=14, color=:gray30, marker=:circle)

    # ring backbone + beads
    lines!(ax, xs, ys; linewidth=1.5, color=:black)
    scatter!(ax, xs_pts, ys_pts; markersize=12, color=bead_colours)

    display(fig)

    # --- record loop (view is fixed — box does not move) ---
    output_path = joinpath(@__DIR__, "active_ring4.mp4")
    record(fig, output_path, 1:frames; framerate=60) do _
        for _ in 1:substeps
            Base.invokelatest(EngineRing.step!, state, p)
        end
        xs[]     = vcat(state.r[1, :], state.r[1, 1])
        ys[]     = vcat(state.r[2, :], state.r[2, 1])
        xs_pts[] = state.r[1, :]
        ys_pts[] = state.r[2, :]
    end
    println("Saved animation to: ", output_path)
end

main()
