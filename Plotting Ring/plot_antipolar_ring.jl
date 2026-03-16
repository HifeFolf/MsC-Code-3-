# plot_antipolar_ring.jl
# Anti-polar active ring: beads 1–8 push counterclockwise (:active),
# beads 9–16 push clockwise (:active_rev). Starts from a perfect circle.
#
# Boundary condition: the whole ring moves freely and exits the box.
# Once EVERY bead has crossed a wall the entire ring is shifted by ±L,
# so it reappears intact on the opposite side with the same velocity.
# Forces are never computed across the boundary.
#
# Colour legend:  dodgerblue = :active (CCW),  crimson = :active_rev (CW)

using GLMakie

include(joinpath(@__DIR__, "..", "Simulation Ring", "Engine_ring.jl"))
using .EngineRing

# Shift the entire ring by ±L once it has completely exited a wall.
function wrap_ring!(r::Matrix{Float64}, L::Float64)
    hw = L / 2.0
    min_x = minimum(r[1, :]);  max_x = maximum(r[1, :])
    min_y = minimum(r[2, :]);  max_y = maximum(r[2, :])
    min_x >  hw && (r[1, :] .-= L)   # completely off right  → enter from left
    max_x < -hw && (r[1, :] .+= L)   # completely off left   → enter from right
    min_y >  hw && (r[2, :] .-= L)   # completely off top    → enter from bottom
    max_y < -hw && (r[2, :] .+= L)   # completely off bottom → enter from top
end

# Unit active-force direction for each bead (sign-flipped for :active_rev).
# All beads are always compact (ring never straddles a boundary), so no
# minimum-image convention is needed here.
function force_directions(r::Matrix{Float64}, bt::Vector{Symbol}, N::Int)
    tx = zeros(N); ty = zeros(N)
    for i in 1:N
        im = (i == 1) ? N : i - 1
        ip = (i == N) ? 1 : i + 1
        dx1 = r[1,ip] - r[1,i];  dy1 = r[2,ip] - r[2,i]
        n1  = sqrt(dx1*dx1 + dy1*dy1)
        dx2 = r[1,i]  - r[1,im]; dy2 = r[2,i]  - r[2,im]
        n2  = sqrt(dx2*dx2 + dy2*dy2)
        dx  = dx1/max(n1, 1e-14) + dx2/max(n2, 1e-14)
        dy  = dy1/max(n1, 1e-14) + dy2/max(n2, 1e-14)
        n   = sqrt(dx*dx + dy*dy)
        s   = bt[i] == :active_rev ? -1.0 : 1.0
        tx[i] = s * dx / max(n, 1e-14)
        ty[i] = s * dy / max(n, 1e-14)
    end
    return tx, ty
end

# Ring backbone as linesegments. No minimum-image needed: the ring is always
# kept intact by wrap_ring!, so all consecutive beads are within ~l of each other.
function make_backbone(r::Matrix{Float64}, N::Int)
    pts = Vector{Point2f}(undef, 2N)
    for i in 1:N
        j = (i == N) ? 1 : i + 1
        pts[2i-1] = Point2f(r[1,i], r[2,i])
        pts[2i]   = Point2f(r[1,j], r[2,j])
    end
    return pts
end

# Force-direction stubs as linesegments.
function make_segments(r::Matrix{Float64}, tx, ty, N::Int, scale::Float64)
    pts = Vector{Point2f}(undef, 2N)
    for i in 1:N
        pts[2i-1] = Point2f(r[1,i], r[2,i])
        pts[2i]   = Point2f(r[1,i] + scale*tx[i], r[2,i] + scale*ty[i])
    end
    return pts
end

function main()
    frames   = 1500   # ~25 s at 60 fps
    substeps = 100

    p = EngineRing.Params()   # N=16, f0=20, kBT=1, ks=2000, κ=1, …
    N = p.N
    l = p.l
    L = 50.0
    hw = L / 2.0

    # --- bead types: first half CCW, second half CW ---
    bead_types = vcat(fill(:active,     N ÷ 2),
                      fill(:active_rev, N ÷ 2))

    # --- initial configuration: perfect circle at centre ---
    R  = l / (2 * sin(π / N))
    r0 = zeros(2, N)
    for i in 1:N
        φ = 2π * (i - 1) / N
        r0[1, i] = R * cos(φ)
        r0[2, i] = R * sin(φ)
    end

    state = EngineRing.State(copy(r0), copy(bead_types))

    # --- colours (fixed per bead) ---
    colour_map   = Dict(:active => :dodgerblue, :active_rev => :crimson)
    bead_colours = [colour_map[t] for t in bead_types]
    seg_colours  = repeat(bead_colours; inner = 2)

    # --- observables ---
    arrow_scale = 0.65
    tx0, ty0    = force_directions(state.r, bead_types, N)

    backbone   = Observable(make_backbone(state.r, N))
    arrow_segs = Observable(make_segments(state.r, tx0, ty0, N, arrow_scale))
    xs_pts     = Observable(state.r[1, :])
    ys_pts     = Observable(state.r[2, :])

    # --- figure ---
    fig = Figure(size = (720, 720))
    ax  = Axis(fig[1, 1];
        xlabel = "x",
        ylabel = "y",
        title  = "Anti-polar ring  (blue = CCW,  red = CW)",
        aspect = DataAspect(),
    )
    xlims!(ax, -hw, hw)
    ylims!(ax, -hw, hw)

    linesegments!(ax, backbone;       linewidth = 1.5, color = :black)
    scatter!(ax,      xs_pts, ys_pts; markersize = 10, color = bead_colours)
    linesegments!(ax, arrow_segs;     linewidth = 2.5, color = seg_colours)

    param_text = "N = $(p.N)   f₀ = $(p.f0)   κ = $(p.κ)   kBT = $(p.kBT)\n" *
                 "Init: perfect circle, anti-polar (beads 1–$(N÷2) CCW, $(N÷2+1)–$(N) CW)"
    text!(ax, -hw + 0.5, hw - 1.5; text = param_text, fontsize = 13,
          color = :black, align = (:left, :top))

    display(fig)

    # --- animation ---
    output_path = joinpath(@__DIR__, "antipolar_ring.mp4")
    record(fig, output_path, 1:frames; framerate = 60) do _
        for _ in 1:substeps
            EngineRing.step!(state, p)
        end
        wrap_ring!(state.r, L)

        backbone[]   = make_backbone(state.r, N)
        xs_pts[]     = state.r[1, :]
        ys_pts[]     = state.r[2, :]
        tx, ty       = force_directions(state.r, bead_types, N)
        arrow_segs[] = make_segments(state.r, tx, ty, N, arrow_scale)
    end
    println("Saved to: ", output_path)
end

Base.invokelatest(main)
