# sweep_antipolar.jl
# Parameter sweep over f0 and κ (steps of 5) for the anti-polar ring.
# N=16, initial perfect circle, beads 1–8 CCW / 9–16 CW — fixed throughout.
#
# Boundary condition: whole-ring wrap. The ring exits freely; once every bead
# has crossed a wall the entire ring is shifted by ±L so it re-enters intact
# from the opposite side with the same velocity. No cross-boundary forces.
#
# Saves one MP4 per (f0, κ) combination; ~25 s each at 60 fps.

using GLMakie, Printf

include(joinpath(@__DIR__, "..", "Simulation Ring", "Engine_ring.jl"))
using .EngineRing

# Shift the entire ring by ±L once it has completely exited a wall.
function wrap_ring!(r::Matrix{Float64}, L::Float64)
    hw = L / 2.0
    min_x = minimum(r[1, :]);  max_x = maximum(r[1, :])
    min_y = minimum(r[2, :]);  max_y = maximum(r[2, :])
    min_x >  hw && (r[1, :] .-= L)
    max_x < -hw && (r[1, :] .+= L)
    min_y >  hw && (r[2, :] .-= L)
    max_y < -hw && (r[2, :] .+= L)
end

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

function make_backbone(r::Matrix{Float64}, N::Int)
    pts = Vector{Point2f}(undef, 2N)
    for i in 1:N
        j = (i == N) ? 1 : i + 1
        pts[2i-1] = Point2f(r[1,i], r[2,i])
        pts[2i]   = Point2f(r[1,j], r[2,j])
    end
    return pts
end

function radius_of_gyration(r::Matrix{Float64})
    N    = size(r, 2)
    xcm  = sum(r[1, :]) / N
    ycm  = sum(r[2, :]) / N
    rg2  = sum((r[1, i] - xcm)^2 + (r[2, i] - ycm)^2 for i in 1:N) / N
    return sqrt(rg2)
end

function make_segments(r::Matrix{Float64}, tx, ty, N::Int, scale::Float64)
    pts = Vector{Point2f}(undef, 2N)
    for i in 1:N
        pts[2i-1] = Point2f(r[1,i], r[2,i])
        pts[2i]   = Point2f(r[1,i] + scale*tx[i], r[2,i] + scale*ty[i])
    end
    return pts
end

function run_one(f0_val::Float64, κ_val::Float64, output_path::String)
    frames   = 1500
    substeps = 100
    L        = 50.0
    hw       = L / 2.0

    p = EngineRing.Params(f0 = f0_val, κ = κ_val)
    N = p.N
    l = p.l

    bead_types = vcat(fill(:active,     N ÷ 2),
                      fill(:active_rev, N ÷ 2))

    R  = l / (2 * sin(π / N))
    r0 = zeros(2, N)
    for i in 1:N
        φ = 2π * (i - 1) / N
        r0[1, i] = R * cos(φ)
        r0[2, i] = R * sin(φ)
    end
    state = EngineRing.State(copy(r0), copy(bead_types))

    colour_map   = Dict(:active => :dodgerblue, :active_rev => :crimson)
    bead_colours = [colour_map[t] for t in bead_types]
    seg_colours  = repeat(bead_colours; inner = 2)

    arrow_scale = 0.65
    tx0, ty0    = force_directions(state.r, bead_types, N)

    backbone   = Observable(make_backbone(state.r, N))
    arrow_segs = Observable(make_segments(state.r, tx0, ty0, N, arrow_scale))
    xs_pts     = Observable(state.r[1, :])
    ys_pts     = Observable(state.r[2, :])

    fig = Figure(size = (720, 720))
    ax  = Axis(fig[1, 1];
        xlabel = "x", ylabel = "y",
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

    rg_text = Observable(@sprintf("Rg = %.3f", radius_of_gyration(state.r)))
    text!(ax, hw - 0.5, hw - 1.5; text = rg_text, fontsize = 13,
          color = :black, align = (:right, :top))

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
        rg_text[]    = @sprintf("Rg = %.3f", radius_of_gyration(state.r))
    end

    println("  saved: ", basename(output_path))
end

function main()
    f0_vals = 5.0:5.0:25.0
    κ_vals  = 5.0:5.0:25.0
    total   = length(f0_vals) * length(κ_vals)
    outdir  = @__DIR__

    println("Running $(total) parameter combinations …")
    count = 0
    for κ in κ_vals, f0 in f0_vals
        count += 1
        fname = @sprintf("antipolar_f0_%02d_k_%02d.mp4", Int(f0), Int(κ))
        println("[$count/$total]  f0=$(f0)  κ=$(κ)")
        Base.invokelatest(run_one, f0, κ, joinpath(outdir, fname))
    end
    println("Done.")
end

Base.invokelatest(main)
