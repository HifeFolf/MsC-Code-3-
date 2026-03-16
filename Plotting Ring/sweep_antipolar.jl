# sweep_antipolar.jl
# Parameter sweep over f0 and κ (steps of 5) for the anti-polar ring.
# N=16, initial perfect circle, beads 1–8 CCW / 9–16 CW — fixed throughout.
#
# Boundary condition: whole-ring wrap. The ring exits freely; once every bead
# has crossed a wall the entire ring is shifted by ±L so it re-enters intact
# from the opposite side with the same velocity. No cross-boundary forces.
#
# Layout: ring animation (left) + 4 time-series plots (right).
# X-axis of time series = MP4 timestamp in seconds for easy cross-referencing.
#
# Saves one MP4 per (f0, κ) combination; ~25 s each at 60 fps.

using GLMakie, Printf

include(joinpath(@__DIR__, "..", "Simulation Ring", "Engine_ring.jl"))
using .EngineRing

function wrap_ring!(r::Matrix{Float64}, L::Float64)
    hw = L / 2.0
    min_x = minimum(r[1, :]); max_x = maximum(r[1, :])
    min_y = minimum(r[2, :]); max_y = maximum(r[2, :])
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

function make_segments(r::Matrix{Float64}, tx, ty, N::Int, scale::Float64)
    pts = Vector{Point2f}(undef, 2N)
    for i in 1:N
        pts[2i-1] = Point2f(r[1,i], r[2,i])
        pts[2i]   = Point2f(r[1,i] + scale*tx[i], r[2,i] + scale*ty[i])
    end
    return pts
end

# Returns (Gxx, Gyy, Gxy, λ1, λ2, Rg, b)
function gyration_vals(r::Matrix{Float64})
    N   = size(r, 2)
    xcm = sum(r[1, :]) / N
    ycm = sum(r[2, :]) / N
    Gxx = sum((r[1, i] - xcm)^2 for i in 1:N) / N
    Gyy = sum((r[2, i] - ycm)^2 for i in 1:N) / N
    Gxy = sum((r[1, i] - xcm) * (r[2, i] - ycm) for i in 1:N) / N
    mid  = (Gxx + Gyy) / 2.0
    half = sqrt(((Gxx - Gyy) / 2.0)^2 + Gxy^2)
    λ1   = mid + half
    λ2   = mid - half
    rg2  = λ1 + λ2
    b    = rg2 > 1e-30 ? (λ1 - λ2)^2 / rg2^2 : 0.0
    return Gxx, Gyy, Gxy, λ1, λ2, sqrt(max(rg2, 0.0)), b
end

function run_one(f0_val::Float64, κ_val::Float64, output_path::String)
    frames    = 1500
    substeps  = 100
    framerate = 60
    L         = 50.0
    hw        = L / 2.0
    t_end     = frames / framerate          # 25.0 s

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

    # ── Ring observables ──────────────────────────────────────────────────
    backbone   = Observable(make_backbone(state.r, N))
    arrow_segs = Observable(make_segments(state.r, tx0, ty0, N, arrow_scale))
    xs_pts     = Observable(state.r[1, :])
    ys_pts     = Observable(state.r[2, :])

    Gxx0, Gyy0, Gxy0, λ1_0, λ2_0, rg0, b0 = gyration_vals(state.r)
    rg_str = Observable(
        @sprintf("Gxx=%.3f  Gyy=%.3f\nGxy=%.3f\nλ₁=%.3f  λ₂=%.3f\nRg²=%.3f\nb=%.3f",
                 Gxx0, Gyy0, Gxy0, λ1_0, λ2_0, rg0^2, b0))

    # ── Time-series observables ───────────────────────────────────────────
    rg_data = Observable(Point2f[])
    λ1_data = Observable(Point2f[])
    λ2_data = Observable(Point2f[])
    b_data  = Observable(Point2f[])
    t_line  = Observable([0.0f0])

    # ── Figure: 1400×780, ring left, 4 time-series right ─────────────────
    fig = Figure(size = (1400, 780))

    ax_ring = Axis(fig[1:4, 1];
        xlabel = "x", ylabel = "y",
        title  = "Anti-polar ring  (blue = CCW,  red = CW)",
        aspect = DataAspect(),
    )
    xlims!(ax_ring, -hw, hw)
    ylims!(ax_ring, -hw, hw)

    linesegments!(ax_ring, backbone;       linewidth = 1.5, color = :black)
    scatter!(ax_ring,      xs_pts, ys_pts; markersize = 10, color = bead_colours)
    linesegments!(ax_ring, arrow_segs;     linewidth = 2.5, color = seg_colours)

    param_text = "N = $(p.N)   f₀ = $(p.f0)   κ = $(p.κ)   kBT = $(p.kBT)\n" *
                 "Init: perfect circle, anti-polar (beads 1–$(N÷2) CCW, $(N÷2+1)–$(N) CW)"
    text!(ax_ring, -hw + 0.5, hw - 1.5; text = param_text, fontsize = 13,
          color = :black, align = (:left, :top))
    text!(ax_ring,  hw - 0.5, hw - 1.5; text = rg_str,     fontsize = 12,
          color = :black, align = (:right, :top))

    # ── Time-series axes ──────────────────────────────────────────────────
    ax_rg = Axis(fig[1, 2]; ylabel = "Rg",  xticklabelsvisible = false)
    ax_λ1 = Axis(fig[2, 2]; ylabel = "λ₁",  xticklabelsvisible = false)
    ax_λ2 = Axis(fig[3, 2]; ylabel = "λ₂",  xticklabelsvisible = false)
    ax_b  = Axis(fig[4, 2]; ylabel = "b",   xlabel = "time  (s)")

    for ax in (ax_rg, ax_λ1, ax_λ2, ax_b)
        xlims!(ax, 0.0, t_end)
        vlines!(ax, t_line; color = :gray50, linestyle = :dash, linewidth = 1.5)
    end
    linkxaxes!(ax_rg, ax_λ1, ax_λ2, ax_b)

    lines!(ax_rg, rg_data; color = :dodgerblue, linewidth = 1.5)
    lines!(ax_λ1, λ1_data; color = :crimson,    linewidth = 1.5)
    lines!(ax_λ2, λ2_data; color = :orange,     linewidth = 1.5)
    lines!(ax_b,  b_data;  color = :green,      linewidth = 1.5)

    # ── Record loop ───────────────────────────────────────────────────────
    record(fig, output_path, 1:frames; framerate = framerate) do frame_idx
        for _ in 1:substeps
            EngineRing.step!(state, p)
        end
        wrap_ring!(state.r, L)

        backbone[]   = make_backbone(state.r, N)
        xs_pts[]     = state.r[1, :]
        ys_pts[]     = state.r[2, :]
        tx, ty       = force_directions(state.r, bead_types, N)
        arrow_segs[] = make_segments(state.r, tx, ty, N, arrow_scale)

        Gxx, Gyy, Gxy, λ1, λ2, rg, b = gyration_vals(state.r)
        rg_str[] = @sprintf(
            "Gxx=%.3f  Gyy=%.3f\nGxy=%.3f\nλ₁=%.3f  λ₂=%.3f\nRg²=%.3f\nb=%.3f",
            Gxx, Gyy, Gxy, λ1, λ2, rg^2, b)

        t = Float32(frame_idx / framerate)
        push!(rg_data[],  Point2f(t, Float32(rg)))
        push!(λ1_data[],  Point2f(t, Float32(λ1)))
        push!(λ2_data[],  Point2f(t, Float32(λ2)))
        push!(b_data[],   Point2f(t, Float32(b)))
        notify(rg_data); notify(λ1_data); notify(λ2_data); notify(b_data)

        t_line[] = [t]
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
