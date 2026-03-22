# render_movie.jl
# Replay a saved trajectory and render it to preview.mp4.
# No physics is recomputed — all positions are loaded from traj.jld2.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/render_movie.jl results/run_0001
#
# Output:  results/run_0001/preview.mp4
#
# Layout:  ring animation (left) + 5 time-series plots (right):
#          Rg, A2, A3, Kloc, b

using GLMakie, Printf

const PROJECT_ROOT_RM = joinpath(@__DIR__, "..")

if !isdefined(Main, :EngineRing)
    include(joinpath(PROJECT_ROOT_RM, "src", "EngineRing.jl"))
    using .EngineRing
end
if !isdefined(Main, :IOUtils)
    include(joinpath(PROJECT_ROOT_RM, "src", "IOUtils.jl"))
    using .IOUtils
end
if !isdefined(Main, :Observables)
    include(joinpath(PROJECT_ROOT_RM, "src", "Observables.jl"))
    using .Observables
end

# ── Rendering geometry helpers (local — GLMakie-specific) ────────────────────

"""
Build the linesegments point list for the ring backbone (N closed bonds).
Each bond i→j contributes two consecutive Point2f entries.
"""
function make_backbone(r::Matrix{Float64}, N::Int)
    pts = Vector{Point2f}(undef, 2N)
    for i in 1:N
        j = (i == N) ? 1 : i + 1
        pts[2i-1] = Point2f(r[1, i], r[2, i])
        pts[2i]   = Point2f(r[1, j], r[2, j])
    end
    return pts
end

"""
Build the linesegments point list for the propulsion arrows.
Each bead i contributes a segment from its position to position + scale·t̂_i.
"""
function make_segments(r::Matrix{Float64}, tx, ty, N::Int, scale::Float64)
    pts = Vector{Point2f}(undef, 2N)
    for i in 1:N
        pts[2i-1] = Point2f(r[1, i], r[2, i])
        pts[2i]   = Point2f(r[1, i] + scale * tx[i], r[2, i] + scale * ty[i])
    end
    return pts
end

# ── Main render function ──────────────────────────────────────────────────────

"""
    render_movie(run_dir; output_name="preview.mp4")

Load saved trajectory and observables from `run_dir`, then render an MP4.
The layout and colour scheme match the original plot_antipolar_ring.jl exactly.
"""
function render_movie(run_dir::String; output_name::String = "preview.mp4")
    # ── Load data ────────────────────────────────────────────────────────────
    _, positions, bead_types = IOUtils.load_trajectory(run_dir)
    _, obs                       = IOUtils.load_observables(run_dir)
    p, rc                        = IOUtils.load_params(run_dir)
    meta                         = IOUtils.load_metadata(run_dir)

    n_frames  = size(positions, 3)
    N         = p.N
    framerate = Int(rc["framerate"])
    L         = Float64(rc["L"])
    hw        = L / 2.0
    t_end     = (n_frames - 1) / framerate   # video seconds, matches MP4 duration

    # ── Colours ──────────────────────────────────────────────────────────────
    colour_map   = Dict(:active => :dodgerblue, :active_rev => :crimson)
    bead_colours = [colour_map[t] for t in bead_types]
    seg_colours  = repeat(bead_colours; inner = 2)

    arrow_scale = 0.65

    # ── Initial frame ─────────────────────────────────────────────────────────
    r0           = positions[:, :, 1]
    tx0, ty0     = Observables.force_directions(r0, bead_types, N)
    gv0          = Observables.gyration_vals(r0)
    cv0          = Observables.contour_vals(r0)

    # ── GLMakie observables ───────────────────────────────────────────────────
    backbone_obs   = Observable(make_backbone(r0, N))
    arrow_segs_obs = Observable(make_segments(r0, tx0, ty0, N, arrow_scale))
    xs_pts         = Observable(r0[1, :])
    ys_pts         = Observable(r0[2, :])

    overlay_str = Observable(
        @sprintf("Gxx=%.3f  Gyy=%.3f\nGxy=%.3f\nA₂=%.3f  A₃=%.3f\nKloc=%.3f\nRg²=%.3f\nb=%.3f",
                 gv0.Gxx, gv0.Gyy, gv0.Gxy,
                 cv0.A2, cv0.A3, cv0.Kloc,
                 gv0.Rg^2, gv0.b))

    rg_data   = Observable(Point2f[])
    A2_data   = Observable(Point2f[])
    A3_data   = Observable(Point2f[])
    Kloc_data = Observable(Point2f[])
    b_data    = Observable(Point2f[])
    t_line    = Observable([0.0f0])

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = Figure(size = (1400, 900))

    ax_ring = Axis(fig[1:5, 1];
        xlabel = "x", ylabel = "y",
        title  = "Anti-polar ring  (blue = CCW,  red = CW)",
        aspect = DataAspect(),
    )
    xlims!(ax_ring, -hw, hw)
    ylims!(ax_ring, -hw, hw)

    linesegments!(ax_ring, backbone_obs;       linewidth = 1.5, color = :black)
    scatter!(ax_ring,      xs_pts, ys_pts;     markersize = 10, color = bead_colours)
    linesegments!(ax_ring, arrow_segs_obs;     linewidth = 2.5, color = seg_colours)

    param_text = "N = $(p.N)   f₀ = $(p.f0)   κ = $(p.κ)   kBT = $(p.kBT)\n" *
                 "Run: $(meta["run_id"])   seed = $(meta["seed"])"
    text!(ax_ring, -hw + 0.5, hw - 1.5; text = param_text,   fontsize = 13,
          color = :black, align = (:left, :top))
    text!(ax_ring,  hw - 0.5, hw - 1.5; text = overlay_str,  fontsize = 12,
          color = :black, align = (:right, :top))

    ax_rg   = Axis(fig[1, 2]; ylabel = "Rg",   xticklabelsvisible = false)
    ax_A2   = Axis(fig[2, 2]; ylabel = "A₂",   xticklabelsvisible = false)
    ax_A3   = Axis(fig[3, 2]; ylabel = "A₃",   xticklabelsvisible = false)
    ax_Kloc = Axis(fig[4, 2]; ylabel = "Kloc", xticklabelsvisible = false)
    ax_b    = Axis(fig[5, 2]; ylabel = "b",    xlabel = "time  (s)")

    for ax in (ax_rg, ax_A2, ax_A3, ax_Kloc, ax_b)
        xlims!(ax, 0.0, t_end)
        vlines!(ax, t_line; color = :gray50, linestyle = :dash, linewidth = 1.5)
    end
    linkxaxes!(ax_rg, ax_A2, ax_A3, ax_Kloc, ax_b)

    lines!(ax_rg,   rg_data;   color = :dodgerblue,   linewidth = 1.5)
    lines!(ax_A2,   A2_data;   color = :mediumpurple,  linewidth = 1.5)
    lines!(ax_A3,   A3_data;   color = :darkorange,    linewidth = 1.5)
    lines!(ax_Kloc, Kloc_data; color = :teal,          linewidth = 1.5)
    lines!(ax_b,    b_data;    color = :green,          linewidth = 1.5)

    display(fig)

    # ── Record loop (replays saved positions — no physics) ────────────────────
    output_path = joinpath(run_dir, output_name)

    record(fig, output_path, 1:n_frames; framerate = framerate) do frame_idx
        r = positions[:, :, frame_idx]

        backbone_obs[]   = make_backbone(r, N)
        xs_pts[]         = r[1, :]
        ys_pts[]         = r[2, :]
        tx, ty           = Observables.force_directions(r, bead_types, N)
        arrow_segs_obs[] = make_segments(r, tx, ty, N, arrow_scale)

        gv = Observables.gyration_vals(r)
        cv = Observables.contour_vals(r)
        overlay_str[] = @sprintf(
            "Gxx=%.3f  Gyy=%.3f\nGxy=%.3f\nA₂=%.3f  A₃=%.3f\nKloc=%.3f\nRg²=%.3f\nb=%.3f",
            gv.Gxx, gv.Gyy, gv.Gxy,
            cv.A2, cv.A3, cv.Kloc,
            gv.Rg^2, gv.b)

        t = Float32((frame_idx - 1) / framerate)
        push!(rg_data[],   Point2f(t, Float32(obs["Rg"][frame_idx])))
        push!(A2_data[],   Point2f(t, Float32(obs["A2"][frame_idx])))
        push!(A3_data[],   Point2f(t, Float32(obs["A3"][frame_idx])))
        push!(Kloc_data[], Point2f(t, Float32(obs["Kloc"][frame_idx])))
        push!(b_data[],    Point2f(t, Float32(obs["b"][frame_idx])))
        notify(rg_data); notify(A2_data); notify(A3_data)
        notify(Kloc_data); notify(b_data)

        t_line[] = [t]
    end

    println("Saved: $output_path")
    return output_path
end

# ── Standalone entry point ────────────────────────────────────────────────────

# ▼▼▼  Edit this line to choose which run to render when using the Run button  ▼▼▼
const RUN_DIR = joinpath(@__DIR__, "..", "results", "run_0021")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

run_dir = isempty(ARGS) ? RUN_DIR : ARGS[1]
render_movie(run_dir)
