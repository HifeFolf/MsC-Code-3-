# motion_summary_plot.jl
# Generate a static combined morphology + motion summary PNG from a saved run.
# Uses CairoMakie (headless — no display or window required).
#
# Usage (standalone):
#
#   julia --project scripts/motion_summary_plot.jl results/run_0022
#
# Or called from analyze_motion.jl:
#
#   include("scripts/motion_summary_plot.jl")
#   motion_summary_plot("results/run_0022")
#
# Output:  <run_dir>/motion_summary.png

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_MSP = joinpath(@__DIR__, "..")

if !isdefined(Main, :EngineRing)
    include(joinpath(PROJECT_ROOT_MSP, "src", "EngineRing.jl"))
    using .EngineRing
end
if !isdefined(Main, :IOUtils)
    include(joinpath(PROJECT_ROOT_MSP, "src", "IOUtils.jl"))
    using .IOUtils
end
include(joinpath(PROJECT_ROOT_MSP, "src", "MotionAnalysis.jl"))
using .MotionAnalysis

# ── Colour map for window labels ──────────────────────────────────────────────

const WINDOW_LABEL_COLORS = Dict(
    "arrested"    => RGBAf(0.70, 0.85, 1.00, 0.70),   # light blue
    "drifting"    => RGBAf(1.00, 0.95, 0.60, 0.70),   # light yellow
    "rotating"    => RGBAf(0.65, 1.00, 0.65, 0.70),   # light green
    "oscillatory" => RGBAf(1.00, 0.75, 0.55, 0.70),   # light orange
    "locked"      => RGBAf(1.00, 0.65, 0.65, 0.70),   # light red
    "switching"   => RGBAf(0.85, 0.65, 1.00, 0.70),   # light purple
)

"""
    motion_summary_plot(run_dir; output_name="motion_summary.png")

Load all motion and morphology data from `run_dir` and write a 6-row
static summary figure to `run_dir/<output_name>`.

Panels:
  Row 1: COM trajectory (x,y colored by time) | Rg vs time | b vs time
  Row 2: COM speed vs time | Ω_shape vs time | shape similarity vs time
  Row 3: Θ_unwrapped vs time                  [full width]
  Row 4: Circulation (all + half1 + half2)    [full width]
  Row 5: Window label timeline                 [full width]
  Row 6: Run summary text                      [full width]
"""
function motion_summary_plot(run_dir::String; output_name::String = "motion_summary.png")

    # ── Load data ────────────────────────────────────────────────────────────
    for fname in ("motion_observables.jld2", "motion_windows.jld2",
                  "observables.jld2", "params.toml", "metadata.json")
        isfile(joinpath(run_dir, fname)) ||
            error("$fname not found in $run_dir — run analyze_motion.jl first")
    end

    frame_obs   = IOUtils.load_motion_frame_obs(run_dir)
    windows     = IOUtils.load_motion_windows(run_dir)
    _, morph    = IOUtils.load_observables(run_dir)
    p, rc       = IOUtils.load_params(run_dir)
    meta        = IOUtils.load_metadata(run_dir)

    n_frames  = length(frame_obs.com_speed)
    framerate = Int(rc["framerate"])
    vtimes    = [(i - 1) / framerate for i in 1:n_frames]
    t_end     = vtimes[end]

    # ── Labels ───────────────────────────────────────────────────────────────
    # Motion label: majority vote over window labels
    motion_label = let
        counts = Dict{String, Int}()
        for w in windows
            l = w["motion_label"]
            counts[l] = get(counts, l, 0) + 1
        end
        argmax(counts)
    end

    # Morphology label: classify from the already-loaded morphology observables.
    morph_label = string(MotionAnalysis.classify_morphology(morph))

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = Figure(size = (1400, 1300))

    title_str = "Motion analysis — Run: $(meta["run_id"])  " *
                "f₀=$(p.f0)  κ=$(p.κ)  kBT=$(p.kBT)  N=$(p.N)  " *
                "morphology: $morph_label  |  motion: $motion_label"
    Label(fig[0, 1:3], title_str; fontsize = 14, tellwidth = false)

    # ── Row 1: COM path | Rg | b ──────────────────────────────────────────────
    ax_com = Axis(fig[1, 1];
        xlabel = "x", ylabel = "y",
        title  = "COM path (colored by time)",
        aspect = DataAspect(),
    )
    sc = scatter!(ax_com, frame_obs.com_x, frame_obs.com_y;
                  color = vtimes, colormap = :plasma, markersize = 3)
    scatter!(ax_com, [frame_obs.com_x[1]],   [frame_obs.com_y[1]];
             color = :green, markersize = 12, marker = :circle,   label = "start")
    scatter!(ax_com, [frame_obs.com_x[end]], [frame_obs.com_y[end]];
             color = :red,   markersize = 12, marker = :star5,    label = "end")
    Colorbar(fig[1, 1][1, 2], sc; label = "video time (s)", width = 10)
    axislegend(ax_com; position = :lt, labelsize = 11)

    ax_Rg = Axis(fig[1, 2]; ylabel = "Rg",             xlabel = "video time (s)")
    ax_b  = Axis(fig[1, 3]; ylabel = "b (asphericity)", xlabel = "video time (s)")
    lines!(ax_Rg, vtimes, morph["Rg"]; color = :dodgerblue, linewidth = 1.5)
    lines!(ax_b,  vtimes, morph["b"];  color = :green,       linewidth = 1.5)

    # ── Row 2: COM speed | Ω_shape | shape similarity ─────────────────────────
    ax_spd = Axis(fig[2, 1]; ylabel = "COM speed",       xlabel = "video time (s)")
    ax_omg = Axis(fig[2, 2]; ylabel = "Ω_shape (rad/t)", xlabel = "video time (s)")
    ax_shp = Axis(fig[2, 3];
        ylabel = "Shape autocorrelation\nvs frame 1",
        xlabel = "video time (s)",
        title  = "Aligned shape autocorr. (centered, no rotation removed)",
    )

    lines!(ax_spd, vtimes, frame_obs.com_speed; color = :darkorange, linewidth = 1.5)
    lines!(ax_shp, vtimes, frame_obs.shape_similarity; color = :teal, linewidth = 1.5)
    hlines!(ax_shp, [0.0]; color = :gray70, linewidth = 0.8, linestyle = :dash)

    # Ω_shape: colour-coded by theta reliability
    for t in 2:n_frames
        reliable = frame_obs.theta_reliable[t] > 0.5
        col      = reliable ? RGBAf(0.5, 0.0, 0.7, 0.8) : RGBAf(0.7, 0.7, 0.7, 0.4)
        t > 2 && lines!(ax_omg, vtimes[t-1:t], frame_obs.omega_shape[t-1:t];
                        color = col, linewidth = 1.2)
    end
    lines!(ax_omg, Float32[], Float32[]; color = RGBAf(0.5, 0.0, 0.7, 0.8),
           linewidth = 1.5, label = "reliable")
    lines!(ax_omg, Float32[], Float32[]; color = RGBAf(0.7, 0.7, 0.7, 0.6),
           linewidth = 1.5, label = "unreliable (b < $(MotionAnalysis.MOTION_THRESHOLDS[:b_reliable]))")
    axislegend(ax_omg; position = :rt, labelsize = 10)

    # ── Row 3: Θ_unwrapped ────────────────────────────────────────────────────
    ax_th = Axis(fig[3, 1:3];
        ylabel = "Θ_shape (rad, unwrapped)",
        xlabel = "video time (s)",
        title  = "Shape angle Θ(t) — unwrapped principal-axis orientation  " *
                 "(masked grey where b < $(MotionAnalysis.MOTION_THRESHOLDS[:b_reliable]))",
    )
    xlims!(ax_th, 0.0, t_end)

    # Plot reliable and unreliable segments as separate coloured/grey runs
    for t in 2:n_frames
        reliable = frame_obs.theta_reliable[t] > 0.5
        col = reliable ? RGBAf(0.18, 0.45, 0.75, 0.85) : RGBAf(0.7, 0.7, 0.7, 0.35)
        lines!(ax_th, vtimes[t-1:t], frame_obs.theta_unwrapped[t-1:t];
               color = col, linewidth = 1.2)
    end
    # Dummy lines for legend
    lines!(ax_th, Float32[], Float32[]; color = RGBAf(0.18, 0.45, 0.75, 0.85),
           linewidth = 1.5, label = "reliable")
    lines!(ax_th, Float32[], Float32[]; color = RGBAf(0.7, 0.7, 0.7, 0.5),
           linewidth = 1.5, label = "unreliable")
    axislegend(ax_th; position = :rt, labelsize = 10)

    # ── Row 4: Contour circulation ────────────────────────────────────────────
    ax_circ = Axis(fig[4, 1:3];
        ylabel = "Circulation  (v·t̂)",
        xlabel = "video time (s)",
        title  = "Contour circulation  (solid = all beads,  dashed = CCW half,  dotted = CW half)",
    )
    xlims!(ax_circ, 0.0, t_end)
    hlines!(ax_circ, [0.0]; color = :gray80, linewidth = 1.0)
    lines!(ax_circ, vtimes, frame_obs.circulation; color = :black,      linewidth = 1.8, label = "all")
    lines!(ax_circ, vtimes, frame_obs.circ_half1;  color = :dodgerblue, linewidth = 1.3,
           linestyle = :dash,  label = "CCW half")
    lines!(ax_circ, vtimes, frame_obs.circ_half2;  color = :crimson,    linewidth = 1.3,
           linestyle = :dot,   label = "CW half")
    axislegend(ax_circ; position = :rt)

    # ── Row 5: Window label timeline ──────────────────────────────────────────
    ax_win = Axis(fig[5, 1:3];
        title  = "Window labels",
        xlabel = "video time (s)",
    )
    xlims!(ax_win, 0.0, t_end)
    ylims!(ax_win, 0.0, 1.0)
    hideydecorations!(ax_win)

    for w in windows
        col   = get(WINDOW_LABEL_COLORS, w["motion_label"], RGBAf(0.85, 0.85, 0.85, 0.70))
        t1    = Float64(w["t_start"])
        t2    = Float64(w["t_end"])
        poly!(ax_win,
              Point2f[(t1, 0), (t2, 0), (t2, 1), (t1, 1)];
              color = col, strokewidth = 0.5, strokecolor = :gray60)
        text!(ax_win, (t1 + t2) / 2, 0.5;
              text      = w["motion_label"],
              align     = (:center, :center),
              fontsize  = 11,
              color     = :black)
    end

    # ── Row 6: Summary text ────────────────────────────────────────────────────
    ax_txt = Axis(fig[6, 1:3])
    hidedecorations!(ax_txt)
    hidespines!(ax_txt)
    xlims!(ax_txt, 0, 1);  ylims!(ax_txt, 0, 1)

    spd_mean   = mean(frame_obs.com_speed[2:end])
    spd_std    = std(frame_obs.com_speed[2:end])
    omg_signed = mean(frame_obs.omega_shape[2:end])
    omg_abs    = mean(abs.(frame_obs.omega_shape[2:end]))
    circ_mn    = mean(frame_obs.circulation)
    rel_pct    = 100.0 * mean(frame_obs.theta_reliable)

    x_uw = frame_obs.com_x_unwrapped;  y_uw = frame_obs.com_y_unwrapped
    Δx   = x_uw[end] - x_uw[1];       Δy   = y_uw[end] - y_uw[1]
    net_disp = sqrt(Δx^2 + Δy^2)
    path_len = sum(sqrt((x_uw[t] - x_uw[t-1])^2 + (y_uw[t] - y_uw[t-1])^2)
                   for t in 2:lastindex(x_uw))
    net_ratio = path_len > 1e-10 ? net_disp / path_len : 0.0

    win_size = windows[1]["frame_end"] - windows[1]["frame_start"] + 1

    line1 = @sprintf("Morphology: %s  |  Motion: %s  |  COM speed: %.4f ± %.4f",
                     morph_label, motion_label, spd_mean, spd_std)
    line2 = @sprintf("Net displacement: %.3f  |  Path length: %.3f  |  net/path ratio: %.3f  |  Ω_shape (signed): %.4f  |  |Ω_shape| mean: %.4f",
                     net_disp, path_len, net_ratio, omg_signed, omg_abs)
    line3 = @sprintf("Circulation mean: %.4f  |  θ reliable: %.1f %%  |  Windows: %d × %d frames",
                     circ_mn, rel_pct, length(windows), win_size)

    text!(ax_txt, 0.01, 0.72; text = line1, align = (:left, :center),
          fontsize = 10.5, color = :black)
    text!(ax_txt, 0.01, 0.42; text = line2, align = (:left, :center),
          fontsize = 10.5, color = :black)
    text!(ax_txt, 0.01, 0.12; text = line3, align = (:left, :center),
          fontsize = 10.5, color = :black)

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = joinpath(run_dir, output_name)
    save(output_path, fig, px_per_unit = 2)
    return output_path
end

# ── Standalone entry point ────────────────────────────────────────────────────

# ▼▼▼  Edit this line to choose which run to plot when using the Run button  ▼▼▼
const RUN_DIR_MSP = joinpath(@__DIR__, "..", "results", "run_0002")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

if abspath(PROGRAM_FILE) == @__FILE__
    run_dir_msp = isempty(ARGS) ? RUN_DIR_MSP : ARGS[1]
    path = motion_summary_plot(run_dir_msp)
    println("Saved: $path")
end
