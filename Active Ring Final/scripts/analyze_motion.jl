# analyze_motion.jl
# Motion analysis pipeline: compute per-frame/per-window observables and write
# four focused summary figures.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/analyze_motion.jl results/run_0022
#   julia --project scripts/analyze_motion.jl results/run_0022 300
#
# Arguments:
#   arg 1 — run directory path            (required)
#   arg 2 — window size in frames         (default: 150)
#
# Outputs written to <run_dir>/:
#   motion_observables.jld2     per-frame motion time series
#   motion_windows.jld2         per-window summary statistics
#   motion_windows.csv          same, human-readable
#   motion_summary.json         run-level summary and label
#   motion_morphology.png       COM path, Rg, b, shape similarity
#   motion_rotation.png         Ω_shape and Θ_unwrapped
#   motion_activity.png         COM speed and contour circulation
#   motion_windows.png          window label timeline + summary text

using CairoMakie, Statistics, Dates, Printf

const PROJECT_ROOT = joinpath(@__DIR__, "..")

if !isdefined(Main, :EngineRing)
    include(joinpath(PROJECT_ROOT, "src", "EngineRing.jl"))
    using .EngineRing
end
if !isdefined(Main, :IOUtils)
    include(joinpath(PROJECT_ROOT, "src", "IOUtils.jl"))
    using .IOUtils
end
include(joinpath(PROJECT_ROOT, "src", "MotionAnalysis.jl"))
using .MotionAnalysis

# ── Window label colours ──────────────────────────────────────────────────────

const WINDOW_LABEL_COLORS = Dict(
    "arrested"    => RGBAf(0.70, 0.85, 1.00, 0.70),
    "drifting"    => RGBAf(1.00, 0.95, 0.60, 0.70),
    "rotating"    => RGBAf(0.65, 1.00, 0.65, 0.70),
    "oscillatory" => RGBAf(1.00, 0.75, 0.55, 0.70),
    "locked"      => RGBAf(1.00, 0.65, 0.65, 0.70),
    "switching"   => RGBAf(0.85, 0.65, 1.00, 0.70),
)

# ── Run selection ─────────────────────────────────────────────────────────────

# ▼▼▼  Edit this line to choose which run to analyse when using the Run button  ▼▼▼
const DEFAULT_RUN_DIR  = joinpath(PROJECT_ROOT, "results", "run_0502")
const DEFAULT_WIN_SIZE = 150
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

run_dir     = isempty(ARGS) ? DEFAULT_RUN_DIR : abspath(ARGS[1])
window_size = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : DEFAULT_WIN_SIZE

isdir(run_dir) || error("Run directory not found: $run_dir")

# ── Load saved data ───────────────────────────────────────────────────────────

println("Loading saved data from: $run_dir")

isfile(joinpath(run_dir, "traj.jld2"))        || error("traj.jld2 not found — run may be incomplete")
isfile(joinpath(run_dir, "observables.jld2")) || error("observables.jld2 not found")
isfile(joinpath(run_dir, "params.toml"))      || error("params.toml not found")

_, positions, bead_types = IOUtils.load_trajectory(run_dir)
_, morphology_obs        = IOUtils.load_observables(run_dir)
p, rc                    = IOUtils.load_params(run_dir)
meta                     = IOUtils.load_metadata(run_dir)

for key in ("Rg", "b", "A2", "A3", "Kloc", "Gxx", "Gyy", "Gxy")
    haskey(morphology_obs, key) ||
        error("observables.jld2 is missing key \"$key\". Re-run the simulation first.")
end

n_frames  = size(positions, 3)
N         = p.N
framerate = Int(rc["framerate"])
L         = Float64(rc["L"])
dt_frame  = Float64(rc["substeps"]) * p.dt
vtimes    = [(i - 1) / framerate for i in 1:n_frames]
t_end     = vtimes[end]

println("  Frames : $n_frames  (dt_frame=$dt_frame, framerate=$framerate fps)")
println("  Beads  : $N   (L=$L)")
println("  Window : $window_size frames ($(n_frames ÷ window_size) windows)")

# ── Compute per-frame motion observables ─────────────────────────────────────

println("\nComputing per-frame motion observables…")
t0 = time()
frame_obs = MotionAnalysis.compute_motion_frame_obs(
    positions, morphology_obs, dt_frame, L, N)
println(@sprintf("  Done in %.1f s", time() - t0))

# ── Compute per-window summaries ──────────────────────────────────────────────

println("Computing per-window summaries…")

windows     = MotionAnalysis.compute_motion_windows(
    frame_obs, morphology_obs, vtimes, window_size)
morph_label = string(MotionAnalysis.classify_morphology(morphology_obs))
run_summary = MotionAnalysis.motion_run_summary(frame_obs, windows, p, meta;
                                                morph_label)

# ── Save outputs ──────────────────────────────────────────────────────────────

println("Saving outputs…")
IOUtils.save_motion_frame_obs(run_dir, frame_obs)
println("  motion_observables.jld2  written")
IOUtils.save_motion_windows(run_dir, windows)
println("  motion_windows.jld2      written")
println("  motion_windows.csv       written")
IOUtils.save_motion_summary(run_dir, run_summary)
println("  motion_summary.json      written")

# ── Validation ────────────────────────────────────────────────────────────────

println("\n── Validation ──────────────────────────────────────────────────────────")

function check_finite(name, v)
    any(!isfinite, v) && @warn "  WARNING: non-finite values in $name"
    return all(isfinite, v)
end

finite_fields = (
    ("com_speed",        frame_obs.com_speed),
    ("omega_shape",      frame_obs.omega_shape),
    ("circulation",      frame_obs.circulation),
    ("circ_half1",       frame_obs.circ_half1),
    ("circ_half2",       frame_obs.circ_half2),
    ("theta_unwrapped",  frame_obs.theta_unwrapped),
    ("shape_similarity", frame_obs.shape_similarity),
)
all(check_finite(name, arr) for (name, arr) in finite_fields) &&
    println("  Finite check         : all fields OK ✓")

spd = frame_obs.com_speed[2:end]
@printf("  COM speed range      : [%.4f, %.4f]  mean = %.4f\n",
        minimum(spd), maximum(spd), mean(spd))

rel_pct = 100.0 * mean(frame_obs.theta_reliable)
@printf("  theta reliable       : %.1f %%", rel_pct)
rel_pct < 30.0 && print("  ⚠ shape nearly circular — Ω_shape unreliable")
println()

circ = frame_obs.circulation
@printf("  Circulation range    : [%.4f, %.4f]  mean = %.4f\n",
        minimum(circ), maximum(circ), mean(circ))

ss1 = frame_obs.shape_similarity[1]
@printf("  Shape similarity [1] : %.6f  %s\n",
        ss1, abs(ss1 - 1.0) < 1e-6 ? "✓" : "⚠ expected 1.0")

label_counts = Dict{String, Int}()
for w in windows
    l = w["motion_label"]
    label_counts[l] = get(label_counts, l, 0) + 1
end
label_str = join(["$l($(label_counts[l]))" for l in sort(collect(keys(label_counts)))], "  ")
println("  Window labels        : $label_str")
println("  Run motion label     : $(run_summary["motion_label"])")

# ── Shared plot header ────────────────────────────────────────────────────────

motion_label = run_summary["motion_label"]
b_reliable   = MotionAnalysis.MOTION_THRESHOLDS[:b_reliable]

header = "Run: $(meta["run_id"])  f₀=$(p.f0)  κ=$(p.κ)  kBT=$(p.kBT)  N=$(p.N)  " *
         "morphology: $morph_label  |  motion: $motion_label"

# ── Figure 1: Morphology ──────────────────────────────────────────────────────
# COM path | Rg vs time | b vs time | shape similarity vs time

println("\nGenerating motion_morphology.png…")
let
    fig = Figure(size = (1100, 800))
    Label(fig[0, 1:2], "Morphology — " * header; fontsize = 13, tellwidth = false)

    ax_com = Axis(fig[1, 1];
        xlabel = "x", ylabel = "y",
        title  = "COM path (colored by time)",
        aspect = DataAspect(),
    )
    sc = scatter!(ax_com, frame_obs.com_x, frame_obs.com_y;
                  color = vtimes, colormap = :plasma, markersize = 3)
    scatter!(ax_com, [frame_obs.com_x[1]],   [frame_obs.com_y[1]];
             color = :green, markersize = 12, marker = :circle, label = "start")
    scatter!(ax_com, [frame_obs.com_x[end]], [frame_obs.com_y[end]];
             color = :red,   markersize = 12, marker = :star5,  label = "end")
    Colorbar(fig[1, 1][1, 2], sc; label = "video time (s)", width = 10)
    axislegend(ax_com; position = :lt, labelsize = 11)

    ax_Rg = Axis(fig[1, 2]; ylabel = "Rg", xlabel = "video time (s)")
    lines!(ax_Rg, vtimes, morphology_obs["Rg"]; color = :dodgerblue, linewidth = 1.5)

    ax_b = Axis(fig[2, 1]; ylabel = "b (asphericity)", xlabel = "video time (s)")
    lines!(ax_b, vtimes, morphology_obs["b"]; color = :green, linewidth = 1.5)

    ax_shp = Axis(fig[2, 2];
        ylabel = "Shape autocorrelation\nvs frame 1",
        xlabel = "video time (s)",
        title  = "Aligned shape autocorr. (centered, no rotation removed)",
    )
    lines!(ax_shp, vtimes, frame_obs.shape_similarity; color = :teal, linewidth = 1.5)
    hlines!(ax_shp, [0.0]; color = :gray70, linewidth = 0.8, linestyle = :dash)

    save(joinpath(run_dir, "motion_morphology.png"), fig, px_per_unit = 2)
end
println("  motion_morphology.png    written")

# ── Figure 2: Rotation ────────────────────────────────────────────────────────
# Ω_shape vs time | Θ_unwrapped vs time

println("Generating motion_rotation.png…")
let
    fig = Figure(size = (1100, 700))
    Label(fig[0, 1], "Rotation — " * header; fontsize = 13, tellwidth = false)

    ax_omg = Axis(fig[1, 1];
        ylabel = "Ω_shape (rad/t)",
        xlabel = "video time (s)",
    )
    xlims!(ax_omg, 0.0, t_end)
    for t in 2:n_frames
        reliable = frame_obs.theta_reliable[t] > 0.5
        col = reliable ? RGBAf(0.5, 0.0, 0.7, 0.8) : RGBAf(0.7, 0.7, 0.7, 0.4)
        t > 2 && lines!(ax_omg, vtimes[t-1:t], frame_obs.omega_shape[t-1:t];
                        color = col, linewidth = 1.2)
    end
    lines!(ax_omg, Float32[], Float32[];
           color = RGBAf(0.5, 0.0, 0.7, 0.8), linewidth = 1.5, label = "reliable")
    lines!(ax_omg, Float32[], Float32[];
           color = RGBAf(0.7, 0.7, 0.7, 0.6), linewidth = 1.5,
           label = "unreliable (b < $b_reliable)")
    axislegend(ax_omg; position = :rt, labelsize = 10)

    ax_th = Axis(fig[2, 1];
        ylabel = "Θ_shape (rad, unwrapped)",
        xlabel = "video time (s)",
        title  = "Shape angle Θ(t) — unwrapped principal-axis orientation (masked grey where b < $b_reliable)",
    )
    xlims!(ax_th, 0.0, t_end)
    for t in 2:n_frames
        reliable = frame_obs.theta_reliable[t] > 0.5
        col = reliable ? RGBAf(0.18, 0.45, 0.75, 0.85) : RGBAf(0.7, 0.7, 0.7, 0.35)
        lines!(ax_th, vtimes[t-1:t], frame_obs.theta_unwrapped[t-1:t];
               color = col, linewidth = 1.2)
    end
    lines!(ax_th, Float32[], Float32[];
           color = RGBAf(0.18, 0.45, 0.75, 0.85), linewidth = 1.5, label = "reliable")
    lines!(ax_th, Float32[], Float32[];
           color = RGBAf(0.7, 0.7, 0.7, 0.5), linewidth = 1.5, label = "unreliable")
    axislegend(ax_th; position = :rt, labelsize = 10)

    save(joinpath(run_dir, "motion_rotation.png"), fig, px_per_unit = 2)
end
println("  motion_rotation.png      written")

# ── Figure 3: Activity ────────────────────────────────────────────────────────
# COM speed vs time | contour circulation vs time

println("Generating motion_activity.png…")
let
    fig = Figure(size = (1100, 600))
    Label(fig[0, 1], "Activity — " * header; fontsize = 13, tellwidth = false)

    ax_spd = Axis(fig[1, 1]; ylabel = "COM speed", xlabel = "video time (s)")
    xlims!(ax_spd, 0.0, t_end)
    lines!(ax_spd, vtimes, frame_obs.com_speed; color = :darkorange, linewidth = 1.5)

    ax_circ = Axis(fig[2, 1];
        ylabel = "Circulation  (v·t̂)",
        xlabel = "video time (s)",
        title  = "Contour circulation  (solid = all beads,  dashed = CCW half,  dotted = CW half)",
    )
    xlims!(ax_circ, 0.0, t_end)
    hlines!(ax_circ, [0.0]; color = :gray80, linewidth = 1.0)
    lines!(ax_circ, vtimes, frame_obs.circulation; color = :black,      linewidth = 1.8, label = "all")
    lines!(ax_circ, vtimes, frame_obs.circ_half1;  color = :dodgerblue, linewidth = 1.3,
           linestyle = :dash, label = "CCW half")
    lines!(ax_circ, vtimes, frame_obs.circ_half2;  color = :crimson,    linewidth = 1.3,
           linestyle = :dot,  label = "CW half")
    axislegend(ax_circ; position = :rt)

    save(joinpath(run_dir, "motion_activity.png"), fig, px_per_unit = 2)
end
println("  motion_activity.png      written")

# ── Figure 4: Window classification + summary text ───────────────────────────

println("Generating motion_windows.png…")
let
    fig = Figure(size = (1100, 450))
    Label(fig[0, 1], "Classification — " * header; fontsize = 13, tellwidth = false)

    ax_win = Axis(fig[1, 1]; title = "Window labels", xlabel = "video time (s)")
    xlims!(ax_win, 0.0, t_end)
    ylims!(ax_win, 0.0, 1.0)
    hideydecorations!(ax_win)

    for w in windows
        col = get(WINDOW_LABEL_COLORS, w["motion_label"], RGBAf(0.85, 0.85, 0.85, 0.70))
        t1  = Float64(w["t_start"])
        t2  = Float64(w["t_end"])
        poly!(ax_win,
              Point2f[(t1, 0), (t2, 0), (t2, 1), (t1, 1)];
              color = col, strokewidth = 0.5, strokecolor = :gray60)
        text!(ax_win, (t1 + t2) / 2, 0.5;
              text     = w["motion_label"],
              align    = (:center, :center),
              fontsize = 11,
              color    = :black)
    end

    ax_txt = Axis(fig[2, 1])
    hidedecorations!(ax_txt);  hidespines!(ax_txt)
    xlims!(ax_txt, 0, 1);      ylims!(ax_txt, 0, 1)

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
    win_size  = windows[1]["frame_end"] - windows[1]["frame_start"] + 1

    line1 = @sprintf("Morphology: %s  |  Motion: %s  |  COM speed: %.4f ± %.4f",
                     morph_label, motion_label, spd_mean, spd_std)
    line2 = @sprintf("Net displacement: %.3f  |  Path length: %.3f  |  net/path ratio: %.3f  |  Ω_shape (signed): %.4f  |  |Ω_shape| mean: %.4f",
                     net_disp, path_len, net_ratio, omg_signed, omg_abs)
    line3 = @sprintf("Circulation mean: %.4f  |  θ reliable: %.1f %%  |  Windows: %d × %d frames",
                     circ_mn, rel_pct, length(windows), win_size)

    text!(ax_txt, 0.01, 0.75; text = line1, align = (:left, :center),
          fontsize = 10.5, color = :black)
    text!(ax_txt, 0.01, 0.45; text = line2, align = (:left, :center),
          fontsize = 10.5, color = :black)
    text!(ax_txt, 0.01, 0.15; text = line3, align = (:left, :center),
          fontsize = 10.5, color = :black)

    save(joinpath(run_dir, "motion_windows.png"), fig, px_per_unit = 2)
end
println("  motion_windows.png       written")

println("\nMotion analysis complete: $run_dir")
