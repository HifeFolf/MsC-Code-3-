# analyze_motion.jl
# Stage 1 motion analysis: compute per-frame and per-window motion observables
# from a saved run directory. No physics is re-run.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/analyze_motion.jl results/run_0022
#   julia --project scripts/analyze_motion.jl results/run_0022 300
#
# Arguments:
#   arg 1 — run directory path               (required)
#   arg 2 — window size in frames            (default: 150)
#
# Outputs written to <run_dir>/:
#   motion_observables.jld2    per-frame motion time series
#   motion_windows.jld2        per-window summary statistics
#   motion_windows.csv         same, human-readable
#   motion_summary.json        run-level summary and provisional label
#   motion_summary.png         combined morphology + motion figure

using Statistics, Dates, Printf

const PROJECT_ROOT_AM = joinpath(@__DIR__, "..")

if !isdefined(Main, :EngineRing)
    include(joinpath(PROJECT_ROOT_AM, "src", "EngineRing.jl"))
    using .EngineRing
end
if !isdefined(Main, :IOUtils)
    include(joinpath(PROJECT_ROOT_AM, "src", "IOUtils.jl"))
    using .IOUtils
end
include(joinpath(PROJECT_ROOT_AM, "src", "MotionAnalysis.jl"))
using .MotionAnalysis

# ── Run selection ─────────────────────────────────────────────────────────────

# ▼▼▼  Edit this line to choose which run to analyse when using the Run button  ▼▼▼
const RUN_DIR_AM      = joinpath(PROJECT_ROOT_AM, "results", "run_0002")
const WINDOW_SIZE_AM  = 150   # frames per analysis window
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

run_dir     = isempty(ARGS) ? RUN_DIR_AM     : abspath(ARGS[1])
window_size = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : WINDOW_SIZE_AM

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

# Check that the new observables exist in this run
for key in ("Rg", "b", "A2", "A3", "Kloc", "Gxx", "Gyy", "Gxy")
    haskey(morphology_obs, key) ||
        error("observables.jld2 is missing key \"$key\". Re-run the simulation first.")
end

n_frames  = size(positions, 3)
N         = p.N
framerate = Int(rc["framerate"])
L         = Float64(rc["L"])
dt_frame  = Float64(rc["substeps"]) * p.dt
vtimes    = [(i - 1) / framerate for i in 1:n_frames]   # video seconds

println("  Frames    : $n_frames   (dt_frame = $dt_frame,  framerate = $framerate fps)")
println("  Beads     : $N   (L = $L)")
println("  Window    : $window_size frames  ($(n_frames ÷ window_size) windows)")

# ── Compute per-frame motion observables ─────────────────────────────────────

println("\nComputing per-frame motion observables…")
t0 = time()

frame_obs = MotionAnalysis.compute_motion_frame_obs(
    positions, morphology_obs, dt_frame, L, N)

println(@sprintf("  Done in %.1f s", time() - t0))

# ── Compute per-window summaries ──────────────────────────────────────────────

println("Computing per-window summaries…")

windows = MotionAnalysis.compute_motion_windows(
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

# ── Validation summary ────────────────────────────────────────────────────────

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
all_ok = all(check_finite(name, arr) for (name, arr) in finite_fields)
all_ok && println("  Finite check         : all fields OK ✓")

# COM speed
spd = frame_obs.com_speed[2:end]
@printf("  COM speed range      : [%.4f, %.4f]  mean = %.4f\n",
        minimum(spd), maximum(spd), mean(spd))

# theta reliability
rel_pct = 100.0 * mean(frame_obs.theta_reliable)
@printf("  theta reliable       : %.1f %% of frames", rel_pct)
rel_pct < 30.0 && print("  ⚠ shape nearly circular — Ω_shape unreliable")
println()

# Circulation
circ = frame_obs.circulation
@printf("  Circulation range    : [%.4f, %.4f]  mean = %.4f\n",
        minimum(circ), maximum(circ), mean(circ))

# Shape similarity at frame 1 (should be 1.0)
ss1 = frame_obs.shape_similarity[1]
@printf("  Shape similarity [1] : %.6f  %s\n",
        ss1, abs(ss1 - 1.0) < 1e-6 ? "✓" : "⚠ expected 1.0")

# Window labels
label_counts = Dict{String, Int}()
for w in windows
    l = w["motion_label"]
    label_counts[l] = get(label_counts, l, 0) + 1
end
label_str = join(["$l($(label_counts[l]))" for l in sort(collect(keys(label_counts)))], "  ")
println("  Window labels        : $label_str")
println("  Run motion label     : $(run_summary["motion_label"])")

# ── Generate motion summary plot ──────────────────────────────────────────────

println("\nGenerating motion_summary.png…")
include(joinpath(@__DIR__, "motion_summary_plot.jl"))
motion_summary_plot(run_dir)
println("  motion_summary.png   written")

println("\nMotion analysis complete: $run_dir")
