# run_single.jl
# Run one anti-polar ring simulation and save all outputs.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/run_single.jl
#   julia --project scripts/run_single.jl configs/my_custom.toml
#   julia --project scripts/run_single.jl configs/default.toml run_0002
#
# Arguments (all optional):
#   arg 1 — path to a TOML config file   (default: configs/default.toml)
#   arg 2 — run_id string                (default: auto-generated run_XXXX)
#
# Outputs written to results/<run_id>/:
#   params.toml        exact parameters used
#   metadata.json      seed, timing, Julia version, etc.
#   traj.jld2          positions at every saved frame  (2 × N × n_frames)
#   observables.jld2   Rg, λ1, λ2, b, Gxx, Gyy, Gxy time series
#   summary.png        static 4-panel figure (produced by summarize_run.jl)

using TOML
using Random
using Dates
using Printf

# ── Locate project root (one level up from scripts/) ────────────────────────
const PROJECT_ROOT = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT, "src", "EngineRing.jl"))
using .EngineRing

include(joinpath(PROJECT_ROOT, "src", "Observables.jl"))
using .Observables

include(joinpath(PROJECT_ROOT, "src", "InitialConditions.jl"))
using .InitialConditions

include(joinpath(PROJECT_ROOT, "src", "IOUtils.jl"))
using .IOUtils

# ── Parse CLI arguments ──────────────────────────────────────────────────────
config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(PROJECT_ROOT, "configs", "default.toml")
cfg         = TOML.parsefile(config_path)

ep  = cfg["params"]
rc  = cfg["run"]

# ── Build EngineRing.Params from config ──────────────────────────────────────
p = EngineRing.Params(
    N         = ep["N"],
    l         = ep["l"],
    ks        = ep["ks"],
    κ         = ep["kappa"],
    theta_max = ep["theta_max"],
    theta_buf = ep["theta_buf"],
    k_stop    = ep["k_stop"],
    sigma     = ep["sigma"],
    epsilon   = ep["epsilon"],
    topology  = Symbol(ep["topology"]),
    f0        = ep["f0"],
    mu        = ep["mu"],
    kBT       = ep["kBT"],
    dt        = ep["dt"],
)

# ── Run config ───────────────────────────────────────────────────────────────
frames    = rc["frames"]
substeps  = rc["substeps"]
framerate = rc["framerate"]
L         = rc["L"]
seed      = rc["seed"]
outdir    = rc["outdir"]

# ── Run ID: from CLI arg 2, or auto-increment based on existing dirs ─────────
function next_run_id(base::String)
    existing = filter(d -> occursin(r"^run_\d+$", d),
                      readdir(base, join = false))
    nums = [parse(Int, d[5:end]) for d in existing]
    n    = isempty(nums) ? 1 : maximum(nums) + 1
    return @sprintf("run_%04d", n)
end

using Printf
base_dir = joinpath(PROJECT_ROOT, outdir)
mkpath(base_dir)
run_id   = length(ARGS) >= 2 ? ARGS[2] : next_run_id(base_dir)
run_dir  = IOUtils.init_run_dir(base_dir, run_id)

println("Run ID  : $run_id")
println("Output  : $run_dir")
println("Frames  : $frames  ×  $substeps substeps  (framerate = $framerate fps)")
println("Params  : N=$(p.N)  f0=$(p.f0)  κ=$(p.κ)  kBT=$(p.kBT)  dt=$(p.dt)")

# ── Save params and metadata BEFORE the loop ─────────────────────────────────
run_cfg = Dict{String, Any}(rc)   # plain dict copy for IOUtils
IOUtils.save_params(run_dir, p, run_cfg)

meta = Dict(
    "run_id"       => run_id,
    "seed"         => seed,
    "timestamp"    => string(now()),
    "julia_version" => string(VERSION),
    "config_file"  => abspath(config_path),
    "n_frames"     => frames,
    "substeps"     => substeps,
    "framerate"    => framerate,
    "L"            => L,
    "bead_layout"  => rc["bead_layout"],
    "wrap_applied" => true,
)
IOUtils.save_metadata(run_dir, meta)

# ── Initial condition ────────────────────────────────────────────────────────
Random.seed!(seed)

bead_types = InitialConditions.antipolar_bead_types(p.N)
state      = InitialConditions.perfect_circle_ic(p, bead_types)

# ── Pre-allocate output arrays ───────────────────────────────────────────────
times     = Vector{Float64}(undef, frames)
positions = Array{Float64, 3}(undef, 2, p.N, frames)

obs_Rg   = Vector{Float64}(undef, frames)
obs_λ1   = Vector{Float64}(undef, frames)
obs_λ2   = Vector{Float64}(undef, frames)
obs_b    = Vector{Float64}(undef, frames)
obs_Gxx  = Vector{Float64}(undef, frames)
obs_Gyy  = Vector{Float64}(undef, frames)
obs_Gxy  = Vector{Float64}(undef, frames)
obs_A2   = Vector{Float64}(undef, frames)
obs_A3   = Vector{Float64}(undef, frames)
obs_Kloc = Vector{Float64}(undef, frames)

# ── Simulation loop ──────────────────────────────────────────────────────────
dt_frame = substeps * p.dt     # simulation time per saved frame
t_sim    = 0.0

println("\nRunning simulation…")
t_wall_start = time()

for frame in 1:frames
    # Physics
    for _ in 1:substeps
        EngineRing.step!(state, p)
    end
    global t_sim += dt_frame

    # Periodic wrapping: shift the whole ring back into [-L/2, L/2]² if needed.
    # This does not alter inter-bead distances or forces; it is purely a
    # coordinate shift applied after all physics for that frame are done.
    hw = L / 2.0
    min_x = minimum(@view state.r[1, :])
    max_x = maximum(@view state.r[1, :])
    min_y = minimum(@view state.r[2, :])
    max_y = maximum(@view state.r[2, :])
    min_x >  hw && (state.r[1, :] .-= L)
    max_x < -hw && (state.r[1, :] .+= L)
    min_y >  hw && (state.r[2, :] .-= L)
    max_y < -hw && (state.r[2, :] .+= L)

    # Record
    times[frame]          = t_sim
    positions[:, :, frame] = state.r

    obs = Observables.compute_frame_obs(state.r)
    obs_Rg[frame]   = obs.Rg
    obs_λ1[frame]   = obs.λ1
    obs_λ2[frame]   = obs.λ2
    obs_b[frame]    = obs.b
    obs_Gxx[frame]  = obs.Gxx
    obs_Gyy[frame]  = obs.Gyy
    obs_Gxy[frame]  = obs.Gxy
    obs_A2[frame]   = obs.A2
    obs_A3[frame]   = obs.A3
    obs_Kloc[frame] = obs.Kloc

    # Progress indicator every 10 %
    if frame % max(1, frames ÷ 10) == 0
        pct = round(Int, 100 * frame / frames)
        println("  $pct %  (frame $frame / $frames,  t_sim = $(round(t_sim, digits=4)))")
    end
end

t_wall = round(time() - t_wall_start, digits=1)
println("Done in $(t_wall) s wall time.")

# ── Save trajectory and observables ─────────────────────────────────────────
println("\nSaving outputs…")

IOUtils.save_trajectory(run_dir, times, positions, bead_types)

obs_dict = Dict{String, Vector{Float64}}(
    "Rg"      => obs_Rg,
    "lambda1" => obs_λ1,
    "lambda2" => obs_λ2,
    "b"       => obs_b,
    "Gxx"     => obs_Gxx,
    "Gyy"     => obs_Gyy,
    "Gxy"     => obs_Gxy,
    "A2"      => obs_A2,
    "A3"      => obs_A3,
    "Kloc"    => obs_Kloc,
)
IOUtils.save_observables(run_dir, times, obs_dict)

println("  traj.jld2        written")
println("  observables.jld2 written")

# ── Generate static summary ──────────────────────────────────────────────────
println("\nGenerating summary…")
include(joinpath(@__DIR__, "summarize_run.jl"))
summarize_run(run_dir)
println("  summary.png      written")

println("\nRun complete: $run_dir")
