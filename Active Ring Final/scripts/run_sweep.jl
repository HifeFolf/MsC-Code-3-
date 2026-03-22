# run_sweep.jl
# Run a parameter sweep defined in a sweep TOML file.
# All runs execute sequentially in one Julia session (avoids recompilation).
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/run_sweep.jl
#   julia --project scripts/run_sweep.jl configs/sweep.toml
#
# Each parameter combination produces one run directory under results/.
# A sweep index CSV is written to results/sweep_index.csv when done.

using TOML
using Printf
using Dates

const PROJECT_ROOT_SW = joinpath(@__DIR__, "..")

# ── Load all shared modules once ─────────────────────────────────────────────
include(joinpath(PROJECT_ROOT_SW, "src", "EngineRing.jl"))
using .EngineRing

include(joinpath(PROJECT_ROOT_SW, "src", "Observables.jl"))
using .Observables

include(joinpath(PROJECT_ROOT_SW, "src", "InitialConditions.jl"))
using .InitialConditions

include(joinpath(PROJECT_ROOT_SW, "src", "IOUtils.jl"))
using .IOUtils

include(joinpath(PROJECT_ROOT_SW, "scripts", "summarize_run.jl"))

# ── Parse sweep config ────────────────────────────────────────────────────────
sweep_config_path = length(ARGS) >= 1 ? ARGS[1] :
                    joinpath(PROJECT_ROOT_SW, "configs", "sweep.toml")

sw_cfg    = TOML.parsefile(sweep_config_path)
base_path = joinpath(PROJECT_ROOT_SW, sw_cfg["sweep"]["base"])
base_cfg  = TOML.parsefile(base_path)

sweep_run_overrides    = get(sw_cfg["sweep"], "run",    Dict())
sweep_param_overrides  = get(sw_cfg["sweep"], "params", Dict())

# ── Cartesian product of sweep parameters ────────────────────────────────────
param_names  = collect(keys(sweep_param_overrides))
param_values = [sweep_param_overrides[k] for k in param_names]

function cartesian_product(lists)
    result = [[]]
    for lst in lists
        result = [vcat(r, [v]) for r in result for v in lst]
    end
    return result
end

combos = cartesian_product(param_values)
n_runs = length(combos)

println("Sweep: $(sweep_config_path)")
println("Base config: $(base_path)")
println("Varying: $(join(param_names, ", "))")
println("Total runs: $n_runs\n")

# ── Shared output directory ───────────────────────────────────────────────────
base_dir = joinpath(PROJECT_ROOT_SW, base_cfg["run"]["outdir"])
mkpath(base_dir)

function next_run_id(base::String)
    existing = filter(d -> occursin(r"^run_\d+$", d), readdir(base, join = false))
    nums = [parse(Int, d[5:end]) for d in existing]
    n    = isempty(nums) ? 1 : maximum(nums) + 1
    return @sprintf("run_%04d", n)
end

# ── Sweep index (written after all runs complete) ─────────────────────────────
sweep_records = Vector{Dict{String, Any}}()

# ── Run loop ──────────────────────────────────────────────────────────────────
for (combo_idx, combo) in enumerate(combos)
    # Build per-combo overrides
    override = Dict(param_names[i] => combo[i] for i in eachindex(param_names))

    println("=" ^ 60)
    println("Combo $combo_idx / $n_runs : $override")
    println("=" ^ 60)

    # Merge: base params ← sweep run overrides ← per-combo param overrides
    ep = merge(base_cfg["params"], Dict(k => v for (k, v) in override
                                        if k in keys(base_cfg["params"])))
    rc = merge(base_cfg["run"],    Dict(string(k) => v
                                        for (k, v) in sweep_run_overrides))

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

    frames    = rc["frames"]
    substeps  = rc["substeps"]
    framerate = rc["framerate"]
    L         = rc["L"]
    seed      = rc["seed"]

    run_id  = next_run_id(base_dir)
    run_dir = IOUtils.init_run_dir(base_dir, run_id)

    # Save params and metadata before running
    IOUtils.save_params(run_dir, p, Dict{String,Any}(rc))
    meta = Dict(
        "run_id"        => run_id,
        "seed"          => seed,
        "timestamp"     => string(now()),
        "julia_version" => string(VERSION),
        "config_file"   => abspath(sweep_config_path),
        "sweep_combo"   => override,
        "combo_index"   => combo_idx,
        "n_frames"      => frames,
        "substeps"      => substeps,
        "framerate"     => framerate,
        "L"             => L,
        "bead_layout"   => rc["bead_layout"],
        "wrap_applied"  => true,
    )
    IOUtils.save_metadata(run_dir, meta)

    using Random
    Random.seed!(seed)

    bead_types = InitialConditions.antipolar_bead_types(p.N)
    state      = InitialConditions.perfect_circle_ic(p, bead_types)

    times     = Vector{Float64}(undef, frames)
    positions = Array{Float64, 3}(undef, 2, p.N, frames)
    obs_Rg    = Vector{Float64}(undef, frames)
    obs_λ1    = Vector{Float64}(undef, frames)
    obs_λ2    = Vector{Float64}(undef, frames)
    obs_b     = Vector{Float64}(undef, frames)
    obs_Gxx   = Vector{Float64}(undef, frames)
    obs_Gyy   = Vector{Float64}(undef, frames)
    obs_Gxy   = Vector{Float64}(undef, frames)

    dt_frame = substeps * p.dt
    t_sim    = 0.0
    hw       = L / 2.0

    t_wall_start = time()
    for frame in 1:frames
        for _ in 1:substeps
            EngineRing.step!(state, p)
        end
        t_sim += dt_frame

        min_x = minimum(@view state.r[1, :])
        max_x = maximum(@view state.r[1, :])
        min_y = minimum(@view state.r[2, :])
        max_y = maximum(@view state.r[2, :])
        min_x >  hw && (state.r[1, :] .-= L)
        max_x < -hw && (state.r[1, :] .+= L)
        min_y >  hw && (state.r[2, :] .-= L)
        max_y < -hw && (state.r[2, :] .+= L)

        times[frame]           = t_sim
        positions[:, :, frame] = state.r

        obs = Observables.compute_frame_obs(state.r)
        obs_Rg[frame]  = obs.Rg
        obs_λ1[frame]  = obs.λ1
        obs_λ2[frame]  = obs.λ2
        obs_b[frame]   = obs.b
        obs_Gxx[frame] = obs.Gxx
        obs_Gyy[frame] = obs.Gyy
        obs_Gxy[frame] = obs.Gxy

        if frame % max(1, frames ÷ 10) == 0
            println("  $(round(Int, 100*frame/frames)) %  (frame $frame / $frames)")
        end
    end
    t_wall = round(time() - t_wall_start, digits = 1)
    println("  Done in $(t_wall) s")

    IOUtils.save_trajectory(run_dir, times, positions, bead_types)
    IOUtils.save_observables(run_dir, times, Dict{String, Vector{Float64}}(
        "Rg" => obs_Rg, "lambda1" => obs_λ1, "lambda2" => obs_λ2,
        "b"  => obs_b,  "Gxx" => obs_Gxx,    "Gyy" => obs_Gyy, "Gxy" => obs_Gxy))

    summarize_run(run_dir)
    println("  Saved: $run_dir\n")

    push!(sweep_records, merge(override, Dict("run_id" => run_id, "wall_s" => t_wall)))
end

# ── Write sweep index CSV ─────────────────────────────────────────────────────
index_path = joinpath(base_dir, "sweep_index.csv")
open(index_path, "w") do io
    header = join(["run_id"; param_names; "wall_s"], ",")
    println(io, header)
    for rec in sweep_records
        row = join([rec["run_id"]; [string(rec[k]) for k in param_names]; string(rec["wall_s"])], ",")
        println(io, row)
    end
end

println("\nSweep complete.")
println("Index: $index_path")
println("Total runs: $n_runs")
