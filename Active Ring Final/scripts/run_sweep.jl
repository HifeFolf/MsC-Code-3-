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
using Random
using JSON3

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
#
# Optional flags (may appear anywhere in ARGS):
#   --resume          skip combos whose combo_index already appears in an
#                     existing run's metadata.json in the output directory
#   --skip-existing   skip combos whose (param → value) dict matches an
#                     existing run's params.toml (safer if config changed)
#   --append          allow adding runs to an output folder that already has
#                     some (normally refused, to avoid mixing two datasets)
#
# With none of these, running against a non-empty output folder is an ERROR:
# rename/move the old folder first for a clean re-run.
#
# Examples:
#   julia --project scripts/run_sweep.jl configs/sweep_large_log.toml --resume
#   julia --project scripts/run_sweep.jl configs/sweep_large_log.toml --skip-existing

flag_resume        = "--resume"        in ARGS
flag_skip_existing = "--skip-existing" in ARGS
flag_append        = "--append"        in ARGS
positional_args    = filter(a -> !startswith(a, "--"), ARGS)

# ▼▼▼  Config used when you press Run with no arguments — edit this line  ▼▼▼
const DEFAULT_SWEEP_CONFIG = "sweep_N17_circle.toml"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

sweep_config_path = length(positional_args) >= 1 ? positional_args[1] :
                    joinpath(PROJECT_ROOT_SW, "configs", DEFAULT_SWEEP_CONFIG)

sw_cfg    = TOML.parsefile(sweep_config_path)
base_path = joinpath(PROJECT_ROOT_SW, sw_cfg["sweep"]["base"])
base_cfg  = TOML.parsefile(base_path)

sweep_run_overrides    = get(sw_cfg["sweep"], "run",    Dict())
sweep_param_overrides  = get(sw_cfg["sweep"], "params", Dict())

# ── Parameter expansion (supports linspace / logspace table syntax) ───────────
#
#   Plain array  :  kappa = [1.0, 2.0, 5.0]
#
#   [sweep.params.kappa]          [sweep.params.kappa]
#   linspace = [0.0, 50.0, 100]   logspace = [0.1, 50.0, 50]
#   prepend_zero = true            prepend_zero = false   # optional, default false

function expand_param(val)
    val isa Vector && return Float64.(val)
    val isa Dict   || error("Sweep param must be an array or table; got $(typeof(val))")
    if haskey(val, "linspace")
        s, e, n = val["linspace"]
        pts = collect(range(Float64(s), Float64(e); length = Int(n)))
    elseif haskey(val, "logspace")
        s, e, n = val["logspace"]
        pts = exp.(range(log(Float64(s)), log(Float64(e)); length = Int(n)))
    else
        error("Sweep parameter table must have a 'linspace' or 'logspace' key")
    end
    get(val, "prepend_zero", false) && pushfirst!(pts, 0.0)
    return pts
end

# ── Cartesian product of sweep parameters ────────────────────────────────────
param_names  = collect(keys(sweep_param_overrides))
param_values = [expand_param(sweep_param_overrides[k]) for k in param_names]

function cartesian_product(lists)
    result = [[]]
    for lst in lists
        result = [vcat(r, [v]) for r in result for v in lst]
    end
    return result
end

combos = cartesian_product(param_values)

# ── IC spec list (optional [[sweep.ics]] array of tables) ────────────────────
# Each entry is a dict of [run] overrides (bead_layout, ic_snapshot_dir, …).
# When present, IC varies as an outer dimension: total runs = |ics| × |combos|.
# When absent, [sweep.run].bead_layout is used for every combo (original behaviour).
ic_specs = haskey(sw_cfg["sweep"], "ics") ?
           [Dict{String,Any}(string(k) => v for (k, v) in spec)
            for spec in sw_cfg["sweep"]["ics"]] :
           [nothing]

all_combos = [(ic, combo) for ic in ic_specs for combo in combos]
n_runs     = length(all_combos)

println("Sweep: $(sweep_config_path)")
println("Base config: $(base_path)")
println("Varying: $(join(param_names, ", "))")
length(ic_specs) > 1 && println("IC variants: $(length(ic_specs))")
println("Total runs: $n_runs\n")

# ── Shared output directory ───────────────────────────────────────────────────
# Allow [sweep.run] outdir to override the base config's outdir
outdir_effective = get(sweep_run_overrides, "outdir", base_cfg["run"]["outdir"])
base_dir = joinpath(PROJECT_ROOT_SW, outdir_effective)
mkpath(base_dir)

# ── Guard: never silently mix two datasets in one folder ─────────────────────
# Run IDs continue from the highest existing one, so simply pressing Run again on
# a folder that already holds runs APPENDS a second dataset next to the first —
# and every analysis script globs run_*, so the two get silently merged.  That is
# almost never what is wanted, so require an explicit choice.
let existing = filter(d -> occursin(r"^run_\d+$", d), readdir(base_dir, join = false))
    if !isempty(existing) && !(flag_resume || flag_skip_existing || flag_append)
        error("""

        Output folder already contains $(length(existing)) run(s):
            $base_dir

        Continuing would APPEND new runs alongside them (ids carry on from the
        highest existing one) and silently mix two datasets in one folder.

        Pick one:
          • rename or move that folder, then run again   → clean re-run
          • --resume          continue an interrupted sweep of the SAME config
          • --skip-existing   skip combos whose params already exist
          • --append          yes, really add more runs to this folder
        """)
    end
end

sweep_run_start = Int(get(sw_cfg["sweep"], "start_id", 500))

function next_run_id(base::String, start_id::Int = sweep_run_start)
    existing = filter(d -> occursin(r"^run_\d+$", d), readdir(base, join = false))
    nums = [parse(Int, d[5:end]) for d in existing]
    # Only consider IDs >= start_id
    sweep_nums = filter(n -> n >= start_id, nums)
    n = isempty(sweep_nums) ? start_id : maximum(sweep_nums) + 1
    return @sprintf("run_%04d", n)
end

# ── Build skip sets for resume / skip-existing ───────────────────────────────

# --resume: collect combo_indices that already have a complete metadata.json
done_combo_indices = Set{Int}()
if flag_resume
    for d in filter(d -> occursin(r"^run_\d+$", d), readdir(base_dir, join = false))
        meta_path = joinpath(base_dir, d, "metadata.json")
        obs_path  = joinpath(base_dir, d, "observables.jld2")
        isfile(meta_path) && isfile(obs_path) || continue
        try
            m = JSON3.read(read(meta_path, String), Dict)
            haskey(m, "combo_index") && push!(done_combo_indices, Int(m["combo_index"]))
        catch; end
    end
    println("--resume: $(length(done_combo_indices)) combos already complete, will skip them.")
end

# --skip-existing: collect (param→value) fingerprints of complete runs
done_fingerprints = Set{String}()
if flag_skip_existing
    for d in filter(d -> occursin(r"^run_\d+$", d), readdir(base_dir, join = false))
        par_path = joinpath(base_dir, d, "params.toml")
        obs_path = joinpath(base_dir, d, "observables.jld2")
        isfile(par_path) && isfile(obs_path) || continue
        try
            ep = TOML.parsefile(par_path)["params"]
            # fingerprint: sorted "name=value" pairs for swept params only
            fp = join(["$k=$(round(ep[k]; sigdigits=8))" for k in sort(param_names)], "|")
            push!(done_fingerprints, fp)
        catch; end
    end
    println("--skip-existing: $(length(done_fingerprints)) param combos already complete, will skip them.")
end

# ── Sweep index (written after all runs complete) ─────────────────────────────
sweep_records = Vector{Dict{String, Any}}()

# ── Run loop ──────────────────────────────────────────────────────────────────
for (combo_idx, (ic_spec, combo)) in enumerate(all_combos)
    # Build per-combo overrides
    override = Dict(param_names[i] => combo[i] for i in eachindex(param_names))

    # ── Skip if already complete ──────────────────────────────────────────────
    if flag_resume && combo_idx in done_combo_indices
        println("Skipping combo $combo_idx / $n_runs (already complete)")
        continue
    end
    if flag_skip_existing
        ic_fp = ic_spec !== nothing ? join(sort(["$k=$v" for (k,v) in ic_spec]), "|") : ""
        fp    = join(["$k=$(round(override[k]; sigdigits=8))" for k in sort(param_names)], "|")
        ic_fp != "" && (fp *= "|$ic_fp")
        if fp in done_fingerprints
            println("Skipping combo $combo_idx / $n_runs (params already exist)")
            continue
        end
    end

    println("=" ^ 60)
    println("Combo $combo_idx / $n_runs : $override")
    println("=" ^ 60)

    # Merge: base params ← sweep run overrides ← per-combo param overrides
    ep = merge(base_cfg["params"], Dict(k => v for (k, v) in override
                                        if k in keys(base_cfg["params"])))
    rc = merge(base_cfg["run"],    Dict(string(k) => v
                                        for (k, v) in sweep_run_overrides))
    ic_spec !== nothing && (rc = merge(rc, ic_spec))

    p = EngineRing.Params(
        N         = Int(round(ep["N"])),
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

    # ── Per-run independent noise stream ─────────────────────────────────────
    # A CONSTANT seed here (the previous behaviour) re-seeded the RNG identically
    # for every combination, so all runs in a sweep replayed the SAME noise
    # sequence and were not independent samples of the thermal noise.
    # Deriving the seed from the combo index gives each run its own stream while
    # keeping the whole sweep exactly reproducible.
    #
    # Arithmetic, not hash(): Julia's hash is not guaranteed stable across
    # versions, which would silently break reproducibility.
    # The 1e6 stride also prevents collisions between sweeps whose base seeds are
    # close together — with a naive seed+combo_idx, base 41/combo 2 would collide
    # with base 42/combo 1 (you have sweeps at seeds 41, 42, 43, 123, 456).
    run_seed = seed * 1_000_000 + combo_idx

    run_id  = next_run_id(base_dir)
    run_dir = IOUtils.init_run_dir(base_dir, run_id)

    # Save params and metadata before running
    IOUtils.save_params(run_dir, p, Dict{String,Any}(rc))
    meta = Dict(
        "run_id"        => run_id,
        "seed"          => seed,          # base seed of the sweep
        "seed_effective" => run_seed,     # actual RNG seed used by THIS run
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
        "ic_spec"       => ic_spec !== nothing ? ic_spec : Dict{String,Any}(),
        "wrap_applied"  => true,
    )
    IOUtils.save_metadata(run_dir, meta)

    Random.seed!(run_seed)

    bead_types = InitialConditions.antipolar_bead_types(p.N)

    if rc["bead_layout"] == "from_snapshot"
        snap_dir   = joinpath(PROJECT_ROOT_SW, get(rc, "ic_snapshot_dir", ""))
        snap_frame = Int(get(rc, "ic_snapshot_frame", -1))
        _, snap_pos, _ = IOUtils.load_trajectory(snap_dir)
        n_snap = size(snap_pos, 3)
        fi     = snap_frame < 0 ? n_snap + snap_frame + 1 : snap_frame
        size(snap_pos, 2) == p.N || error("Snapshot N=$(size(snap_pos,2)) ≠ config N=$(p.N)")
        state  = EngineRing.State(snap_pos[:, :, fi], copy(bead_types))
    else
        state = InitialConditions.create_ic(rc["bead_layout"], p, bead_types)
    end

    times     = Vector{Float64}(undef, frames)
    positions = Array{Float64, 3}(undef, 2, p.N, frames)
    obs_Rg    = Vector{Float64}(undef, frames)
    obs_λ1    = Vector{Float64}(undef, frames)
    obs_λ2    = Vector{Float64}(undef, frames)
    obs_b     = Vector{Float64}(undef, frames)
    obs_Gxx   = Vector{Float64}(undef, frames)
    obs_Gyy   = Vector{Float64}(undef, frames)
    obs_Gxy   = Vector{Float64}(undef, frames)
    obs_A1    = Vector{Float64}(undef, frames)
    obs_A2    = Vector{Float64}(undef, frames)
    obs_A3    = Vector{Float64}(undef, frames)
    obs_A4    = Vector{Float64}(undef, frames)
    obs_A5    = Vector{Float64}(undef, frames)
    obs_A6    = Vector{Float64}(undef, frames)
    obs_Kloc  = Vector{Float64}(undef, frames)
    obs_Kconn = Vector{Float64}(undef, frames)
    obs_P1    = Vector{Float64}(undef, frames)
    obs_P2    = Vector{Float64}(undef, frames)
    obs_P3    = Vector{Float64}(undef, frames)
    obs_P4    = Vector{Float64}(undef, frames)
    obs_P5    = Vector{Float64}(undef, frames)
    obs_P6    = Vector{Float64}(undef, frames)
    obs_Ixx   = Vector{Float64}(undef, frames)
    obs_Iyy   = Vector{Float64}(undef, frames)
    obs_Ixy   = Vector{Float64}(undef, frames)
    obs_Izz   = Vector{Float64}(undef, frames)
    obs_I1    = Vector{Float64}(undef, frames)
    obs_I2    = Vector{Float64}(undef, frames)
    obs_vmean = Vector{Float64}(undef, frames)
    obs_vmax  = Vector{Float64}(undef, frames)
    obs_vcom  = Vector{Float64}(undef, frames)

    dt_frame = substeps * p.dt
    t_sim    = 0.0
    hw       = L / 2.0

    prev_r = copy(state.r)   # used for per-frame speed (initialised to IC)

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

        obs = Observables.compute_frame_obs(state.r;
                  connector_idx = (p.N - 1) ÷ 2 + 1)
        obs_Rg[frame]  = obs.Rg
        obs_λ1[frame]  = obs.λ1
        obs_λ2[frame]  = obs.λ2
        obs_b[frame]   = obs.b
        obs_Gxx[frame]  = obs.Gxx
        obs_Gyy[frame]  = obs.Gyy
        obs_Gxy[frame]  = obs.Gxy
        obs_A1[frame]    = obs.A1
        obs_A2[frame]    = obs.A2
        obs_A3[frame]    = obs.A3
        obs_A4[frame]    = obs.A4
        obs_A5[frame]    = obs.A5
        obs_A6[frame]    = obs.A6
        obs_Kloc[frame]  = obs.Kloc
        obs_Kconn[frame] = obs.Kconn
        obs_P1[frame]    = obs.P1
        obs_P2[frame]    = obs.P2
        obs_P3[frame]    = obs.P3
        obs_P4[frame]    = obs.P4
        obs_P5[frame]    = obs.P5
        obs_P6[frame]    = obs.P6
        obs_Ixx[frame]   = obs.Ixx
        obs_Iyy[frame]   = obs.Iyy
        obs_Ixy[frame]   = obs.Ixy
        obs_Izz[frame]   = obs.Izz
        obs_I1[frame]    = obs.I1
        obs_I2[frame]    = obs.I2

        sv = Observables.speed_vals(state.r, prev_r, dt_frame)
        obs_vmean[frame] = sv.v_mean
        obs_vmax[frame]  = sv.v_max
        obs_vcom[frame]  = sv.v_com
        prev_r .= state.r   # update for next frame (in-place to avoid allocation)

        if frame % max(1, frames ÷ 10) == 0
            println("  $(round(Int, 100*frame/frames)) %  (frame $frame / $frames)")
        end
    end
    t_wall = round(time() - t_wall_start, digits = 1)
    println("  Done in $(t_wall) s")

    IOUtils.save_trajectory(run_dir, times, positions, bead_types)
    IOUtils.save_observables(run_dir, times, Dict{String, Vector{Float64}}(
        "Rg" => obs_Rg, "lambda1" => obs_λ1, "lambda2" => obs_λ2,
        "b"  => obs_b,  "Gxx" => obs_Gxx,    "Gyy" => obs_Gyy, "Gxy" => obs_Gxy,
        "A1" => obs_A1, "A2" => obs_A2, "A3" => obs_A3, "A4" => obs_A4,
        "A5" => obs_A5, "A6" => obs_A6,
        "Kloc" => obs_Kloc, "Kconn" => obs_Kconn,
        "P1" => obs_P1, "P2" => obs_P2, "P3" => obs_P3, "P4" => obs_P4,
        "P5" => obs_P5, "P6" => obs_P6,
        "Ixx" => obs_Ixx, "Iyy" => obs_Iyy, "Ixy" => obs_Ixy, "Izz" => obs_Izz,
        "I1" => obs_I1, "I2" => obs_I2,
        "v_mean" => obs_vmean, "v_max" => obs_vmax, "v_com" => obs_vcom))

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
