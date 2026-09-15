# run_targeted_seeds.jl
# Targeted seed campaign on the direction-selection cells of the N=17 circle sweep.
#
# Reads results/kconn_phase/basin_disagreement_circle.csv and keeps only TRUE
# direction-selection cells — those where the five circle seeds split between the
# outward AND inward basins (n_out ≥ 1 AND n_in ≥ 1).  Circular-borderline rows
# (a seed voting circular but no genuine out/in split) are dropped.
#
# For every remaining (κ, f₀) cell it runs 10 fresh simulations, seeds 1001…1010,
# with IDENTICAL physics and run settings to the circle sweep (base config
# configs/default.toml, antipolar layout, frames 1500, substeps 100) and writes
# standard run directories into results/Results_N17_circle_targeted/.
#
# The per-run physics + observable loop and the IOUtils save path are REUSED
# verbatim from run_sweep.jl (same EngineRing.step!, same periodic wrap, same 30
# observables, same save_params / save_metadata / save_trajectory /
# save_observables).  The only run_sweep step deliberately NOT reproduced is the
# per-run summarize_run() figure: the campaign's purpose (Task 3 transition
# probabilities via reduce_kconn → Kconn) needs only observables.jld2, and
# skipping the summary avoids a heavy CairoMakie dependency and a plotting cost on
# every one of the thousands of runs.  Trajectory saving is kept ON for
# consistency (toggle off with --no-traj — the only large per-run file).
#
# Modes (skip-existing is ALWAYS on, keyed by (κ, f₀, kBT, seed), so the campaign
# restarts cleanly and never repeats a completed run):
#
#   julia --project scripts/run_targeted_seeds.jl            # DEFAULT: checkpoint
#   julia --project scripts/run_targeted_seeds.jl --estimate #   one timed run,
#                                                             #   report cost, STOP
#   julia --project scripts/run_targeted_seeds.jl --run      # full campaign
#   julia --project scripts/run_targeted_seeds.jl --run --no-traj   # skip traj
#
# The default (checkpoint) never launches the full batch — that requires --run.

using TOML
using Printf
using Dates
using Random
using JSON3

const PROJECT_ROOT_TS = joinpath(@__DIR__, "..")

# ── Shared modules (same set run_sweep.jl loads, minus summarize_run) ─────────
include(joinpath(PROJECT_ROOT_TS, "src", "EngineRing.jl"))
using .EngineRing
include(joinpath(PROJECT_ROOT_TS, "src", "Observables.jl"))
using .Observables
include(joinpath(PROJECT_ROOT_TS, "src", "InitialConditions.jl"))
using .InitialConditions
include(joinpath(PROJECT_ROOT_TS, "src", "IOUtils.jl"))
using .IOUtils

# ── Flags / knobs ─────────────────────────────────────────────────────────────
flag_run  = "--run"     in ARGS                       # full campaign
flag_est  = !flag_run || ("--estimate" in ARGS)       # default = checkpoint
save_traj = !("--no-traj" in ARGS)

# ── Optional overrides ───────────────────────────────────────────────────────
# Every default below reproduces the original hardcoded campaign exactly, so an
# argument-free invocation behaves as before.
#
#   --cells PATH    CSV with `kappa` and `f0` columns, plus an OPTIONAL `kbt`
#                   column (a third axis; defaults to the base config kBT).
#                   Replaces the basin-CSV
#                   direction-selection filter with an explicit cell list.
#   --out RELPATH   output folder (relative to the project root)
#   --frames N      saved frames per run
#   --substeps N    physics steps per frame  (run length = frames·substeps·dt)
#   --seeds A:B     inclusive seed range
#   --shard I/N     process only work items I, I+N, I+2N, … of the plan.
#                   Shards are disjoint, so N of them can run concurrently.
function argval(flag::String, default::String)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i + 1]
end

CELLS_CSV   = argval("--cells", "")
OUTDIR_REL  = argval("--out", "results/Results_N17_circle_targeted")
FRAMES      = parse(Int, argval("--frames", "1500"))
SUBSTEPS    = parse(Int, argval("--substeps", "100"))
BEAD_LAYOUT = "antipolar"
let s = split(argval("--seeds", "1001:1010"), ":")
    global SEEDS = parse(Int, s[1]):parse(Int, s[2])
end
let s = split(argval("--shard", "1/1"), "/")
    global SHARD_I = parse(Int, s[1]); global SHARD_N = parse(Int, s[2])
end
1 <= SHARD_I <= SHARD_N || error("--shard I/N needs 1 ≤ I ≤ N (got $SHARD_I/$SHARD_N)")

const DISAGREE_CSV = joinpath(PROJECT_ROOT_TS, "results", "kconn_phase",
                              "basin_disagreement_circle.csv")
OUTDIR = joinpath(PROJECT_ROOT_TS, OUTDIR_REL)

# Each shard writes into a reserved run-id block so concurrent workers can share
# one output folder without ever colliding on a run_id.  Block width is far
# larger than any planned shard, and %04d widens automatically past 9999.
const ID_BLOCK = 100_000
shard_base(i)  = (i - 1) * ID_BLOCK + 1

rk(x) = round(x; sigdigits = 10)                       # key guard vs float drift
base_cfg = TOML.parsefile(joinpath(PROJECT_ROOT_TS, "configs", "default.toml"))
const BASE_KBT = Float64(base_cfg["params"]["kBT"])   # fallback when no kbt column

# ── Direction-selection cells (n_out ≥ 1 AND n_in ≥ 1) ────────────────────────
function load_direction_cells()
    isfile(DISAGREE_CSV) || error("Missing $DISAGREE_CSV")
    lines = filter(!isempty, readlines(DISAGREE_CSV))
    hdr   = split(lines[1], ",")
    ci(n) = findfirst(==(n), hdr)
    k_i, f_i, no_i, ni_i = ci("kappa"), ci("f0"), ci("n_out"), ci("n_in")
    cells = Tuple{Float64,Float64,Float64}[]
    for ln in lines[2:end]
        v = split(ln, ",")
        (parse(Int, v[no_i]) >= 1 && parse(Int, v[ni_i]) >= 1) || continue
        push!(cells, (parse(Float64, v[k_i]), parse(Float64, v[f_i]), BASE_KBT))
    end
    sort!(cells; by = c -> (c[1], c[2], c[3]))
    return cells
end

"""
Load an explicit (κ, f₀) cell list from any CSV carrying `kappa` and `f0`
columns — which is the shape the phase_reliability.jl coherence outputs already
have, so a selection can be piped straight from the scoring step into a campaign.
Duplicate cells are collapsed.
"""
function load_cells_csv(path::String)
    isfile(path) || error("Missing cell list: $path")
    lines = filter(!isempty, readlines(path))
    hdr   = String.(split(lines[1], ","))
    k_i   = findfirst(==("kappa"), hdr)
    f_i   = findfirst(==("f0"), hdr)
    b_i   = findfirst(==("kbt"), hdr)      # OPTIONAL third axis
    (k_i === nothing || f_i === nothing) &&
        error("$path needs `kappa` and `f0` columns (found: $(join(hdr, ", ")))")
    cells = Set{Tuple{Float64,Float64,Float64}}()
    for ln in lines[2:end]
        v = split(ln, ",")
        length(v) >= max(k_i, f_i) || continue
        kb = (b_i === nothing || length(v) < b_i) ? BASE_KBT : parse(Float64, v[b_i])
        push!(cells, (parse(Float64, v[k_i]), parse(Float64, v[f_i]), kb))
    end
    return sort!(collect(cells); by = c -> (c[1], c[2], c[3]))
end

cells      = isempty(CELLS_CSV) ? load_direction_cells() : load_cells_csv(CELLS_CSV)
# Per-cell index, used to DERIVE a per-run RNG seed.  Seeding with the bare seed
# makes every cell replay one noise stream, so neighbouring cells are not
# independent samples — the same defect run_sweep.jl fixes with seed*1e6+combo.
CELL_IDX   = Dict(c => i for (i, c) in enumerate(cells))
n_cells    = length(cells)
total_runs = n_cells * length(SEEDS)

# ── Params / run-config builders (base config ← per-cell κ,f₀ / per-run seed) ──
function make_params(κ, f0, kbt)
    ep = base_cfg["params"]
    EngineRing.Params(
        N = Int(round(ep["N"])), l = ep["l"], ks = ep["ks"], κ = κ,
        theta_max = ep["theta_max"], theta_buf = ep["theta_buf"], k_stop = ep["k_stop"],
        sigma = ep["sigma"], epsilon = ep["epsilon"], topology = Symbol(ep["topology"]),
        f0 = f0, mu = ep["mu"], kBT = kbt, dt = ep["dt"])
end

function make_rc(seed)
    merge(base_cfg["run"], Dict{String,Any}(
        "frames" => FRAMES, "substeps" => SUBSTEPS, "bead_layout" => BEAD_LAYOUT,
        "seed" => seed, "outdir" => OUTDIR_REL))
end

# ── run_id allocation (same scheme as run_sweep.next_run_id) ──────────────────
function next_run_id(base::String, start_id::Int = 1)
    isdir(base) || return @sprintf("run_%04d", start_id)
    existing = filter(d -> occursin(r"^run_\d+$", d), readdir(base, join = false))
    nums = [parse(Int, d[5:end]) for d in existing]
    nums = filter(n -> n >= start_id, nums)
    return @sprintf("run_%04d", isempty(nums) ? start_id : maximum(nums) + 1)
end

# ── Shard-local run-id allocation ────────────────────────────────────────────
# next_run_id re-scans the whole output folder per run, which is O(n) per run
# (O(n²) over a campaign) and — worse — races when several shards share a folder:
# two workers scanning at the same moment pick the SAME id and the second one
# dies in init_run_dir.  Instead each shard draws from its own reserved id block
# and keeps the counter in memory, seeded once from whatever it already wrote so
# an interrupted shard resumes cleanly.
const RUN_COUNTER = Ref(0)

function init_run_counter!()
    lo = shard_base(SHARD_I)
    hi = lo + ID_BLOCK - 1
    n  = lo
    if isdir(OUTDIR)
        for d in readdir(OUTDIR)
            m = match(r"^run_(\d+)$", d)
            m === nothing && continue
            v = parse(Int, m[1])
            (lo <= v <= hi && v >= n) && (n = v + 1)
        end
    end
    RUN_COUNTER[] = n
end

function take_run_id!()
    RUN_COUNTER[] == 0 && init_run_counter!()
    id = RUN_COUNTER[]
    RUN_COUNTER[] += 1
    return @sprintf("run_%04d", id)      # %04d widens automatically past 9999
end

# ── Skip-existing set: (κ, f₀, seed) of every COMPLETE targeted run ────────────
function completed_set()
    s = Set{Tuple{Float64,Float64,Int}}()
    isdir(OUTDIR) || return s
    for d in filter(d -> occursin(r"^run_\d+$", d), readdir(OUTDIR, join = false))
        rp = joinpath(OUTDIR, d)
        (isfile(joinpath(rp, "observables.jld2")) &&
         isfile(joinpath(rp, "params.toml")) &&
         isfile(joinpath(rp, "metadata.json"))) || continue
        try
            ep   = TOML.parsefile(joinpath(rp, "params.toml"))["params"]
            seed = Int(IOUtils.load_metadata(rp)["seed"])
            push!(s, (rk(ep["kappa"]), rk(ep["f0"]), rk(ep["kBT"]), seed))
        catch; end
    end
    return s
end

# ── One simulation → standard run directory (run_sweep loop, verbatim) ────────
function simulate_and_save(κ, f0, seed; kbt = BASE_KBT, cell_idx::Int = 0, save_traj::Bool = true)
    p  = make_params(κ, f0, kbt)
    rc = make_rc(seed)
    frames    = rc["frames"]; substeps = rc["substeps"]
    framerate = rc["framerate"]; L = rc["L"]

    mkpath(OUTDIR)
    run_id  = take_run_id!()
    run_dir = IOUtils.init_run_dir(OUTDIR, run_id)

    IOUtils.save_params(run_dir, p, Dict{String,Any}(rc))
    IOUtils.save_metadata(run_dir, Dict(
        "run_id" => run_id, "seed" => seed,
        "seed_effective" => seed * 1_000_000 + cell_idx, "timestamp" => string(now()),
        "julia_version" => string(VERSION), "campaign" => "circle_targeted",
        "kappa" => κ, "f0" => f0, "kBT" => kbt, "n_frames" => frames, "substeps" => substeps,
        "framerate" => framerate, "L" => L, "bead_layout" => rc["bead_layout"],
        "wrap_applied" => true))

    run_seed = seed * 1_000_000 + cell_idx      # independent stream per (cell, seed)
    Random.seed!(run_seed)
    bead_types = InitialConditions.antipolar_bead_types(p.N)
    state      = InitialConditions.create_ic(rc["bead_layout"], p, bead_types)

    times     = Vector{Float64}(undef, frames)
    positions = Array{Float64, 3}(undef, 2, p.N, frames)
    obs_Rg = Vector{Float64}(undef, frames); obs_λ1 = Vector{Float64}(undef, frames)
    obs_λ2 = Vector{Float64}(undef, frames); obs_b  = Vector{Float64}(undef, frames)
    obs_Gxx = Vector{Float64}(undef, frames); obs_Gyy = Vector{Float64}(undef, frames)
    obs_Gxy = Vector{Float64}(undef, frames)
    obs_A1 = Vector{Float64}(undef, frames); obs_A2 = Vector{Float64}(undef, frames)
    obs_A3 = Vector{Float64}(undef, frames); obs_A4 = Vector{Float64}(undef, frames)
    obs_A5 = Vector{Float64}(undef, frames); obs_A6 = Vector{Float64}(undef, frames)
    obs_Kloc = Vector{Float64}(undef, frames); obs_Kconn = Vector{Float64}(undef, frames)
    obs_P1 = Vector{Float64}(undef, frames); obs_P2 = Vector{Float64}(undef, frames)
    obs_P3 = Vector{Float64}(undef, frames); obs_P4 = Vector{Float64}(undef, frames)
    obs_P5 = Vector{Float64}(undef, frames); obs_P6 = Vector{Float64}(undef, frames)
    obs_Ixx = Vector{Float64}(undef, frames); obs_Iyy = Vector{Float64}(undef, frames)
    obs_Ixy = Vector{Float64}(undef, frames); obs_Izz = Vector{Float64}(undef, frames)
    obs_I1 = Vector{Float64}(undef, frames); obs_I2 = Vector{Float64}(undef, frames)
    obs_vmean = Vector{Float64}(undef, frames); obs_vmax = Vector{Float64}(undef, frames)
    obs_vcom  = Vector{Float64}(undef, frames)

    dt_frame = substeps * p.dt
    t_sim    = 0.0
    hw       = L / 2.0
    prev_r   = copy(state.r)

    t_wall_start = time()
    for frame in 1:frames
        for _ in 1:substeps
            EngineRing.step!(state, p)
        end
        t_sim += dt_frame

        min_x = minimum(@view state.r[1, :]); max_x = maximum(@view state.r[1, :])
        min_y = minimum(@view state.r[2, :]); max_y = maximum(@view state.r[2, :])
        min_x >  hw && (state.r[1, :] .-= L)
        max_x < -hw && (state.r[1, :] .+= L)
        min_y >  hw && (state.r[2, :] .-= L)
        max_y < -hw && (state.r[2, :] .+= L)

        times[frame]           = t_sim
        positions[:, :, frame] = state.r

        obs = Observables.compute_frame_obs(state.r; connector_idx = (p.N - 1) ÷ 2 + 1)
        obs_Rg[frame] = obs.Rg; obs_λ1[frame] = obs.λ1; obs_λ2[frame] = obs.λ2
        obs_b[frame] = obs.b
        obs_Gxx[frame] = obs.Gxx; obs_Gyy[frame] = obs.Gyy; obs_Gxy[frame] = obs.Gxy
        obs_A1[frame] = obs.A1; obs_A2[frame] = obs.A2; obs_A3[frame] = obs.A3
        obs_A4[frame] = obs.A4; obs_A5[frame] = obs.A5; obs_A6[frame] = obs.A6
        obs_Kloc[frame] = obs.Kloc; obs_Kconn[frame] = obs.Kconn
        obs_P1[frame] = obs.P1; obs_P2[frame] = obs.P2; obs_P3[frame] = obs.P3
        obs_P4[frame] = obs.P4; obs_P5[frame] = obs.P5; obs_P6[frame] = obs.P6
        obs_Ixx[frame] = obs.Ixx; obs_Iyy[frame] = obs.Iyy; obs_Ixy[frame] = obs.Ixy
        obs_Izz[frame] = obs.Izz; obs_I1[frame] = obs.I1; obs_I2[frame] = obs.I2

        sv = Observables.speed_vals(state.r, prev_r, dt_frame)
        obs_vmean[frame] = sv.v_mean; obs_vmax[frame] = sv.v_max; obs_vcom[frame] = sv.v_com
        prev_r .= state.r
    end
    t_wall = time() - t_wall_start

    save_traj && IOUtils.save_trajectory(run_dir, times, positions, bead_types)
    IOUtils.save_observables(run_dir, times, Dict{String, Vector{Float64}}(
        "Rg" => obs_Rg, "lambda1" => obs_λ1, "lambda2" => obs_λ2,
        "b" => obs_b, "Gxx" => obs_Gxx, "Gyy" => obs_Gyy, "Gxy" => obs_Gxy,
        "A1" => obs_A1, "A2" => obs_A2, "A3" => obs_A3, "A4" => obs_A4,
        "A5" => obs_A5, "A6" => obs_A6, "Kloc" => obs_Kloc, "Kconn" => obs_Kconn,
        "P1" => obs_P1, "P2" => obs_P2, "P3" => obs_P3, "P4" => obs_P4,
        "P5" => obs_P5, "P6" => obs_P6,
        "Ixx" => obs_Ixx, "Iyy" => obs_Iyy, "Ixy" => obs_Ixy, "Izz" => obs_Izz,
        "I1" => obs_I1, "I2" => obs_I2,
        "v_mean" => obs_vmean, "v_max" => obs_vmax, "v_com" => obs_vcom))

    return run_dir, t_wall
end

dir_bytes(dir) = sum(filesize(joinpath(dir, f)) for f in readdir(dir); init = 0)
fmt_gb(bytes)  = bytes / 1024^3

# ── Header ────────────────────────────────────────────────────────────────────
println("Targeted seed campaign")
println("  source : $(isempty(CELLS_CSV) ? DISAGREE_CSV * "  (n_out ≥ 1 AND n_in ≥ 1)" : CELLS_CSV)")
println("  output : $OUTDIR")
@printf("  cells  : %d\n", n_cells)
@printf("  seeds  : %d…%d  (%d per cell)\n", first(SEEDS), last(SEEDS), length(SEEDS))
@printf("  length : %d frames × %d substeps\n", FRAMES, SUBSTEPS)
@printf("  planned runs : %d cells × %d seeds = %d\n", n_cells, length(SEEDS), total_runs)
println("  trajectory saving : $(save_traj ? "ON" : "OFF (--no-traj)")")
n_cells == 0 && error("No cells selected — nothing to do.")

done      = completed_set()
planned   = [(c, s) for c in cells for s in SEEDS]     # element = ((κ, f₀), seed)
if SHARD_N > 1
    planned = planned[SHARD_I:SHARD_N:end]
    @printf("  SHARD %d/%d → %d of the planned runs (ids from %d)\n",
            SHARD_I, SHARD_N, length(planned), shard_base(SHARD_I))
end
total_runs = length(planned)
is_pending(e) = !((rk(e[1][1]), rk(e[1][2]), rk(e[1][3]), e[2]) in done)
n_already = count(!is_pending, planned)

# ── CHECKPOINT (default): exactly one timed run, then report cost and STOP ─────
if flag_est
    println("\n" * "="^64)
    println("CHECKPOINT MODE — one timed run, then report (no full batch).")
    println("="^64)

    # First planned (cell, seed) that is not already complete.
    pick_idx = findfirst(is_pending, planned)

    if pick_idx === nothing
        println("\nAll $total_runs runs already complete — nothing to time.")
        println("Disk currently used by campaign: $(round(fmt_gb(dir_bytes(OUTDIR)); digits=2)) GB")
        exit(0)
    end

    (κ, f0, kbt), seed = planned[pick_idx]
    @printf("\nTiming one run:  κ=%.4f  f₀=%.4f  seed=%d  (%d frames × %d substeps)…\n",
            κ, f0, seed, FRAMES, SUBSTEPS)
    run_dir, t_wall = simulate_and_save(κ, f0, seed; kbt = kbt, cell_idx = CELL_IDX[(κ, f0, kbt)], save_traj = save_traj)

    run_bytes = dir_bytes(run_dir)
    obs_bytes = filesize(joinpath(run_dir, "observables.jld2")) +
                filesize(joinpath(run_dir, "params.toml")) +
                filesize(joinpath(run_dir, "metadata.json"))

    est_wall_s = t_wall * total_runs
    println("\n" * "-"^64)
    println("COST REPORT")
    println("-"^64)
    @printf("(a) cells in campaign                              : %d\n", n_cells)
    @printf("(b) runs planned for THIS shard (%d/%d)             : %d  (%d already complete)\n",
            SHARD_I, SHARD_N, total_runs, n_already)
    @printf("(c) measured wall time for one run                 : %.1f s\n", t_wall)
    @printf("    estimated total wall time (%d runs)            : %.1f min  =  %.2f h\n",
            total_runs, est_wall_s / 60, est_wall_s / 3600)
    @printf("    estimated remaining (%d runs)                  : %.2f h\n",
            total_runs - n_already, t_wall * (total_runs - n_already) / 3600)
    @printf("(d) disk per run  : %.2f MB  (full)   |  %.2f MB  (observables+meta only)\n",
            run_bytes / 1024^2, obs_bytes / 1024^2)
    @printf("    estimated total disk : %.2f GB  (with traj)  |  %.2f GB  (--no-traj)\n",
            fmt_gb(run_bytes * total_runs), fmt_gb(obs_bytes * total_runs))
    println("-"^64)
    println("STOPPED at checkpoint.  Re-run with  --run  to launch the full campaign")
    println("(skip-existing reuses this timed run; add --no-traj to skip trajectories).")
    exit(0)
end

# ── FULL CAMPAIGN (--run): all cells × seeds, skip-existing, sequential ────────
println("\n" * "="^64)
@printf("FULL CAMPAIGN — %d runs (%d already complete, will skip)\n", total_runs, n_already)
println("="^64)

records = Tuple{String,Float64,Float64,Int,Float64}[]   # run_id, κ, f0, seed, wall
for (i, e) in enumerate(planned)
    (κ, f0, kbt), seed = e
    if (rk(κ), rk(f0), seed) in done
        @printf("[%d/%d] skip κ=%.4f f₀=%.4f seed=%d (complete)\n", i, total_runs, κ, f0, seed)
        continue
    end
    @printf("[%d/%d] run  κ=%.4f f₀=%.4f seed=%d … ", i, total_runs, κ, f0, seed)
    run_dir, t_wall = simulate_and_save(κ, f0, seed; kbt = kbt, cell_idx = CELL_IDX[(κ, f0, kbt)], save_traj = save_traj)
    @printf("%.1f s  → %s\n", t_wall, basename(run_dir))
    push!(records, (basename(run_dir), κ, f0, seed, round(t_wall; digits = 1)))
end

# Lightweight campaign index (append-safe: rewrite from scratch of this session's runs)
idx_path = joinpath(OUTDIR, SHARD_N > 1 ?
                    @sprintf("targeted_index_shard%02d.csv", SHARD_I) : "targeted_index.csv")
open(idx_path, "w") do io
    println(io, "run_id,kappa,f0,seed,wall_s")
    for (rid, κ, f0, s, w) in records
        println(io, join((rid, κ, f0, s, w), ","))
    end
end
@printf("\nCampaign complete.  %d new runs this session.  Index: %s\n", length(records), idx_path)
