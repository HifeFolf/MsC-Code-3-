module IOUtils

using JLD2
using TOML
using JSON3
using Dates
using Main.EngineRing

export init_run_dir,
       save_params,   load_params,
       save_metadata, load_metadata,
       save_trajectory,   load_trajectory,
       save_observables,  load_observables

# ── Run directory ─────────────────────────────────────────────────────────────

"""
    init_run_dir(base_dir, run_id) -> String

Create and return the path `base_dir/run_id/`.
Errors if the directory already exists (prevents accidental overwrites).
"""
function init_run_dir(base_dir::String, run_id::String)
    dir = joinpath(base_dir, run_id)
    isdir(dir) && error("Run directory already exists: $dir\nChoose a different run_id or delete it first.")
    mkpath(dir)
    return dir
end

# ── Params ────────────────────────────────────────────────────────────────────

"""
    save_params(dir, p, run_cfg)

Write engine Params and run config dict to `dir/params.toml`.
Engine params go under [params]; run config goes under [run].
"""
function save_params(dir::String, p, run_cfg::Dict)
    d = Dict(
        "params" => Dict(
            "N"           => p.N,
            "l"           => p.l,
            "ks"          => p.ks,
            "kappa"       => p.κ,
            "theta_max"   => p.theta_max,
            "theta_buf"   => p.theta_buf,
            "k_stop"      => p.k_stop,
            "sigma"       => p.sigma,
            "epsilon"     => p.epsilon,
            "topology"    => string(p.topology),
            "f0"          => p.f0,
            "mu"          => p.mu,
            "kBT"         => p.kBT,
            "dt"          => p.dt,
        ),
        "run" => run_cfg,
    )
    open(joinpath(dir, "params.toml"), "w") do io
        TOML.print(io, d)
    end
end

"""
    load_params(dir) -> (Params, Dict)

Read `dir/params.toml` and reconstruct an EngineRing.Params and the run config dict.
"""
function load_params(dir::String)
    d      = TOML.parsefile(joinpath(dir, "params.toml"))
    ep     = d["params"]
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
    return p, d["run"]
end

# ── Metadata ──────────────────────────────────────────────────────────────────

"""
    save_metadata(dir, meta::Dict)

Write `meta` as JSON to `dir/metadata.json`.
"""
function save_metadata(dir::String, meta::Dict)
    open(joinpath(dir, "metadata.json"), "w") do io
        JSON3.pretty(io, meta)
    end
end

"""
    load_metadata(dir) -> Dict
"""
function load_metadata(dir::String)
    return JSON3.read(read(joinpath(dir, "metadata.json"), String), Dict)
end

# ── Trajectory ────────────────────────────────────────────────────────────────

"""
    save_trajectory(dir, times, positions, bead_types)

Write to `dir/traj.jld2`:
- "times"      :: Vector{Float64}         length = n_frames
- "positions"  :: Array{Float64, 3}       2 × N × n_frames
- "bead_types" :: Vector{String}          length = N  (Symbols stored as String)
"""
function save_trajectory(dir::String,
                         times::Vector{Float64},
                         positions::Array{Float64, 3},
                         bead_types::Vector{Symbol})
    jldsave(joinpath(dir, "traj.jld2");
            times      = times,
            positions  = positions,
            bead_types = string.(bead_types))
end

"""
    load_trajectory(dir) -> (times, positions, bead_types)

Load from `dir/traj.jld2`. bead_types is returned as Vector{Symbol}.
"""
function load_trajectory(dir::String)
    jldopen(joinpath(dir, "traj.jld2"), "r") do f
        times      = f["times"]
        positions  = f["positions"]
        bead_types = Symbol.(f["bead_types"])
        return times, positions, bead_types
    end
end

# ── Observables ───────────────────────────────────────────────────────────────

"""
    save_observables(dir, times, obs)

Write to `dir/observables.jld2`:
- "times"   :: Vector{Float64}
- one entry per key in obs :: Dict{String, Vector{Float64}}

Recognised keys: "Rg", "b", "lambda1", "lambda2", "Gxx", "Gyy", "Gxy"
"""
function save_observables(dir::String,
                          times::Vector{Float64},
                          obs::Dict{String, Vector{Float64}})
    jldsave(joinpath(dir, "observables.jld2"); times = times, (Symbol(k) => v for (k, v) in obs)...)
end

"""
    load_observables(dir) -> (times, obs::Dict{String, Vector{Float64}})
"""
function load_observables(dir::String)
    jldopen(joinpath(dir, "observables.jld2"), "r") do f
        times = f["times"]
        keys_ = filter(k -> k != "times", keys(f))
        obs   = Dict{String, Vector{Float64}}(k => f[k] for k in keys_)
        return times, obs
    end
end

end # module IOUtils
