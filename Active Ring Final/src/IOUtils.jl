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
       save_observables,  load_observables,
       save_motion_frame_obs, load_motion_frame_obs,
       save_motion_windows,   load_motion_windows,
       save_motion_summary

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

# ── Motion observables (per-frame) ────────────────────────────────────────────

"""
    save_motion_frame_obs(dir, frame_obs)

Write all per-frame motion vectors to `dir/motion_observables.jld2`.
`frame_obs` is the NamedTuple returned by MotionAnalysis.compute_motion_frame_obs.
"""
function save_motion_frame_obs(dir::String, frame_obs)
    jldsave(joinpath(dir, "motion_observables.jld2");
        com_x            = frame_obs.com_x,
        com_y            = frame_obs.com_y,
        com_x_unwrapped  = frame_obs.com_x_unwrapped,
        com_y_unwrapped  = frame_obs.com_y_unwrapped,
        com_speed        = frame_obs.com_speed,
        theta_shape      = frame_obs.theta_shape,
        theta_unwrapped  = frame_obs.theta_unwrapped,
        omega_shape      = frame_obs.omega_shape,
        theta_reliable   = frame_obs.theta_reliable,
        circulation      = frame_obs.circulation,
        circ_half1       = frame_obs.circ_half1,
        circ_half2       = frame_obs.circ_half2,
        shape_similarity = frame_obs.shape_similarity,
    )
end

"""
    load_motion_frame_obs(dir) -> NamedTuple

Load per-frame motion observables from `dir/motion_observables.jld2`.
"""
function load_motion_frame_obs(dir::String)
    jldopen(joinpath(dir, "motion_observables.jld2"), "r") do f
        return (
            com_x            = f["com_x"],
            com_y            = f["com_y"],
            com_x_unwrapped  = f["com_x_unwrapped"],
            com_y_unwrapped  = f["com_y_unwrapped"],
            com_speed        = f["com_speed"],
            theta_shape      = f["theta_shape"],
            theta_unwrapped  = f["theta_unwrapped"],
            omega_shape      = f["omega_shape"],
            theta_reliable   = f["theta_reliable"],
            circulation      = f["circulation"],
            circ_half1       = f["circ_half1"],
            circ_half2       = f["circ_half2"],
            shape_similarity = f["shape_similarity"],
        )
    end
end

# ── Motion windows (per-window) ───────────────────────────────────────────────

# Numeric fields stored in motion_windows.jld2 (string "motion_label" is separate)
const WINDOW_NUMERIC_FIELDS = [
    "window", "frame_start", "frame_end", "t_start", "t_end",
    "Rg_mean", "Rg_std", "b_mean", "b_std",
    "A2_mean", "A2_std", "A3_mean", "A3_std", "Kloc_mean", "Kloc_std",
    "com_speed_mean", "com_speed_std", "com_net_disp",
    "com_path_length", "com_net_disp_ratio",
    "omega_mean", "omega_std", "omega_abs_mean", "omega_abs_std",
    "omega_reliable_frac",
    "circ_mean", "circ_std",
    "circ_half1_mean", "circ_half1_std",
    "circ_half2_mean", "circ_half2_std",
]

"""
    save_motion_windows(dir, windows)

Write per-window data to:
- `dir/motion_windows.jld2`   — programmatic access
- `dir/motion_windows.csv`    — human-readable

`windows` is the Vector{Dict{String,Any}} returned by
MotionAnalysis.compute_motion_windows.
"""
function save_motion_windows(dir::String, windows::Vector{Dict{String, Any}})
    # JLD2: store each numeric field as a vector over windows
    jld_dict = Dict{String, Any}()
    for field in WINDOW_NUMERIC_FIELDS
        jld_dict[field] = Float64[w[field] for w in windows]
    end
    jld_dict["motion_label"] = String[w["motion_label"] for w in windows]
    jldsave(joinpath(dir, "motion_windows.jld2"); (Symbol(k) => v for (k, v) in jld_dict)...)

    # CSV: one row per window
    csv_path = joinpath(dir, "motion_windows.csv")
    open(csv_path, "w") do io
        # Header
        all_fields = vcat(WINDOW_NUMERIC_FIELDS, ["motion_label"])
        println(io, join(all_fields, ","))
        # Rows
        for w in windows
            vals = vcat(
                [string(w[f]) for f in WINDOW_NUMERIC_FIELDS],
                [w["motion_label"]],
            )
            println(io, join(vals, ","))
        end
    end
end

"""
    load_motion_windows(dir) -> Vector{Dict{String, Any}}

Load per-window data from `dir/motion_windows.jld2`.
"""
function load_motion_windows(dir::String)
    jldopen(joinpath(dir, "motion_windows.jld2"), "r") do f
        labels  = f["motion_label"]
        n       = length(labels)
        windows = Vector{Dict{String, Any}}(undef, n)
        for i in 1:n
            d = Dict{String, Any}("motion_label" => labels[i])
            for field in WINDOW_NUMERIC_FIELDS
                # Backward-compatible: old JLD2 files lack newer fields → NaN
                vec = haskey(f, field) ? f[field] : fill(NaN, n)
                d[field] = vec[i]
            end
            windows[i] = d
        end
        return windows
    end
end

# ── Motion run summary ────────────────────────────────────────────────────────

"""
    save_motion_summary(dir, summary::Dict)

Write the run-level motion summary dict as JSON to `dir/motion_summary.json`.
"""
function save_motion_summary(dir::String, summary::Dict)
    open(joinpath(dir, "motion_summary.json"), "w") do io
        JSON3.pretty(io, summary)
    end
end

end # module IOUtils
