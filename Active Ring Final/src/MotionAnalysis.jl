module MotionAnalysis

using Statistics

export compute_motion_frame_obs,
       compute_motion_windows,
       classify_motion_window,
       classify_motion_run,
       classify_morphology,
       motion_run_summary

# ── Provisional classifier thresholds ────────────────────────────────────────
# Edit this dict to tune the motion classifier. All thresholds are in the same
# units as the observables (simulation length units, radians, time units).

const MOTION_THRESHOLDS = Dict{Symbol, Float64}(
    :drift_speed   => 0.05,   # mean COM speed to qualify as "drifting"
    :omega_thresh  => 0.01,   # |mean Ω_shape| to qualify as "rotating"
    :circ_locked   => 0.10,   # |mean circulation| to qualify as "locked"
    :osc_cv        => 0.50,   # coefficient of variation in speed for "oscillatory"
    :b_reliable    => 0.02,   # minimum asphericity b for theta_shape to be trusted
    :switch_frac   => 0.30,   # fraction of mismatching windows to call a run "switching"
)

# ── Morphology classifier ─────────────────────────────────────────────────────
# Extracted from phase_diagram.jl so Stage 1 can attach a morphology label
# without importing that script.  The thresholds and centroids are identical.

const MORPH_CENTROIDS = Dict(
    :circular   => (0.004, 0.022, 0.017),
    :curled     => (0.062, 0.475, 0.306),
    :elongated  => (0.522, 0.293, 0.065),
    :triangular => (0.213, 0.180, 0.081),
)

const MORPH_N_SNAPSHOTS = 150

"""
    classify_morphology(obs, n_snapshots) -> Symbol

Classify the dominant morphology of a run by majority vote over
`n_snapshots` evenly-spaced frames.  `obs` must contain "Rg", "b", "A2", "A3".
Returns one of :circular, :curled, :elongated, :triangular.
Returns :unknown if any required key is missing.
"""
function classify_morphology(obs::Dict{String, Vector{Float64}},
                             n_snapshots::Int = MORPH_N_SNAPSHOTS)
    for k in ("Rg", "b", "A2", "A3")
        haskey(obs, k) || return :unknown
    end

    n       = length(obs["Rg"])
    indices = round.(Int, range(1, n, length = min(n_snapshots, n)))
    votes   = Dict{Symbol, Int}(:circular => 0, :curled => 0,
                                :elongated => 0, :triangular => 0)

    for i in indices
        Rg = obs["Rg"][i]; b = obs["b"][i]
        A2 = obs["A2"][i]; A3 = obs["A3"][i]

        regime = if Rg < 2.0 && A2 > 0.25
            :curled
        elseif b < 0.05 && A2 < 0.08
            :circular
        elseif b > 0.25 && A2 > 0.15 && Rg > 2.0
            :elongated
        elseif 0.05 < b < 0.30 && A3 > 0.04 && 0.08 < A2 < 0.25
            :triangular
        else
            best = :circular; best_d = Inf
            for (reg, (cb, cA2, cA3)) in MORPH_CENTROIDS
                d = (b - cb)^2 + (A2 - cA2)^2 + (A3 - cA3)^2
                d < best_d && (best_d = d; best = reg)
            end
            best
        end
        votes[regime] = get(votes, regime, 0) + 1
    end

    return argmax(votes)
end

# ── Helper: local bond tangent at each bead ───────────────────────────────────

"""
    bond_tangents(r, N) -> (tx, ty)

Unit tangent vector at each bead, pointing in the contour traversal direction.
Uses the averaged tangent from the two neighbouring bonds (periodic wrap-around).
`r` is a 2×N position matrix for one frame.
"""
function bond_tangents(r::Matrix{Float64}, N::Int)
    tx = Vector{Float64}(undef, N)
    ty = Vector{Float64}(undef, N)
    for i in 1:N
        im = (i == 1) ? N : i - 1
        ip = (i == N) ? 1 : i + 1
        dx1 = r[1, ip] - r[1, i];  dy1 = r[2, ip] - r[2, i]
        n1  = sqrt(dx1^2 + dy1^2)
        dx2 = r[1, i]  - r[1, im]; dy2 = r[2, i]  - r[2, im]
        n2  = sqrt(dx2^2 + dy2^2)
        dx  = dx1 / max(n1, 1e-14) + dx2 / max(n2, 1e-14)
        dy  = dy1 / max(n1, 1e-14) + dy2 / max(n2, 1e-14)
        n   = sqrt(dx^2 + dy^2)
        tx[i] = dx / max(n, 1e-14)
        ty[i] = dy / max(n, 1e-14)
    end
    return tx, ty
end

# ── Helper: per-bead velocities via finite differences ────────────────────────

"""
    bead_velocities(positions, dt_frame, L) -> (vx, vy)

Estimate per-bead velocities (N×F matrices) using central finite differences
(forward/backward at the endpoints). Periodic wrap events (displacement > L/2)
are corrected so velocities remain physically meaningful across box boundaries.
"""
function bead_velocities(positions::Array{Float64,3}, dt_frame::Float64, L::Float64)
    _, N, F = size(positions)
    vx = zeros(N, F)
    vy = zeros(N, F)
    half_L = L / 2.0

    function unwrap_d(d)
        d >  half_L && return d - L
        d < -half_L && return d + L
        return d
    end

    for t in 1:F
        t_m = max(t - 1, 1)
        t_p = min(t + 1, F)
        dt  = (t_p - t_m) * dt_frame   # 2·dt_frame (central) or dt_frame (edges)
        for i in 1:N
            vx[i, t] = unwrap_d(positions[1, i, t_p] - positions[1, i, t_m]) / dt
            vy[i, t] = unwrap_d(positions[2, i, t_p] - positions[2, i, t_m]) / dt
        end
    end
    return vx, vy
end

# ── Per-frame motion observables ──────────────────────────────────────────────

"""
    compute_motion_frame_obs(positions, morphology_obs, dt_frame, L, N) -> NamedTuple

Compute all per-frame motion observables from a saved trajectory.

Returns a NamedTuple with Vector{Float64} fields of length n_frames:
  com_x, com_y              raw center-of-mass coordinates (wrapped)
  com_x_unwrapped, com_y_unwrapped  cumulative unwrapped COM (for speed/MSD)
  com_speed                 |ΔR_cm| / dt_frame, using unwrapped COM
  theta_shape               principal-axis angle (radians, in (−π/2, π/2])
  theta_unwrapped           cumulative unwrapped theta (mod-π unwrapping)
  omega_shape               Δtheta_unwrapped / dt_frame
  theta_reliable            1.0 if b > b_reliable threshold, else 0.0
  circulation               (1/N) Σᵢ vᵢ·t̂ᵢ  over all beads
  circ_half1                same, beads 1..N÷2  (CCW active half)
  circ_half2                same, beads N÷2+1..N (CW active half)
  shape_similarity          normalized Frobenius overlap vs frame 1 (centered only)

morphology_obs must contain "b", "Gxx", "Gyy", "Gxy".
"""
function compute_motion_frame_obs(positions::Array{Float64,3},
                                  morphology_obs::Dict{String, Vector{Float64}},
                                  dt_frame::Float64,
                                  L::Float64,
                                  N::Int)
    F      = size(positions, 3)
    half_L = L / 2.0
    b_th   = MOTION_THRESHOLDS[:b_reliable]
    half1  = 1:(N ÷ 2)
    half2  = (N ÷ 2 + 1):N

    # ── Center-of-mass positions ─────────────────────────────────────────────
    com_x = [mean(positions[1, :, t]) for t in 1:F]
    com_y = [mean(positions[2, :, t]) for t in 1:F]

    # Unwrap COM: accumulate increments after correcting for ±L jumps.
    # The whole ring is shifted together at each frame, so COM jumps are ±L.
    com_x_uw = Vector{Float64}(undef, F)
    com_y_uw = Vector{Float64}(undef, F)
    com_x_uw[1] = com_x[1]
    com_y_uw[1] = com_y[1]
    for t in 2:F
        dx = com_x[t] - com_x[t-1]
        dy = com_y[t] - com_y[t-1]
        dx >  half_L && (dx -= L)
        dx < -half_L && (dx += L)
        dy >  half_L && (dy -= L)
        dy < -half_L && (dy += L)
        com_x_uw[t] = com_x_uw[t-1] + dx
        com_y_uw[t] = com_y_uw[t-1] + dy
    end

    # COM speed from unwrapped trajectory; frame 1 is defined as 0
    com_speed = Vector{Float64}(undef, F)
    com_speed[1] = 0.0
    for t in 2:F
        dxc = com_x_uw[t] - com_x_uw[t-1]
        dyc = com_y_uw[t] - com_y_uw[t-1]
        com_speed[t] = sqrt(dxc^2 + dyc^2) / dt_frame
    end

    # ── Principal-axis angle ─────────────────────────────────────────────────
    Gxx_arr = morphology_obs["Gxx"]
    Gyy_arr = morphology_obs["Gyy"]
    Gxy_arr = morphology_obs["Gxy"]
    b_arr   = morphology_obs["b"]

    theta_shape     = Vector{Float64}(undef, F)
    theta_reliable  = Vector{Float64}(undef, F)
    theta_unwrapped = Vector{Float64}(undef, F)
    omega_shape     = Vector{Float64}(undef, F)

    for t in 1:F
        theta_shape[t]    = 0.5 * atan(2.0 * Gxy_arr[t], Gxx_arr[t] - Gyy_arr[t])
        theta_reliable[t] = b_arr[t] >= b_th ? 1.0 : 0.0
    end

    # Unwrap mod π: principal axis is a direction (not a vector), defined mod π.
    # Use round() to choose the nearest integer multiple of π to subtract.
    theta_unwrapped[1] = theta_shape[1]
    for t in 2:F
        dθ = theta_shape[t] - theta_shape[t-1]
        dθ -= π * round(dθ / π)          # wrap increment to (−π/2, π/2]
        theta_unwrapped[t] = theta_unwrapped[t-1] + dθ
    end

    omega_shape[1] = 0.0
    for t in 2:F
        omega_shape[t] = (theta_unwrapped[t] - theta_unwrapped[t-1]) / dt_frame
    end

    # ── Bead velocities and contour circulation ──────────────────────────────
    vx, vy = bead_velocities(positions, dt_frame, L)

    circulation = Vector{Float64}(undef, F)
    circ_half1  = Vector{Float64}(undef, F)
    circ_half2  = Vector{Float64}(undef, F)

    for t in 1:F
        r_t        = positions[:, :, t]
        tx_t, ty_t = bond_tangents(r_t, N)

        dot_all   = 0.0
        dot_half1 = 0.0
        dot_half2 = 0.0
        for i in 1:N
            d = vx[i, t] * tx_t[i] + vy[i, t] * ty_t[i]
            dot_all += d
            i in half1 && (dot_half1 += d)
            i in half2 && (dot_half2 += d)
        end
        circulation[t] = dot_all   / N
        circ_half1[t]  = dot_half1 / length(half1)
        circ_half2[t]  = dot_half2 / length(half2)
    end

    # ── Shape similarity vs frame 1 (translation removed only) ──────────────
    r0 = positions[:, :, 1]
    cm0x = mean(r0[1, :]);  cm0y = mean(r0[2, :])
    r0_c = copy(r0)
    r0_c[1, :] .-= cm0x;   r0_c[2, :] .-= cm0y
    norm0 = sqrt(sum(r0_c .^ 2))

    shape_similarity = Vector{Float64}(undef, F)
    for t in 1:F
        rt = positions[:, :, t]
        cmx = mean(rt[1, :]);  cmy = mean(rt[2, :])
        rt_c = copy(rt)
        rt_c[1, :] .-= cmx;   rt_c[2, :] .-= cmy
        norm_t = sqrt(sum(rt_c .^ 2))
        denom  = norm0 * norm_t
        shape_similarity[t] = denom > 1e-14 ? sum(r0_c .* rt_c) / denom : 0.0
    end

    return (
        com_x            = com_x,
        com_y            = com_y,
        com_x_unwrapped  = com_x_uw,
        com_y_unwrapped  = com_y_uw,
        com_speed        = com_speed,
        theta_shape      = theta_shape,
        theta_unwrapped  = theta_unwrapped,
        omega_shape      = omega_shape,
        theta_reliable   = theta_reliable,
        circulation      = circulation,
        circ_half1       = circ_half1,
        circ_half2       = circ_half2,
        shape_similarity = shape_similarity,
    )
end

# ── Per-window summary statistics ─────────────────────────────────────────────

"""
    compute_motion_windows(frame_obs, morphology_obs, vtimes, window_size)
        -> Vector{Dict{String, Any}}

Divide the trajectory into windows of `window_size` frames (the last window
absorbs any remainder). For each window compute means, stds, and a provisional
motion label.

`vtimes` is the video-time vector (seconds), length = n_frames.
`morphology_obs` must contain "Rg", "b", "A2", "A3", "Kloc".
"""
function compute_motion_windows(frame_obs,
                                morphology_obs::Dict{String, Vector{Float64}},
                                vtimes::Vector{Float64},
                                window_size::Int)
    F         = length(vtimes)
    n_windows = max(1, F ÷ window_size)
    windows   = Vector{Dict{String, Any}}(undef, n_windows)

    for w in 1:n_windows
        i1 = (w - 1) * window_size + 1
        i2 = (w == n_windows) ? F : w * window_size   # last window takes remainder
        sl = i1:i2

        # morphology stats from saved observables
        Rg_sl   = morphology_obs["Rg"][sl]
        b_sl    = morphology_obs["b"][sl]
        A2_sl   = morphology_obs["A2"][sl]
        A3_sl   = morphology_obs["A3"][sl]
        Kloc_sl = morphology_obs["Kloc"][sl]

        # motion stats from frame_obs
        speed_sl  = frame_obs.com_speed[sl]
        omega_sl  = frame_obs.omega_shape[sl]
        circ_sl   = frame_obs.circulation[sl]
        c1_sl     = frame_obs.circ_half1[sl]
        c2_sl     = frame_obs.circ_half2[sl]
        rel_sl    = frame_obs.theta_reliable[sl]

        # net COM displacement over this window (from unwrapped COM)
        dx_net   = frame_obs.com_x_unwrapped[i2] - frame_obs.com_x_unwrapped[i1]
        dy_net   = frame_obs.com_y_unwrapped[i2] - frame_obs.com_y_unwrapped[i1]
        net_disp = sqrt(dx_net^2 + dy_net^2)

        # COM path length (arc length of unwrapped trajectory in this window)
        x_uw = frame_obs.com_x_unwrapped[i1:i2]
        y_uw = frame_obs.com_y_unwrapped[i1:i2]
        path_len = sum(sqrt((x_uw[t] - x_uw[t-1])^2 + (y_uw[t] - y_uw[t-1])^2)
                       for t in 2:lastindex(x_uw))

        # net displacement / path length: 0 → orbiting/looping, 1 → straight drift
        net_disp_ratio = path_len > 1e-10 ? net_disp / path_len : 0.0

        w_dict = Dict{String, Any}(
            "window"              => w,
            "frame_start"         => i1,
            "frame_end"           => i2,
            "t_start"             => vtimes[i1],
            "t_end"               => vtimes[i2],
            # morphology
            "Rg_mean"             => mean(Rg_sl),
            "Rg_std"              => std(Rg_sl),
            "b_mean"              => mean(b_sl),
            "b_std"               => std(b_sl),
            "A2_mean"             => mean(A2_sl),
            "A2_std"              => std(A2_sl),
            "A3_mean"             => mean(A3_sl),
            "A3_std"              => std(A3_sl),
            "Kloc_mean"           => mean(Kloc_sl),
            "Kloc_std"            => std(Kloc_sl),
            # motion
            "com_speed_mean"      => mean(speed_sl),
            "com_speed_std"       => std(speed_sl),
            "com_net_disp"        => net_disp,
            "com_path_length"     => path_len,
            "com_net_disp_ratio"  => net_disp_ratio,
            "omega_mean"          => mean(omega_sl),
            "omega_std"           => std(omega_sl),
            "omega_abs_mean"      => mean(abs.(omega_sl)),
            "omega_abs_std"       => std(abs.(omega_sl)),
            "omega_reliable_frac" => mean(rel_sl),
            "circ_mean"           => mean(circ_sl),
            "circ_std"            => std(circ_sl),
            "circ_half1_mean"     => mean(c1_sl),
            "circ_half1_std"      => std(c1_sl),
            "circ_half2_mean"     => mean(c2_sl),
            "circ_half2_std"      => std(c2_sl),
        )

        w_dict["motion_label"] = string(classify_motion_window(w_dict))
        windows[w] = w_dict
    end

    return windows
end

# ── Provisional motion classifier ─────────────────────────────────────────────

"""
    classify_motion_window(w::Dict) -> Symbol

Assign a provisional motion class to one analysis window.
Rules are evaluated top-to-bottom; first match wins.
Thresholds are taken from MOTION_THRESHOLDS and can be edited there.

Classes:
  :oscillatory  — high variability in COM speed (coefficient of variation)
  :rotating     — significant angular velocity with reliable shape
  :locked       — strong internal circulation with low COM drift
  :drifting     — significant net COM motion
  :arrested     — default (low speed, low rotation, low circulation)
"""
function classify_motion_window(w::Dict{String, Any})
    th    = MOTION_THRESHOLDS
    speed = w["com_speed_mean"]
    omega = abs(w["omega_mean"])
    circ  = abs(w["circ_mean"])
    cv    = w["com_speed_std"] / max(speed, 1e-12)
    rfrac = w["omega_reliable_frac"]

    # 1. Oscillatory: speed varies a lot relative to its mean
    cv > th[:osc_cv]                                        && return :oscillatory

    # 2. Rotating: sustained angular velocity with trustworthy shape axis
    (omega > th[:omega_thresh] && rfrac > 0.5)              && return :rotating

    # 3. Locked: strong internal flow, ring stays put
    (circ > th[:circ_locked] && speed < th[:drift_speed])   && return :locked

    # 4. Drifting: COM translates
    speed > th[:drift_speed]                                 && return :drifting

    # 5. Arrested: nothing significant happening
    return :arrested
end

"""
    classify_motion_run(windows) -> Symbol

Derive a run-level motion label by majority vote over all window labels.
If the dominant label covers less than (1 − switch_frac) of windows, the run
is labelled :switching to indicate unstable dynamics.
"""
function classify_motion_run(windows::Vector{Dict{String, Any}})
    n      = length(windows)
    counts = Dict{Symbol, Int}()
    for w in windows
        l = Symbol(w["motion_label"])
        counts[l] = get(counts, l, 0) + 1
    end
    dominant  = argmax(counts)
    dom_frac  = counts[dominant] / n
    (1.0 - dom_frac) > MOTION_THRESHOLDS[:switch_frac] && return :switching
    return dominant
end

# ── Run-level summary ─────────────────────────────────────────────────────────

"""
    motion_run_summary(frame_obs, windows, p, meta; morph_label="unknown") -> Dict

Collect run-level motion statistics and the provisional label into a Dict
suitable for JSON serialisation.

`morph_label` is an optional morphology class string produced by
`classify_morphology`.  Defaults to "unknown" for backward compatibility.
"""
function motion_run_summary(frame_obs, windows, p, meta;
                            morph_label::String = "unknown")
    run_label = classify_motion_run(windows)

    # total COM displacement and path length over the whole run
    Δx = frame_obs.com_x_unwrapped[end] - frame_obs.com_x_unwrapped[1]
    Δy = frame_obs.com_y_unwrapped[end] - frame_obs.com_y_unwrapped[1]
    net_disp = sqrt(Δx^2 + Δy^2)

    x_uw = frame_obs.com_x_unwrapped
    y_uw = frame_obs.com_y_unwrapped
    path_len = sum(sqrt((x_uw[t] - x_uw[t-1])^2 + (y_uw[t] - y_uw[t-1])^2)
                   for t in 2:lastindex(x_uw))
    net_disp_ratio = path_len > 1e-10 ? net_disp / path_len : 0.0

    # abs-ω stats (skip frame 1 which is defined as 0)
    omega_valid = frame_obs.omega_shape[2:end]

    return Dict{String, Any}(
        "run_id"                 => meta["run_id"],
        "motion_label"           => string(run_label),
        "morph_label"            => morph_label,
        "n_frames"               => length(frame_obs.com_speed),
        "n_windows"              => length(windows),
        "com_speed_mean"         => mean(frame_obs.com_speed[2:end]),
        "com_speed_std"          => std(frame_obs.com_speed[2:end]),
        "com_total_displacement" => net_disp,
        "com_path_length"        => path_len,
        "com_net_disp_ratio"     => net_disp_ratio,
        "omega_global_mean"      => mean(omega_valid),
        "omega_global_std"       => std(omega_valid),
        "omega_abs_mean"         => mean(abs.(omega_valid)),
        "omega_abs_std"          => std(abs.(omega_valid)),
        "circ_global_mean"       => mean(frame_obs.circulation),
        "circ_global_std"        => std(frame_obs.circulation),
        "theta_reliable_frac"    => mean(frame_obs.theta_reliable),
        "window_labels"          => [w["motion_label"] for w in windows],
        "params"                 => Dict("f0" => p.f0, "kappa" => p.κ, "kBT" => p.kBT, "N" => p.N),
        "thresholds_used"        => Dict(string(k) => v for (k, v) in MOTION_THRESHOLDS),
    )
end

end # module MotionAnalysis
