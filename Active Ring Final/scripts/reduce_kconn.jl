# reduce_kconn.jl
# Per-run reduction of the passive-junction turning angle Kconn into one CSV row
# per run, cached per initial condition (IC) so the phase-diagram figure never
# has to re-read the raw observables.
#
# This script reads ONLY existing sweep outputs — no engine, no new simulations.
# For every run_XXXX it loads times + Kconn (radians) via IOUtils.load_observables,
# drops the burn-in (all frames with t < BURN_IN_T seconds, selected from the
# `times` vector so it survives framerate/length changes), and writes:
#
#   ic, kappa, f0, seed, N, mean_Kconn, std_Kconn, persistence_Kconn,
#   n_used, two_pi_over_N, sigma_th, P_out, P_circ, P_in, crossing_rate
#
#   mean_Kconn / std_Kconn  — temporal mean / std of Kconn over the post-burn-in
#                             window, in RADIANS (the raw unit of Kconn; the
#                             figure converts to degrees for display).
#   persistence_Kconn       — lag-PERSIST_LAG_FRAMES (=60-frame) autocorrelation of
#                             Kconn over the same window (the persistence axis).
#                             60 frames is the VALIDATED lag — the bimodal
#                             persistence with a real Otsu valley was established
#                             at 60 frames, so 60 frames is the standard and it
#                             matches phase_classification.jl exactly.
#   two_pi_over_N           — equilibrium turning angle 2π/N (radians); for a
#                             perfect circle every turning angle equals this.
#   sigma_th                — analytic thermal jitter of one turning angle (rad),
#                             sqrt(kBT·l/κ), from equipartition on the bending
#                             potential U = (κ/l)(1−cos θ): near equilibrium
#                             d²U/dθ² = κ/l, so ⟨δθ²⟩ = kBT·l/κ.  kBT and l are
#                             read from params.toml (never hardcoded).
#   P_out / P_circ / P_in   — fraction of post-burn-in frames with Kconn above /
#                             within / below a ±3·sigma_th band around 2π/N.  The
#                             band width is κ-dependent by construction (3σ_th(κ)),
#                             so there is no tuned constant.  The three sum to 1.
#   crossing_rate           — outer-band transitions per sim-unit of time, with the
#                             circular band as a buffer (see the helper below).
#
# Units sanity check: on the circle IC at high κ / low f0 the ring stays circular,
# so mean_Kconn should sit near two_pi_over_N (≈ 0.3696 rad ≈ 21.2° for N=17).
#
# Caching: one CSV per IC under results/kconn_phase/<ic>.csv, keyed by seed.
# Re-running a folder upserts that folder's seed (replaces its rows, leaves other
# seeds untouched) so adding seed 41/43/… later is an append, not a recompute,
# and multi-seed aggregation downstream is a plain groupby on the `seed` column.
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/reduce_kconn.jl                       # both seed-42 ICs
#   julia --project scripts/reduce_kconn.jl Results_N17_circle    # one folder
#   julia --project scripts/reduce_kconn.jl Results_N17_circle Results_N17_quadkinked

using Statistics, Printf, TOML

const PROJECT_ROOT_RK = joinpath(@__DIR__, "..")

# IOUtils.load_observables is the authoritative loader for times/Kconn; it needs
# EngineRing in Main (read-only — we never touch the engine or run anything).
include(joinpath(PROJECT_ROOT_RK, "src", "EngineRing.jl"))
using .EngineRing
include(joinpath(PROJECT_ROOT_RK, "src", "IOUtils.jl"))
using .IOUtils

# Burn-in: drop every frame with t < BURN_IN_T sim-units.  Documented in all three
# clocks so the two-clock (sim vs movie) ambiguity cannot recur:
#   2.5 sim-units  =  500 frames (dt_frame = 0.005 sim-units)  ≈  8.3 movie-seconds (@60 fps).
# Runs are 7.5 sim-units / 1500 frames, so this keeps the last 5 sim-units / 1000 frames.
const BURN_IN_T          = 2.5

# Persistence lag, defined in FRAMES (a fixed frame count, not a wall-clock time,
# so it survives the two-clock ambiguity and matches phase_classification.jl):
#   60 frames  =  1 movie-second (@60 fps)  =  0.3 sim-units (dt_frame = 0.005).
# 60 frames is the standard because the validated bimodal persistence (real Otsu
# valley) was established at 60 frames.
const PERSIST_LAG_FRAMES = 60
const OUT_DIR            = joinpath(PROJECT_ROOT_RK, "results", "kconn_phase")

# Exact column order for the cached CSV (leave seed in so multi-seed aggregation
# downstream is a groupby with no schema change).
const COLS = ["ic", "kappa", "f0", "seed", "N",
              "mean_Kconn", "std_Kconn", "persistence_Kconn",
              "n_used", "two_pi_over_N",
              "sigma_th", "P_out", "P_circ", "P_in", "crossing_rate"]

# ── Helpers ───────────────────────────────────────────────────────────────────

"IC label from the results folder name: strip the Results_N17_ prefix and any
 _seedNN suffix (seed itself comes from metadata.json, the authoritative source)."
function ic_label(folder::AbstractString)
    s = replace(folder, r"^Results_N\d+_" => "")
    s = replace(s, r"_seed\d+$" => "")
    return s
end

"Normalised lag-`lag` autocorrelation of x; NaN if undefined (too short / constant)."
function autocorr(x::AbstractVector{<:Real}, lag::Int)
    n = length(x)
    (lag < 1 || n <= lag + 1) && return NaN
    μ     = mean(x)
    denom = sum(abs2, x .- μ)
    denom <= 0 && return NaN                       # constant series → undefined
    num = 0.0
    @inbounds for i in 1:(n - lag)
        num += (x[i] - μ) * (x[i + lag] - μ)
    end
    return num / denom
end

"Occupancy fractions of the three ±3σ_th bands around `eqK`, and the outer-band
 crossing rate (crossings per unit of `tspan`).

 Bands (κ-dependent width via σ_th, no tuned constant):
     outward  :  x >  eqK + 3σ_th
     circular :  |x − eqK| ≤ 3σ_th          ← buffer zone
     inward   :  x <  eqK − 3σ_th

 Crossing rule: a crossing is counted ONLY when the series moves from one OUTER
 band to the OTHER outer band (outward↔inward).  The circular band is a buffer,
 so excursions into it and back to the SAME outer band do not count.  We remember
 the last outer band visited and increment only when a newly-entered outer band
 differs from it; frames in the circular band are skipped (they neither count nor
 reset the memory)."
function band_occupancy_and_crossings(x::AbstractVector{<:Real}, eqK::Real,
                                      σ_th::Real, tspan::Real)
    hi = eqK + 3σ_th
    lo = eqK - 3σ_th
    n  = length(x)
    n_out  = count(>(hi), x)
    n_in   = count(<(lo), x)
    n_circ = n - n_out - n_in

    last_outer = 0            # 0 = none yet, +1 = outward, −1 = inward
    n_cross    = 0
    @inbounds for v in x
        band = v > hi ? 1 : (v < lo ? -1 : 0)
        band == 0 && continue                       # circular buffer: skip
        (last_outer != 0 && band != last_outer) && (n_cross += 1)
        last_outer = band
    end
    rate = tspan > 0 ? n_cross / tspan : 0.0
    return (P_out = n_out / n, P_circ = n_circ / n, P_in = n_in / n,
            crossing_rate = rate)
end

"Reduce one run directory to a NamedTuple row, or `nothing` if it cannot be used."
function reduce_run(run_path::AbstractString, ic::AbstractString)
    params_file = joinpath(run_path, "params.toml")
    meta_file   = joinpath(run_path, "metadata.json")
    obs_file    = joinpath(run_path, "observables.jld2")
    (isfile(params_file) && isfile(meta_file) && isfile(obs_file)) || return nothing

    ep   = TOML.parsefile(params_file)["params"]
    κ    = Float64(ep["kappa"])
    f₀   = Float64(ep["f0"])
    N    = Int(ep["N"])
    kBT  = Float64(ep["kBT"])                       # from params.toml, never hardcoded
    l    = Float64(ep["l"])                         # from params.toml, never hardcoded
    seed = Int(IOUtils.load_metadata(run_path)["seed"])

    times, obs = IOUtils.load_observables(run_path)
    haskey(obs, "Kconn") || return nothing
    K = obs["Kconn"]

    # Burn-in by the times vector (not a frame index) so it survives length /
    # framerate changes.
    keep = findall(>=(BURN_IN_T), times)
    length(keep) < 2 && return nothing             # nothing usable after burn-in
    Kb = K[keep]
    tb = times[keep]

    # Analytic thermal jitter of one turning angle (rad), equipartition on the
    # bending potential U = (κ/l)(1−cos θ):  σ_th = sqrt(kBT·l/κ).
    eqK  = 2π / N
    σ_th = sqrt(kBT * l / κ)

    # Band occupancy + outer-band crossing rate, over the SAME post-burn-in window.
    # tspan is the window duration in sim-units (from the times vector).
    occ = band_occupancy_and_crossings(Kb, eqK, σ_th, tb[end] - tb[1])

    return (ic = ic, kappa = κ, f0 = f₀, seed = seed, N = N,
            mean_Kconn = mean(Kb), std_Kconn = std(Kb),
            persistence_Kconn = autocorr(Kb, PERSIST_LAG_FRAMES),
            n_used = length(Kb), two_pi_over_N = eqK,
            sigma_th = σ_th,
            P_out = occ.P_out, P_circ = occ.P_circ, P_in = occ.P_in,
            crossing_rate = occ.crossing_rate)
end

# ── CSV cache (upsert by seed) ────────────────────────────────────────────────

row_to_strings(r) = [string(r.ic), string(r.kappa), string(r.f0), string(r.seed),
                     string(r.N), string(r.mean_Kconn), string(r.std_Kconn),
                     string(r.persistence_Kconn), string(r.n_used),
                     string(r.two_pi_over_N),
                     string(r.sigma_th), string(r.P_out), string(r.P_circ),
                     string(r.P_in), string(r.crossing_rate)]

"Load an existing per-IC CSV as a vector of column→string dicts (empty if absent)."
function load_csv(path)
    isfile(path) || return Vector{Dict{String,String}}()
    lines = filter(!isempty, readlines(path))
    isempty(lines) && return Vector{Dict{String,String}}()
    header = split(lines[1], ",")
    out = Vector{Dict{String,String}}()
    for ln in lines[2:end]
        vals = split(ln, ",")
        push!(out, Dict(string(h) => string(v) for (h, v) in zip(header, vals)))
    end
    return out
end

"Write rows (column→string dicts) to CSV in the canonical column order."
function write_csv(path, rows)
    open(path, "w") do io
        println(io, join(COLS, ","))
        for r in rows
            println(io, join([get(r, c, "") for c in COLS], ","))
        end
    end
end

# ── Reduce one folder ─────────────────────────────────────────────────────────

function reduce_folder(folder::AbstractString)
    dir = joinpath(PROJECT_ROOT_RK, "results", folder)
    isdir(dir) || (@warn "Results folder not found, skipping" folder; return)
    ic = ic_label(folder)

    run_dirs = sort(filter(d -> occursin(r"^run_\d+$", d), readdir(dir, join = false)))
    @printf("\n%s  (ic = %s):  scanning %d run dirs…\n", folder, ic, length(run_dirs))

    rows   = NamedTuple[]
    n_skip = 0
    for rd in run_dirs
        r = reduce_run(joinpath(dir, rd), ic)
        r === nothing ? (n_skip += 1) : push!(rows, r)
    end
    isempty(rows) && (@warn "No usable runs in $folder"; return)

    seeds_here = sort(unique(getproperty.(rows, :seed)))
    @printf("  reduced %d runs  (%d skipped/failed),  seed(s): %s\n",
            length(rows), n_skip, join(seeds_here, ", "))

    # Upsert into the per-IC cache: drop existing rows for the seeds we just
    # recomputed, keep all other seeds, then append the fresh rows.
    mkpath(OUT_DIR)
    out_path = joinpath(OUT_DIR, "$(ic).csv")
    existing = load_csv(out_path)
    seedset  = Set(string.(seeds_here))
    kept     = filter(d -> !(get(d, "seed", "") in seedset), existing)
    new_dicts = [Dict(zip(COLS, row_to_strings(r))) for r in rows]
    merged    = vcat(kept, new_dicts)
    # tidy ordering: seed, then kappa, then f0
    sort!(merged; by = d -> (parse(Float64, d["seed"]),
                             parse(Float64, d["kappa"]),
                             parse(Float64, d["f0"])))
    write_csv(out_path, merged)
    @printf("  wrote %s  (%d rows total)\n", out_path, length(merged))

    sanity_report(rows)
    return out_path
end

# ── Sanity report (mean_Kconn vs 2π/N on the sampled grid) ────────────────────

function sanity_report(rows)
    κs  = sort(unique(getproperty.(rows, :kappa)))
    f0s = sort(unique(getproperty.(rows, :f0)))
    N   = rows[1].N
    eq  = 2π / N
    means = getproperty.(rows, :mean_Kconn)

    @printf("  grid: %d κ × %d f0   (2π/N = %.4f rad = %.2f°, N=%d)\n",
            length(κs), length(f0s), eq, rad2deg(eq), N)
    @printf("  mean_Kconn range: %.4f … %.4f rad  (%.1f° … %.1f°)\n",
            minimum(means), maximum(means), rad2deg(minimum(means)), rad2deg(maximum(means)))

    # Stiff / low-force corner: the circle should stay circular here → near 2π/N.
    κmax = maximum(κs); f0min = minimum(f0s)
    corner = findfirst(r -> r.kappa == κmax && r.f0 == f0min, rows)
    if corner !== nothing
        c = rows[corner]
        @printf("  corner κ=%.2f, f0=%.2f:  mean=%.4f rad (%.2f°)  std=%.4f  persist=%.3f  n=%d\n",
                c.kappa, c.f0, c.mean_Kconn, rad2deg(c.mean_Kconn),
                c.std_Kconn, c.persistence_Kconn, c.n_used)
    end
    near = count(m -> abs(m - eq) <= deg2rad(2.0), means)
    @printf("  %d / %d cells within ±2° of 2π/N\n", near, length(means))

    # Equipartition check: in the known-circular region (lowest-f0 decade) the
    # measured std(Kconn) should track the analytic σ_th = sqrt(kBT·l/κ).  σ_th
    # depends only on κ, so we report, per κ, the mean measured std over the
    # lowest-decade f0 cells alongside σ_th and their ratio.  A ratio persistently
    # >1 would mean the ±3σ_th band is too tight (correlations through ring
    # closure) and should be widened by that factor.
    f0decade = minimum(f0s) * 10.0
    low = filter(r -> r.f0 <= f0decade, rows)
    if !isempty(low)
        @printf("  equipartition check (lowest-f0 decade, f0 ≤ %.2f): std(Kconn) vs σ_th per κ\n",
                f0decade)
        for κ in κs
            cells = filter(r -> r.kappa == κ, low)
            isempty(cells) && continue
            std_meas = mean(getproperty.(cells, :std_Kconn))
            σ        = cells[1].sigma_th          # σ_th is κ-only, same across these f0
            @printf("    κ=%7.3f:  std=%.4f  σ_th=%.4f  ratio=%.2f  (%d f0 cells)\n",
                    κ, std_meas, σ, std_meas / σ, length(cells))
        end
    end
end

# ── Main ──────────────────────────────────────────────────────────────────────

folders = isempty(ARGS) ? ["Results_N17_circle", "Results_N17_quadkinked"] : ARGS
println("Reducing Kconn for: ", join(folders, ", "))
println("Burn-in: drop t < $(BURN_IN_T) sim-units (=500 frames ≈8.3 movie-s)   |   " *
        "persistence lag = $(PERSIST_LAG_FRAMES) frames (=1 movie-s @60fps =0.3 sim-units)")
for folder in folders
    reduce_folder(folder)
end
println("\nDone.")
