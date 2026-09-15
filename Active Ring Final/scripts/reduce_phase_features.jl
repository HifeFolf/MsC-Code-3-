# reduce_phase_features.jl
# Per-run feature reduction for phase-classifier scoring, cached one CSV per
# results folder.  Reads ONLY existing sweep outputs — no engine, no new runs.
#
# Why this exists:
#   Scoring a classifier means re-labelling thousands of runs every time a
#   threshold moves.  Re-reading observables.jld2 each pass makes that a
#   minutes-long loop and discourages the iteration the scoring is FOR.  So the
#   expensive part (trajectory → scalars) is done once here, and every candidate
#   classifier becomes a pure function of one CSV row.
#
#   The feature set is deliberately a SUPERSET of what the three existing
#   classifiers need, so they can be scored against each other on identical
#   input:
#     • route model      (phase_diagram_kconn_states.jl) — dev, f_in/f_out, Rg, bstd
#     • final-state model(phase_diagram_final_state.jl)  — *_fin columns
#     • persistence model(phase_classification.jl)       — K_std, persist
#
# Window convention (see the times-vector gotcha below):
#   Runs are 7.5 s of sim-time — 1500 frames × substeps 100 × dt 5e-5 = 0.005 s
#   per frame.  metadata.json "framerate": 60 is the MOVIE playback rate and says
#   nothing about data spacing.  Every window here is therefore sliced from the
#   `times` vector, never from a frame count.
#
#   TAIL_SECONDS = 5.0 gives t ∈ [2.5, 7.5] s — the validated post-burn-in window.
#   (phase_diagram_kconn_states.jl asks for 15 s, which exceeds the run length, so
#   its settling gate never actually engages; it reports this, but the effect is
#   that it classifies the whole post-transient trajectory.  Scoring needs the
#   gate to mean something, so the tail is set to a value that fits the run.)
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/reduce_phase_features.jl
#   julia --project scripts/reduce_phase_features.jl Results_N17_circle_indep
#   julia --project scripts/reduce_phase_features.jl Results_N17_circle_targeted
#   julia --project scripts/reduce_phase_features.jl --all      # both scoring sets
#   julia --project scripts/reduce_phase_features.jl <folder> --force   # ignore cache
#
# Output:  results/<folder>/phase_features.csv   (one row per run)

using Statistics, Printf, TOML, JLD2, JSON3

const PROJECT_ROOT_RPF = joinpath(@__DIR__, "..")

# ── Window / feature constants ───────────────────────────────────────────────
const SKIP_SECONDS   = 0.5    # drop IC-relaxation transient
const TAIL_SECONDS   = 5.0    # settle window = last 5 s  (t ∈ [2.5, 7.5])
const FINAL_SECONDS  = 1.0    # "state at the buzzer" window
const B_WIN_SECONDS  = 0.25   # sub-window for LOCAL shape-fluctuation measure
const RG_CURLED_FRAC = 0.70   # Rg < this × (N/2π) ⇒ folded onto itself
const PERSIST_LAG_S  = 0.30   # autocorrelation lag in SECONDS (= 60 frames at
                              # 0.005 s/frame, the validated lag from
                              # reduce_kconn.jl / phase_classification.jl)

# The two folders that together form the scoring set:
#   _indep    — 1500 cells × 1 independent seed  → neighbour coherence
#   _targeted — 243 cells × 10 independent seeds → per-cell reproducibility
const SCORING_FOLDERS = ["Results_N17_circle_indep", "Results_N17_circle_targeted"]

# ── Analysis cutoff (--cutoff T) ─────────────────────────────────────────────
# Reduce as if each run had ENDED at T seconds of sim-time; NaN = use the whole
# run.  A 30 s trajectory contains its own 7.5 s run, so reducing one folder at
# two cutoffs and comparing the label distributions is a direct test of whether
# the settled states are true attractors or long-lived metastable ones — i.e.
# whether the original 7.5 s window was ever long enough.
const CUTOFF_S = let i = findfirst(==("--cutoff"), ARGS)
    (i === nothing || i == length(ARGS)) ? NaN : parse(Float64, ARGS[i + 1])
end

# ── Small helpers ────────────────────────────────────────────────────────────

"Lag-`lag` autocorrelation of `x`. NaN if too short or degenerate."
function autocorr(x::AbstractVector, lag::Int)
    n = length(x)
    (lag < 1 || n <= lag + 1) && return NaN
    a = @view x[1:(n - lag)]
    b = @view x[(1 + lag):n]
    sa = std(a); sb = std(b)
    (sa == 0 || sb == 0 || isnan(sa) || isnan(sb)) && return NaN
    return mean((a .- mean(a)) .* (b .- mean(b))) / (sa * sb)
end

"""
Median std of `x` over sliding windows of `w` samples.

LOCAL, not global, on purpose: a slow monotonic drift (the ring elongating, or
folding) inflates a global std even though nothing is fluctuating.  The local
measure separates a slitherer from a slow drift by ~7×, where the global one
does not separate them at all.
"""
function local_std(x::AbstractVector, w::Int)
    n = length(x)
    (n < 2 || w < 2) && return n < 2 ? NaN : std(x)
    n < 2w && return std(x)
    step = max(1, w ÷ 2)
    return median(std(@view x[i:(i + w - 1)]) for i in 1:step:(n - w + 1))
end

"First index at or after time `t0`, clamped to a valid range."
win_start(t, t0) = min(something(findfirst(≥(t0), t), 1), max(1, length(t) - 1))

csvfmt(x::Bool) = x ? "1" : "0"
csvfmt(x::Integer) = string(x)
csvfmt(x::AbstractFloat) = isnan(x) ? "NaN" : @sprintf("%.8g", x)
csvfmt(x) = string(x)

const COLUMNS = ["folder", "run", "seed", "kappa", "f0", "kbt", "N", "eq_deg", "t_end", "n_frames",
                 "K_mean", "K_std", "dev", "f_in", "f_out", "persist",
                 "Rg_med", "bstd", "vratio",
                 "dev_fin", "Rg_fin", "bstd_fin",
                 "folded_end", "t_col"]

# ── Per-run reduction ────────────────────────────────────────────────────────

"""
Reduce one run directory to a NamedTuple of scalar features, or `nothing` if the
run has no usable Kconn data.

All windows are sliced from the `times` vector so the reduction survives any
change of frame count or substep spacing.
"""
function reduce_run(run_path::String, folder::String, run_name::String)
    obs_p = joinpath(run_path, "observables.jld2")
    par_p = joinpath(run_path, "params.toml")
    (isfile(obs_p) && isfile(par_p)) || return nothing

    ep = TOML.parsefile(par_p)["params"]
    N  = Int(ep["N"])
    eq = 360.0 / N

    d = jldopen(obs_p, "r") do f
        (haskey(f, "Kconn") && haskey(f, "Rg") && haskey(f, "b")) && return (
            t  = haskey(f, "times") ? f["times"] : collect(1:length(f["Kconn"])),
            K  = rad2deg.(f["Kconn"]),
            Rg = f["Rg"],
            b  = f["b"],
            vc = haskey(f, "v_com")  ? f["v_com"]  : nothing,
            vm = haskey(f, "v_mean") ? f["v_mean"] : nothing,
        )
        return nothing
    end
    d === nothing && return nothing
    # Truncate as if the run had ENDED at CUTOFF_S.  Every window below is
    # expressed relative to n / d.t[n], so this one line makes the whole
    # reduction behave exactly as it would on a shorter run — which is what lets
    # a 30 s trajectory be labelled at 7.5 s and at 30 s and the two compared.
    n = isnan(CUTOFF_S) ? length(d.K) :
        min(length(d.K), searchsortedlast(d.t, CUTOFF_S))
    n < 10 && return nothing

    t_end = d.t[n]
    dt    = n > 1 ? (t_end - d.t[1]) / (n - 1) : 1.0
    wb    = max(3, round(Int, B_WIN_SECONDS / dt))
    lag   = max(1, round(Int, PERSIST_LAG_S / dt))

    # seed: prefer seed_effective (post-fix, per-run), fall back to base seed
    seed = -1
    meta_p = joinpath(run_path, "metadata.json")
    if isfile(meta_p)
        try
            m = JSON3.read(read(meta_p, String))
            seed = haskey(m, :seed_effective) ? Int(m[:seed_effective]) :
                   haskey(m, :seed)           ? Int(m[:seed]) : -1
        catch
            seed = -1
        end
    end

    # ── settle window: last TAIL_SECONDS, floored past the transient ─────────
    i0 = win_start(d.t, max(SKIP_SECONDS, t_end - TAIL_SECONDS))
    Kw = @view d.K[i0:n]
    K_mean = mean(Kw)
    K_std  = std(Kw)
    dev    = K_mean - eq
    f_in   = count(<(5.0),       Kw) / length(Kw)   # "well inward"  sample fraction
    f_out  = count(>(eq + 20.0), Kw) / length(Kw)   # "well outward" sample fraction
    persist = autocorr(Kw, lag)
    Rg_med  = median(@view d.Rg[i0:n])
    bstd    = local_std(@view(d.b[i0:n]), wb)

    # Translation fraction: median COM speed / median mean bead speed.
    # Median, not mean — the periodic wrap shifts the ring by L in a single frame
    # and would otherwise inject a huge spurious COM-speed spike.
    vratio = NaN
    if d.vc !== nothing && d.vm !== nothing
        vcm = median(@view d.vc[i0:n]); vmm = median(@view d.vm[i0:n])
        vratio = vmm > 0 ? vcm / vmm : NaN
    end

    # ── final window: the state at the buzzer ────────────────────────────────
    j0 = win_start(d.t, t_end - FINAL_SECONDS)
    dev_fin  = mean(@view d.K[j0:n]) - eq
    Rg_fin   = median(@view d.Rg[j0:n])
    bstd_fin = local_std(@view(d.b[j0:n]), wb)

    # ── collapse time: first entry into the folded state that LASTS to the end.
    # Scanned over the WHOLE trajectory, not the tail — it is a diagnostic of
    # WHEN the fold began, which routinely predates any analysis window.
    thr = RG_CURLED_FRAC * (N / (2π))
    folded_end = d.Rg[n] < thr
    t_col = NaN
    if folded_end
        k = n
        while k > 1 && d.Rg[k - 1] < thr
            k -= 1
        end
        t_col = d.t[k]
    end

    return (folder = folder, run = run_name, seed = seed,
            kappa = Float64(ep["kappa"]), f0 = Float64(ep["f0"]),
            kbt = Float64(ep["kBT"]), N = N,
            eq_deg = eq, t_end = t_end, n_frames = n,
            K_mean = K_mean, K_std = K_std, dev = dev, f_in = f_in, f_out = f_out,
            persist = persist, Rg_med = Rg_med, bstd = bstd, vratio = vratio,
            dev_fin = dev_fin, Rg_fin = Rg_fin, bstd_fin = bstd_fin,
            folded_end = folded_end, t_col = t_col)
end

# ── Folder driver ────────────────────────────────────────────────────────────

function reduce_folder(folder::String; force::Bool = false)
    results_dir = joinpath(PROJECT_ROOT_RPF, "results", folder)
    isdir(results_dir) || error("Results folder not found: $results_dir")
    out_csv = joinpath(results_dir, isnan(CUTOFF_S) ? "phase_features.csv" :
                                    @sprintf("phase_features_t%g.csv", CUTOFF_S))

    if isfile(out_csv) && !force
        nrows = max(0, countlines(out_csv) - 1)
        println("✓ $folder — cache present ($nrows rows). Use --force to rebuild.")
        return out_csv
    end

    run_dirs = sort(filter(d -> occursin(r"^run_\d+$", d), readdir(results_dir)))
    isempty(run_dirs) && error("No run_* directories in $results_dir")
    println("Reducing $(length(run_dirs)) runs from $folder …")

    rows = String[]
    nok = 0
    for (i, rd) in enumerate(run_dirs)
        r = reduce_run(joinpath(results_dir, rd), folder, rd)
        if r !== nothing
            push!(rows, join((csvfmt(getfield(r, Symbol(c))) for c in COLUMNS), ","))
            nok += 1
        end
        if i % 250 == 0
            @printf("  %5d / %d  (%d usable)\n", i, length(run_dirs), nok)
        end
    end

    open(out_csv, "w") do io
        println(io, join(COLUMNS, ","))
        for r in rows
            println(io, r)
        end
    end
    @printf("✓ %s — wrote %d / %d runs → %s\n",
            folder, nok, length(run_dirs), relpath(out_csv, PROJECT_ROOT_RPF))
    return out_csv
end

# ── Entry point ──────────────────────────────────────────────────────────────

force = "--force" in ARGS
# Drop flags AND any value consumed by a flag (e.g. the T in `--cutoff 7.5`),
# so a cutoff value is never mistaken for a folder name.
targets = String[]
let skip_next = false          # `let` keeps this a true local; a bare top-level
    for a in ARGS              # loop variable would hit Julia's soft-scope rule
        if skip_next
            skip_next = false
        elseif startswith(a, "--")
            a == "--cutoff" && (skip_next = true)
        else
            push!(targets, a)
        end
    end
end
if isempty(targets) || "--all" in ARGS
    targets = SCORING_FOLDERS
end

println("Window: last $(TAIL_SECONDS) s (floor $(SKIP_SECONDS) s) · " *
        "final $(FINAL_SECONDS) s · local-b $(B_WIN_SECONDS) s · " *
        "persist lag $(PERSIST_LAG_S) s")
println("Cutoff: $(isnan(CUTOFF_S) ? "full run" : "$(CUTOFF_S) s (run treated as ending there)")\n")
for f in targets
    reduce_folder(f; force = force)
end
