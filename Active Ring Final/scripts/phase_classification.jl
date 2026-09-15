# phase_classification.jl
# Phase map built from a PARAMETER COMBINATION rather than one hand-thresholded
# scalar, with the dynamical cut taken from a real valley in the data.
#
# Why a sliding window (not std over the whole run / its tail):  the curled state
# is an ABSORBING TRAP, so a run can oscillate hard and then LOCK for the rest of
# the sim.  A whole-run/tail std is then diluted to ≈0 by the frozen segment, and
# autocorrelation is *inflated* by the constant tail — both hide the runs that were
# genuinely dynamic.  The two axes below are computed on a 1 s sliding window and
# are robust to when (or whether) a run later locks:
#
#   amplitude       = log10 (peak window std)   — how far the junction swings
#   active-duration = time spent near that peak — whether the motion is SUSTAINED
#                       one-shot collapse    → short active-duration
#                       sustained oscillation → long active-duration
#
# Cuts (both data-driven, Otsu — no hand-picked magnitude):
#   moves ↔ frozen       :  Otsu on the amplitude distribution
#   sustained ↔ one-shot :  Otsu on active-duration among the movers   → Dynamic
#   structural split     :  ⟨Kconn⟩ vs 360/N  → inward / circular / outward
#
# Sampling rate is read from the times vector (≈200 Hz); the 60 fps in metadata is
# movie-only, so window sizes are set in SECONDS, not frame counts.
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/phase_classification.jl
#   julia --project scripts/phase_classification.jl Results_N17_circle 1 1500

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_PC = joinpath(@__DIR__, "..")
const SKIP_SECONDS    = 0.5    # drop brief IC-relaxation transient (sim-time)
const WIN_SECONDS     = 1.0    # sliding-window length for local std(Kconn)
const FROZE_EPS_DEG   = 3.0    # window std below this ⇒ junction "frozen"
const CIRC_BAND_DEG   = 8.0    # |⟨Kconn⟩−360/N| < this ⇒ circular

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_PC, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Otsu threshold: split a 1-D sample by maximising between-class variance ───
function otsu_threshold(x::AbstractVector{<:Real}; nbins::Int = 128)
    lo, hi = extrema(x)
    hi <= lo && return lo
    edges   = range(lo, hi; length = nbins + 1)
    centers = [(edges[i] + edges[i+1]) / 2 for i in 1:nbins]
    counts  = zeros(Int, nbins)
    for v in x
        b = clamp(searchsortedlast(edges, v), 1, nbins)
        counts[b] += 1
    end
    total   = sum(counts)
    sum_all = sum(counts[i] * centers[i] for i in 1:nbins)
    wB = 0; sumB = 0.0; best = -1.0; thr = centers[1]
    for i in 1:nbins-1
        wB += counts[i]
        wB == 0 && continue
        wF = total - wB
        wF == 0 && break
        sumB += counts[i] * centers[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        between = wB * wF * (mB - mF)^2
        if between > best
            best = between
            thr  = edges[i+1]
        end
    end
    return thr
end

# Activity metrics for one Kconn(t) series (degrees), robust to a frozen tail.
# Returns peak local std (deg), the time spent near that peak (s), and lock-in (s).
function activity_metrics(t::AbstractVector, Kdeg::AbstractVector)
    n  = length(Kdeg)
    dt = n > 1 ? (t[end] - t[1]) / (n - 1) : 1.0
    w  = max(2, round(Int, WIN_SECONDS / dt))
    sw = max(1, w ÷ 4)
    step_s = sw * dt
    i0 = something(findfirst(≥(SKIP_SECONDS), t), 1)

    sds = Float64[]; ctr = Float64[]
    i = i0
    while i + w - 1 ≤ n
        push!(sds, std(@view Kdeg[i:i+w-1]))
        push!(ctr, t[i + w ÷ 2])
        i += sw
    end
    isempty(sds) && return (peak = std(@view Kdeg[i0:n]), active = 0.0, lock = NaN)

    peak   = maximum(sds)
    active = count(≥(0.5 * peak), sds) * step_s          # time near the peak (s)
    lock   = NaN
    if sds[end] < FROZE_EPS_DEG
        k = length(sds)
        while k > 1 && sds[k-1] < FROZE_EPS_DEG; k -= 1; end
        lock = ctr[k]
    end
    return (peak = peak, active = active, lock = lock)
end

# ── Scan runs ────────────────────────────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder…")
println("Amplitude × active-duration on a $(WIN_SECONDS) s sliding window; skip $(SKIP_SECONDS) s transient.")

κ_raw   = Float64[]
f0_raw  = Float64[]
amp_raw = Float64[]   # log10 peak-window std(Kconn) [deg]  (amplitude axis)
act_raw = Float64[]   # active-duration [s]                 (sustained-ness axis)
dev_raw = Float64[]   # ⟨Kconn⟩ − 360/N  in degrees          (structural axis)
N_ring  = Ref(17)

for rd in run_dirs
    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    N_ring[] = Int(ep["N"])

    data = jldopen(obs_file, "r") do f
        haskey(f, "Kconn") || return nothing
        n = length(f["Kconn"])
        t = haskey(f, "times") ? f["times"] : collect(1:n)   # fallback: frame index
        (t = t, Kdeg = rad2deg.(f["Kconn"]))
    end
    data === nothing && continue
    length(data.Kdeg) < 10 && continue

    am = activity_metrics(data.t, data.Kdeg)
    i0 = something(findfirst(≥(SKIP_SECONDS), data.t), 1)
    K_eq_deg = 360.0 / N_ring[]

    push!(κ_raw,  Float64(ep["kappa"]))
    push!(f0_raw, Float64(ep["f0"]))
    push!(amp_raw, log10(max(am.peak, 1e-2)))
    push!(act_raw, am.active)
    push!(dev_raw, mean(@view data.Kdeg[i0:end]) - K_eq_deg)
end

n_cells = length(amp_raw)
println("Loaded $n_cells runs with Kconn data.")
n_cells == 0 && error("No Kconn data found — was this sweep run with run_sweep.jl?")

# ── Data-driven cuts: amplitude (moves?), then active-duration among movers ───
τ_amp  = otsu_threshold(amp_raw)
movers = amp_raw .> τ_amp
τ_act  = count(movers) ≥ 4 ? otsu_threshold(act_raw[movers]) : Inf
@printf("Otsu cut on amplitude  (log10 peak std) : %.3f  (= %.2f°)\n", τ_amp, 10^τ_amp)
@printf("Otsu cut on active-duration (movers)    : %.3f s\n", τ_act)

# ── Classify ─────────────────────────────────────────────────────────────────
# 1 = Dynamic   2 = Circular   3 = Inward-buckled   4 = Outward-buckled
function classify(amp, act, dev)
    (amp > τ_amp && act > τ_act) && return 1       # large AND sustained motion
    abs(dev) < CIRC_BAND_DEG     && return 2       # near equilibrium
    dev < 0                      && return 3       # inward
    return 4                                       # outward
end
codes = [classify(amp_raw[i], act_raw[i], dev_raw[i]) for i in 1:n_cells]

phase_colors = [:crimson, :gainsboro, :darkorange, :steelblue]
phase_labels = ["Dynamic (sustained)", "Circular (static)",
                "Inward-buckled (static)", "Outward-buckled (static)"]
for c in 1:4
    @printf("  %-26s %d cells\n", phase_labels[c], count(==(c), codes))
end

# ── κ × f₀ grid ──────────────────────────────────────────────────────────────
kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f0_raw))
nκ = length(kappa_vals); nf = length(f0_vals)
code_lookup = Dict(zip(zip(κ_raw, f0_raw), codes))
# active-duration is only meaningful for movers — a flat run trivially has every
# window "near its (tiny) peak", so mask non-movers to NaN (drawn grey).
actm_raw    = [movers[i] ? act_raw[i] : NaN for i in 1:n_cells]
act_lookup  = Dict(zip(zip(κ_raw, f0_raw), actm_raw))
code_mat = fill(NaN, nκ, nf)
act_mat  = fill(NaN, nκ, nf)
for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
    haskey(code_lookup, (κ, f₀)) && (code_mat[ic, ir] = code_lookup[(κ, f₀)])
    haskey(act_lookup,  (κ, f₀)) && (act_mat[ic, ir]  = act_lookup[(κ, f₀)])
end

# ── Linear stability boundary (optional) ─────────────────────────────────────
function load_stability_boundary()
    for N_try in [N_ring[], 16, 17]
        csv_path = joinpath(PROJECT_ROOT_PC, "results", "linear_stability",
                            "phase_boundary_N$(N_try).csv")
        isfile(csv_path) || continue
        rows    = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
        κ_stab  = [parse(Float64, r[1]) for r in rows]
        f0_stab = [parse(Float64, r[2]) for r in rows]
        in_range = (f0_stab .>= f0_vals[1]) .& (f0_stab .<= f0_vals[end]) .&
                   (κ_stab  .>= kappa_vals[1]) .& (κ_stab  .<= kappa_vals[end])
        any(in_range) || continue
        println("  Stability boundary loaded (N=$N_try).")
        return (f0_stab[in_range], κ_stab[in_range])
    end
    return (nothing, nothing)
end
stab_f0, stab_κ = load_stability_boundary()
function overlay_stability!(axis)
    stab_κ === nothing && return
    lines!(axis, stab_f0, stab_κ; color = :white, linewidth = 2.5,
           linestyle = :dash, label = "Linear stability")
    axislegend(axis; position = :lt, backgroundcolor = (:black, 0.4),
               labelcolor = :white)
end

# ── Figure ────────────────────────────────────────────────────────────────────
N_val = N_ring[]
fig = Figure(size = (1150, 1450))
Label(fig[0, 1:2], "Phase classification by amplitude × active-duration — $subfolder  (N=$N_val)";
      fontsize = 15, tellwidth = false)

# Panel 1: active-duration distribution among movers + Otsu cut
ax1 = Axis(fig[1, 1];
           xlabel = "active-duration  (s near peak std, movers only)",
           ylabel = "cells",
           title  = "Sustained ↔ one-shot cut among movers")
if any(movers)
    hist!(ax1, act_raw[movers]; bins = 40, color = (:steelblue, 0.7))
    isfinite(τ_act) && vlines!(ax1, [τ_act]; color = :crimson, linewidth = 2,
                               label = @sprintf("Otsu cut (%.2f s)", τ_act))
    isfinite(τ_act) && axislegend(ax1; position = :rt)
end

# Panel 2: the parameter-combination plane (amplitude × active-duration)
ax2 = Axis(fig[1, 2];
           xlabel = "amplitude   log₁₀ peak std(Kconn) [deg]",
           ylabel = "active-duration  [s]",
           title  = "Feature plane — dynamic cells separate top-right")
for c in 1:4
    sel = codes .== c
    any(sel) && scatter!(ax2, amp_raw[sel], act_raw[sel];
                         color = phase_colors[c], markersize = 5,
                         label = phase_labels[c])
end
vlines!(ax2, [τ_amp]; color = :black, linestyle = :dash)
isfinite(τ_act) && hlines!(ax2, [τ_act]; color = :black, linestyle = :dash)
axislegend(ax2; position = :lt, framevisible = true)

# Panel 3: continuous active-duration heatmap (dynamical order parameter)
ax3 = Axis(fig[2, 1];
           xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
           title  = "Active-duration among movers  (grey = frozen / circular)",
           xscale = log10, yscale = log10,
           xticks = LogTicks(WilkinsonTicks(6)),
           yticks = LogTicks(WilkinsonTicks(6)))
hm3 = heatmap!(ax3, f0_vals, kappa_vals, act_mat';
               colormap = :viridis, nan_color = :lightgrey)
Colorbar(fig[2, 1, Right()], hm3; label = "active-duration [s]", width = 14)
isfinite(τ_act) && contour!(ax3, f0_vals, kappa_vals, act_mat'; levels = [τ_act],
         color = :crimson, linewidth = 2)
overlay_stability!(ax3)

# Panel 4: categorical phase map
ax4 = Axis(fig[2, 2];
           xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
           title  = "Phase map  (amplitude × active-duration, structural = ⟨Kconn⟩)",
           xscale = log10, yscale = log10,
           xticks = LogTicks(WilkinsonTicks(6)),
           yticks = LogTicks(WilkinsonTicks(6)))
heatmap!(ax4, f0_vals, kappa_vals, code_mat';
         colormap = cgrad(phase_colors; categorical = true),
         colorrange = (0.5, 4.5), nan_color = :white)
Legend(fig[2, 2, Right()],
       [PolyElement(color = c) for c in phase_colors], phase_labels, "Phase";
       framevisible = true)
overlay_stability!(ax4)

out_path = joinpath(results_dir, "phase_classification.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
