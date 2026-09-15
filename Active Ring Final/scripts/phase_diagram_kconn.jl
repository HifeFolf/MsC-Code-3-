# phase_diagram_kconn.jl
# Composite 2×2 phase diagram of the passive-junction turning angle Kconn,
# built from the cached per-IC reductions written by reduce_kconn.jl.
#
#   Rows    = initial condition   (circle on top, quadkinked on the bottom)
#   Columns = ⟨Kconn⟩ (left)  |  std(Kconn) (right)
#
# Axes: f0 (active force) on x, κ (bending stiffness) on y.  Log scaling is
# applied per-axis only when the sampled values are log-spaced — detected from
# the spacing in the data, never assumed.
#
# Colour:
#   • The two ⟨Kconn⟩ panels SHARE one diverging normalization centred on the
#     equilibrium turning angle 2π/N (≈ 21.2° for N=17): white = circular,
#     blue = outward buckled (+), red = inward buckled (−).
#   • The two std panels SHARE one sequential (viridis) normalization.  Default
#     is raw degrees with a robust upper clip (the std range is heavy-tailed);
#     flip USE_LOG_STD for a log scale that separates the thermal floor from
#     genuine junction motion.
#
# Missing / failed grid cells are left blank (white) — never interpolated.
#
# This script only READS results/kconn_phase/<ic>.csv.  It plots a single seed
# (SEED, default 42); when more seeds are added to the cache, seed aggregation
# is a groupby on the `seed` column with no change to this plotting code.
#
# Usage (from the Active Ring Final/ project root, after reduce_kconn.jl):
#   julia --project scripts/phase_diagram_kconn.jl

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_PK = joinpath(@__DIR__, "..")
const OUT_DIR         = joinpath(PROJECT_ROOT_PK, "results", "kconn_phase")

# ── Knobs ─────────────────────────────────────────────────────────────────────
const SEED           = 42              # which seed to plot from the cache
const SHOW_FLC_GUIDE = true            # dashed guide line at f0 = FL_C · κ
const FL_C           = 75.0            # flexure-number stability boundary Fl_c
const USE_LOG_STD    = false           # std panels: false = raw °, true = log °
const STD_CLIP_Q     = 0.98            # raw-std shared upper clip (heavy tail)
const STD_FLOOR      = 0.3             # ° lower clip when USE_LOG_STD = true

# IC rows, top → bottom: (cache name, display label)
const IC_ROWS = [("circle", "Circle"), ("quadkinked", "Quadkinked")]

# ── Detect log-sampling from the spacing of the sampled axis values ───────────
# Log-sampled ⇔ the log-spaced differences are more uniform than the linear ones.
function is_log_sampled(v::AbstractVector{<:Real})
    length(v) < 3 && return false
    any(<=(0), v) && return false
    cv(x) = (m = mean(x); m == 0 ? Inf : std(x) / abs(m))
    return cv(diff(log.(v))) < cv(diff(v))
end

# ── Load one IC's cached reduction into grid matrices (degrees) ───────────────
function load_ic(ic::AbstractString)
    path = joinpath(OUT_DIR, "$(ic).csv")
    isfile(path) || error("Missing cache $path — run reduce_kconn.jl first.")
    lines = filter(!isempty, readlines(path))
    hdr   = split(lines[1], ",")
    ci(name) = findfirst(==(name), hdr)

    κraw = Float64[]; fraw = Float64[]; mraw = Float64[]; sraw = Float64[]; Ns = Int[]
    for ln in lines[2:end]
        v = split(ln, ",")
        parse(Int, v[ci("seed")]) == SEED || continue
        push!(κraw, parse(Float64, v[ci("kappa")]))
        push!(fraw, parse(Float64, v[ci("f0")]))
        push!(mraw, parse(Float64, v[ci("mean_Kconn")]))
        push!(sraw, parse(Float64, v[ci("std_Kconn")]))
        push!(Ns,   parse(Int,     v[ci("N")]))
    end
    isempty(κraw) && error("No rows for seed $SEED in $path")

    κs = sort(unique(κraw)); fs = sort(unique(fraw))
    ik = Dict(v => i for (i, v) in enumerate(κs))
    jf = Dict(v => i for (i, v) in enumerate(fs))
    Mmean = fill(NaN, length(κs), length(fs))   # [κ, f0], in degrees
    Mstd  = fill(NaN, length(κs), length(fs))
    for (κ, f, m, s) in zip(κraw, fraw, mraw, sraw)
        Mmean[ik[κ], jf[f]] = rad2deg(m)
        Mstd[ik[κ], jf[f]]  = rad2deg(s)
    end

    nmiss = count(isnan, Mmean)
    @printf("  %-12s seed %d:  %d κ × %d f0  (%d cells, %d missing)  κ-log=%s f0-log=%s\n",
            ic, SEED, length(κs), length(fs), length(κs) * length(fs), nmiss,
            is_log_sampled(κs), is_log_sampled(fs))
    return (ic = ic, κ = κs, f0 = fs, Mmean = Mmean, Mstd = Mstd, N = Ns[1],
            klog = is_log_sampled(κs), flog = is_log_sampled(fs))
end

# ── Axis scale / ticks helpers ────────────────────────────────────────────────
scale_of(islog) = islog ? log10 : identity
ticks_of(islog) = islog ? LogTicks(WilkinsonTicks(5)) : Makie.automatic

# Dashed Fl_c guide:  f0 = FL_C · κ, clipped to the plotted (f0, κ) window.
function add_flc_guide!(ax, d)
    SHOW_FLC_GUIDE || return
    κlo = max(minimum(d.κ), minimum(d.f0) / FL_C)
    κhi = min(maximum(d.κ), maximum(d.f0) / FL_C)
    κlo < κhi || return                         # boundary outside sampled window
    κline = range(κlo, κhi; length = 64)
    lines!(ax, FL_C .* κline, collect(κline);
           color = :black, linewidth = 2, linestyle = :dash)
end

# ── Load both ICs ─────────────────────────────────────────────────────────────
println("Loading cached reductions (seed $SEED)…")
datas = [load_ic(ic) for (ic, _) in IC_ROWS]

# ── Shared colour normalizations across BOTH ICs ──────────────────────────────
all_mean = vcat([filter(!isnan, vec(d.Mmean)) for d in datas]...)
all_std  = vcat([filter(!isnan, vec(d.Mstd))  for d in datas]...)

N_val = datas[1].N
K_eq  = 360.0 / N_val                                   # equilibrium angle (deg)
half  = (isempty(all_mean) ? 20.0 : maximum(abs.(all_mean .- K_eq))) * 1.02
clo, chi = K_eq - half, K_eq + half

if USE_LOG_STD
    svmin = max(isempty(all_std) ? STD_FLOOR : minimum(all_std), STD_FLOOR)
    svmax = isempty(all_std) ? 1.0 : maximum(all_std)
else
    svmin = 0.0
    svmax = isempty(all_std) ? 1.0 : quantile(all_std, STD_CLIP_Q)
end

@printf("⟨Kconn⟩ shared range: %.1f°…%.1f°  (white = 2π/N = %.2f°)\n", clo, chi, K_eq)
@printf("std shared range: %.2f°…%.2f°  (%s)\n", svmin, svmax,
        USE_LOG_STD ? "log" : "raw, clipped at p$(round(Int,100*STD_CLIP_Q))")
SHOW_FLC_GUIDE && println("Fl_c guide: f0 = $(round(Int,FL_C))·κ (dashed)")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (1180, 900))
Label(fig[0, 1:4],
      "Passive-bead turning angle Kconn — phase diagram   (N=$N_val, kBT=0.1, seed $SEED)";
      fontsize = 15, font = :bold, tellwidth = false)

hm_mean_objs = Any[]   # collected so the shared colorbars can reference them
hm_std_objs  = Any[]
for (row, (d, (_, label))) in enumerate(zip(datas, IC_ROWS))
    # ── Left column: ⟨Kconn⟩ (diverging, centred on 2π/N) ─────────────────────
    ax_m = Axis(fig[row, 1];
                title  = "$label  ·  ⟨Kconn⟩",
                xlabel = "f₀  (active force)", ylabel = "κ  (bending stiffness)",
                xscale = scale_of(d.flog), yscale = scale_of(d.klog),
                xticks = ticks_of(d.flog), yticks = ticks_of(d.klog))
    hm_mean = heatmap!(ax_m, d.f0, d.κ, d.Mmean';
                       colormap = :RdBu, colorrange = (clo, chi), nan_color = :white)
    push!(hm_mean_objs, hm_mean)
    add_flc_guide!(ax_m, d)

    # ── Right column: std(Kconn) (sequential) ─────────────────────────────────
    ax_s = Axis(fig[row, 3];
                title  = "$label  ·  std(Kconn)",
                xlabel = "f₀  (active force)", ylabel = "κ  (bending stiffness)",
                xscale = scale_of(d.flog), yscale = scale_of(d.klog),
                xticks = ticks_of(d.flog), yticks = ticks_of(d.klog))
    Sdisp = USE_LOG_STD ? map(x -> isnan(x) ? NaN : max(x, STD_FLOOR), d.Mstd) : d.Mstd
    hm_std = heatmap!(ax_s, d.f0, d.κ, Sdisp';
                      colormap   = :viridis,
                      colorscale = USE_LOG_STD ? log10 : identity,
                      colorrange = (svmin, svmax), nan_color = :white)
    push!(hm_std_objs, hm_std)
    add_flc_guide!(ax_s, d)
end

# ── Shared colorbars (one per column-pair, spanning both rows) ────────────────
Colorbar(fig[1:2, 2], hm_mean_objs[1];
         label = "⟨Kconn⟩ at passive bead  (°)   — white = 2π/N (circular)", width = 16)
Colorbar(fig[1:2, 4], hm_std_objs[1];
         label = "std(Kconn)  (°)$(USE_LOG_STD ? ", log" : "")", width = 16)

colgap!(fig.layout, 8)

out_path = joinpath(PROJECT_ROOT_PK, "results", "phase_diagram_kconn.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
