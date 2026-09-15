# phase_diagram_kconn_categorical.jl
# Categorical comparison of the two ICs (circle vs quadkinked) built on the
# extended reduce_kconn.jl CSVs.  The point of the figure is the HEADLINE
# agreement panel: where the two initial conditions settle into DIFFERENT phases
# at the same (κ, f₀) is the hysteresis / bistability region.
#
# Classification per cell (single pooled cut, so identical physics cannot be
# labelled differently in the two rows):
#   persistence > pooled-Otsu cut → DYNAMIC
#         crossing_rate > 0 → switching     (hops between inward & outward bands)
#         else              → coherent      (sustained oscillation / rotation)
#   otherwise                     → STATIC, structural label = dominant occupancy
#         P_circ largest → circular
#         P_in   largest → inward
#         P_out  largest → outward
#
# Persistence is the CSV's persistence_Kconn column, which reduce_kconn.jl now
# computes at the standard 60-frame lag (= 1 movie-second @60 fps = 0.3 sim-units).
# One Otsu threshold is taken on the POOLED persistence of BOTH ICs together and
# applied to both — per-folder cuts would let the same physics get different
# labels across rows and corrupt the comparison.
#
# House style (shared with the other phase scripts, which this does NOT modify):
#   log-log axes, LogTicks(WilkinsonTicks(6)), categorical cgrad, nan_color for
#   missing cells (holes tolerated, never interpolated), linear-stability white
#   dashed overlay where the boundary CSV exists, px_per_unit = 2.
#
# Reads only results/kconn_phase/{circle,quadkinked}.csv (seed SEED).  The seed
# key stays in the CSV for later multi-seed work; this script plots one seed.
#
# Usage (from the Active Ring Final/ project root, after reduce_kconn.jl):
#   julia --project scripts/phase_diagram_kconn_categorical.jl

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_CC = joinpath(@__DIR__, "..")
const OUT_DIR         = joinpath(PROJECT_ROOT_CC, "results", "kconn_phase")
const SEED            = 42

# Categories 1..5 and their categorical colours / labels (house palette).
const CAT_COLORS = [:gainsboro, :darkorange, :steelblue, :crimson, :purple]
const CAT_LABELS = ["circular (static)", "inward (static)", "outward (static)",
                    "coherent (dynamic)", "switching (dynamic)"]
const CAT_SHORT  = ["circular", "inward", "outward", "coherent", "switching"]

# Agreement cells are neutral; disagreement pairs get distinct colours drawn
# from a categorical colormap sized to however many pairs the data produces.
const AGREE_COLOR = :gray85

# ── Reused verbatim from phase_classification.jl: Otsu 1-D threshold ──────────
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

# ── Load one IC's CSV rows (seed SEED) as records ─────────────────────────────
function load_ic(ic::AbstractString)
    path = joinpath(OUT_DIR, "$(ic).csv")
    isfile(path) || error("Missing cache $path — run reduce_kconn.jl first.")
    lines = filter(!isempty, readlines(path))
    hdr   = split(lines[1], ",")
    ci(name) = findfirst(==(name), hdr)
    recs = NamedTuple[]
    for ln in lines[2:end]
        v = split(ln, ",")
        parse(Int, v[ci("seed")]) == SEED || continue
        push!(recs, (
            κ     = parse(Float64, v[ci("kappa")]),
            f0    = parse(Float64, v[ci("f0")]),
            per   = parse(Float64, v[ci("persistence_Kconn")]),   # "NaN" → NaN
            Pout  = parse(Float64, v[ci("P_out")]),
            Pcirc = parse(Float64, v[ci("P_circ")]),
            Pin   = parse(Float64, v[ci("P_in")]),
            cr    = parse(Float64, v[ci("crossing_rate")]),
            N     = parse(Int,     v[ci("N")]),
        ))
    end
    isempty(recs) && error("No rows for seed $SEED in $path")
    return recs
end

# ── Classify one cell given the pooled persistence cut τ ──────────────────────
# NaN persistence (constant/frozen series) falls through to the static branch.
function classify(r, τ)
    if !isnan(r.per) && r.per > τ
        return r.cr > 0 ? 5 : 4                       # switching vs coherent
    end
    m = max(r.Pcirc, r.Pin, r.Pout)
    r.Pcirc == m && return 1                          # circular
    r.Pin   == m && return 2                          # inward
    return 3                                          # outward
end

# ── Load both ICs & the single pooled Otsu cut ────────────────────────────────
println("Loading extended reductions (seed $SEED)…")
circ = load_ic("circle")
quad = load_ic("quadkinked")

pooled_per = filter(!isnan, Float64[getfield.(circ, :per); getfield.(quad, :per)])
τ = otsu_threshold(pooled_per)
@printf("Pooled Otsu cut on persistence (60-frame lag, %d cells): %.3f\n",
        length(pooled_per), τ)

# ── Common (κ, f₀) grid (both ICs share the same log-spaced sweep) ─────────────
rk(x) = round(x; sigdigits = 10)                      # guard tiny float drift
κs  = sort(unique(rk.(Float64[getfield.(circ, :κ);  getfield.(quad, :κ)])))
f0s = sort(unique(rk.(Float64[getfield.(circ, :f0); getfield.(quad, :f0)])))
nκ, nf = length(κs), length(f0s)
ik = Dict(v => i for (i, v) in enumerate(κs))
jf = Dict(v => i for (i, v) in enumerate(f0s))
N_val = circ[1].N

function code_matrix(recs, τ)
    M = fill(NaN, nκ, nf)                             # [κ, f0]; NaN = missing cell
    for r in recs
        M[ik[rk(r.κ)], jf[rk(r.f0)]] = classify(r, τ)
    end
    return M
end
Mc = code_matrix(circ, τ)
Mq = code_matrix(quad, τ)

# ── Agreement / disagreement (headline bistability panel) ─────────────────────
# Amat: NaN = cell missing in either IC; 1 = the two ICs agree; 1+k = the k-th
# disagreement pair.  Pairs are ordered by descending cell count so the dominant
# (largest) bistable pair leads both the colours and the legend.
function build_agreement(Mc, Mq)
    paircount = Dict{Tuple{Int,Int},Int}()
    disagree  = Tuple{Int,Int,Tuple{Int,Int}}[]   # (i, j, pair)
    Amat = fill(NaN, size(Mc))
    n_agree = 0; n_common = 0
    for i in axes(Mc, 1), j in axes(Mc, 2)
        (isnan(Mc[i, j]) || isnan(Mq[i, j])) && continue
        n_common += 1
        a, b = Int(Mc[i, j]), Int(Mq[i, j])
        if a == b
            Amat[i, j] = 1.0; n_agree += 1
        else
            p = (a, b)
            paircount[p] = get(paircount, p, 0) + 1
            push!(disagree, (i, j, p))
        end
    end
    pairs = sort(collect(keys(paircount)); by = p -> (-paircount[p], p))
    pidx  = Dict(p => k for (k, p) in enumerate(pairs))
    for (i, j, p) in disagree
        Amat[i, j] = 1.0 + pidx[p]
    end
    return (Amat = Amat, pairs = pairs, paircount = paircount,
            n_common = n_common, n_agree = n_agree, n_disagree = n_common - n_agree)
end
ag = build_agreement(Mc, Mq)
Amat = ag.Amat; pairs = ag.pairs; paircount = ag.paircount
n_common = ag.n_common; n_agree = ag.n_agree; n_disagree = ag.n_disagree

# ── Shared linear-stability boundary loader (as in the other phase scripts) ───
function load_stability_boundary()
    for N_try in [N_val, 16, 17]
        csv_path = joinpath(PROJECT_ROOT_CC, "results", "linear_stability",
                            "phase_boundary_N$(N_try).csv")
        isfile(csv_path) || continue
        rows    = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
        κ_stab  = [parse(Float64, r[1]) for r in rows]
        f0_stab = [parse(Float64, r[2]) for r in rows]
        in_range = (f0_stab .>= f0s[1]) .& (f0_stab .<= f0s[end]) .&
                   (κ_stab  .>= κs[1]) .& (κ_stab  .<= κs[end])
        any(in_range) || continue
        println("  Stability boundary loaded (N=$N_try).")
        return (f0_stab[in_range], κ_stab[in_range])
    end
    return (nothing, nothing)
end
stab_f0, stab_κ = load_stability_boundary()
function overlay_stability!(ax)
    stab_κ === nothing && return
    lines!(ax, stab_f0, stab_κ; color = :white, linewidth = 2.5, linestyle = :dash)
end

# ── Reporting (quote the size of the bistable region) ─────────────────────────
println("\nPer-label cell counts (seed $SEED):")
@printf("  %-20s %8s %8s\n", "label", "circle", "quadkink")
for c in 1:5
    nc = count(==(Float64(c)), filter(!isnan, vec(Mc)))
    nq = count(==(Float64(c)), filter(!isnan, vec(Mq)))
    @printf("  %-20s %8d %8d\n", CAT_LABELS[c], nc, nq)
end
@printf("\nCommon cells: %d   agree: %d   DISAGREE (bistable): %d  (%.1f%% of common)\n",
        n_common, n_agree, n_disagree, 100 * n_disagree / max(n_common, 1))
if !isempty(pairs)
    println("Disagreement pairs (circle → quadkinked):")
    for p in pairs
        @printf("  %-11s → %-11s : %d cells\n", CAT_SHORT[p[1]], CAT_SHORT[p[2]], paircount[p])
    end
end

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (1500, 950))
Label(fig[0, 1:3],
      "Kconn categorical phase — circle vs quadkinked   (N=$N_val, kBT=0.1, seed $SEED)";
      fontsize = 15, font = :bold, tellwidth = false)

map_axis(pos, title) = Axis(fig[pos...];
    title  = title,
    xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
    xscale = log10, yscale = log10,
    xticks = LogTicks(WilkinsonTicks(6)),
    yticks = LogTicks(WilkinsonTicks(6)))

cat_cmap = cgrad(CAT_COLORS; categorical = true)

# (a) pooled persistence histogram + the single Otsu cut
per_c = filter(!isnan, getfield.(circ, :per))
per_q = filter(!isnan, getfield.(quad, :per))
bins  = range(minimum(pooled_per), maximum(pooled_per); length = 61)
ax_a = Axis(fig[1, 1];
            xlabel = "persistence  (autocorr of Kconn @ 60 frames = 1 movie-s)",
            ylabel = "cells",
            title  = "(a) Pooled persistence — one Otsu cut for both ICs")
hist!(ax_a, pooled_per; bins = bins, color = (:grey60, 0.45), label = "pooled")
stephist!(ax_a, per_c; bins = bins, color = :darkorange, label = "circle")
stephist!(ax_a, per_q; bins = bins, color = :seagreen,   label = "quadkinked")
vlines!(ax_a, [τ]; color = :crimson, linewidth = 2, label = @sprintf("Otsu cut = %.2f", τ))
axislegend(ax_a; position = :rt, framevisible = true, labelsize = 10)

# (b) circle categorical map
ax_b = map_axis((1, 2), "(b) Circle — categorical phase")
heatmap!(ax_b, f0s, κs, Mc';
         colormap = cat_cmap, colorrange = (0.5, 5.5), nan_color = :white)
overlay_stability!(ax_b)

# (c) quadkinked categorical map
ax_c = map_axis((2, 1), "(c) Quadkinked — categorical phase")
heatmap!(ax_c, f0s, κs, Mq';
         colormap = cat_cmap, colorrange = (0.5, 5.5), nan_color = :white)
overlay_stability!(ax_c)

# Shared 5-category legend (applies to b and c)
Legend(fig[1, 3],
       [PolyElement(color = c) for c in CAT_COLORS], CAT_LABELS, "Phase";
       framevisible = true)

# (d) agreement / bistability map (headline)
npairs      = length(pairs)
pair_colors = npairs == 0 ? Vector{RGBAf}() :
              cgrad(:tab20, max(npairs, 2); categorical = true).colors[1:npairs]
agr_cols    = [Makie.to_color(AGREE_COLOR); pair_colors]
ax_d = map_axis((2, 2), "(d) Agreement — disagreement = bistable region")
heatmap!(ax_d, f0s, κs, Amat';
         colormap = cgrad(agr_cols; categorical = true),
         colorrange = (0.5, npairs + 1.5), nan_color = :white)
overlay_stability!(ax_d)

# Agreement legend: "agree" + one entry per disagreement pair
agr_elems  = [PolyElement(color = AGREE_COLOR)]
agr_labels = ["agree ($(n_agree))"]
for (k, p) in enumerate(pairs)
    push!(agr_elems, PolyElement(color = pair_colors[k]))
    push!(agr_labels, "$(CAT_SHORT[p[1]]) → $(CAT_SHORT[p[2]]) ($(paircount[p]))")
end
Legend(fig[2, 3], agr_elems, agr_labels, "circle → quadkinked"; framevisible = true)

Label(fig[3, 1:3],
      "White dashed = linear-stability boundary (where the N=$N_val CSV exists). " *
      "Missing cells left blank (never interpolated).";
      fontsize = 10, color = :grey30, tellwidth = false)

colgap!(fig.layout, 10)
out_path = joinpath(OUT_DIR, "phase_diagram_kconn_categorical.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
