# phase_diagram_confidence.jl
# The phase diagram rendered so that it carries its own uncertainty.
#
# Why this exists:
#   A modal-label map paints a 5-vs-5 coin flip in exactly the same flat colour
#   as a cell where all ten seeds agreed.  On this grid 45 of 1079 cells (4.2%)
#   are EXACT ties whose colour is therefore decided by directory sort order,
#   and 149 (13.8%) sit at or below the n=10 chance null of ~0.62.  The reader of
#   such a map cannot tell a decided cell from an undecided one.
#
#   The fix is not a better threshold — the underlying quantity is continuous.
#   So the seed agreement is rendered as a colour channel instead of being
#   collapsed into a category.
#
# LEFT — phase, with confidence.  Hue = modal label.  Saturation = confidence,
#   c = (agreement − 0.5) / 0.5 clamped to [0,1]: full colour when all seeds
#   agree, fading to neutral grey at a 50/50 split.  An ambiguous cell now LOOKS
#   ambiguous, and the arbitrary tie-break stops mattering because a tied cell
#   renders grey rather than committing to a hue.
#
# RIGHT — P(fold), no categories at all.  The fraction of seeds ending folded,
#   on a continuous blue→orange scale.  This is the probability field: the
#   honest object when the boundary is genuinely stochastic rather than sharp.
#   Cells where the slithering state takes ≥30% of seeds are ringed, since that
#   third outcome is invisible on a two-way fold/not-fold axis.
#
# Input is the per-run label dump (one row per run), so cells with any number of
# seeds work:
#   julia --project scripts/phase_reliability.jl route \
#         --grid <folder> --seeds <folder> --no-plot --dump-runs
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_confidence.jl
#   julia --project scripts/phase_diagram_confidence.jl Results_N17_phi_long
#   julia --project scripts/phase_diagram_confidence.jl Results_N17_phi_long phase_runlabels_route_t7.5.csv

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_PDC = joinpath(@__DIR__, "..")

folder     = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_phi_long"
labelsfile = length(ARGS) >= 2 ? ARGS[2] : "phase_runlabels_route.csv"

# Phase hues at full confidence.  Deliberately NOT aliceblue for the glider:
# that is #F0F8FF, indistinguishable from the white no-data wedge.
const HUE = Dict(1 => (0.86, 0.08, 0.24),    # slithering — crimson
                 2 => (0.95, 0.55, 0.10),    # crescent   — orange
                 3 => (0.27, 0.47, 0.72))    # glider     — blue
const GREY = (0.85, 0.85, 0.85)              # fully ambiguous
const NAMES = Dict(1 => "Slithering", 2 => "Crescent (folded)", 3 => "Ellipse / glider")
const SLITHER_MARK = 0.30                    # ring cells with ≥ this slither share

blend(h, c) = RGBAf(c * h[1] + (1 - c) * GREY[1],
                    c * h[2] + (1 - c) * GREY[2],
                    c * h[3] + (1 - c) * GREY[3], 1.0)

# ── Load per-run labels ──────────────────────────────────────────────────────
path = joinpath(PROJECT_ROOT_PDC, "results", folder, labelsfile)
isfile(path) || error("Missing $path\nGenerate it with phase_reliability.jl --dump-runs")
cells = Dict{Tuple{Float64,Float64}, Vector{Int}}()
for ln in readlines(path)[2:end]
    isempty(strip(ln)) && continue
    v = split(ln, ",")
    length(v) < 6 && continue
    push!(get!(cells, (parse(Float64, v[4]), parse(Float64, v[5],)), Int[]), parse(Int, v[6]))
end
isempty(cells) && error("No rows parsed from $path")

κs = sort(unique(k[1] for k in keys(cells)))
fs = sort(unique(k[2] for k in keys(cells)))
ki = Dict(v => i for (i, v) in enumerate(κs))
fi = Dict(v => i for (i, v) in enumerate(fs))

img   = fill(RGBAf(0, 0, 0, 0), length(fs), length(κs))   # [x=f, y=κ]
pfold = fill(NaN, length(fs), length(κs))
slith = Tuple{Float64,Float64}[]
n_tie = n_amb = 0

for (k, v) in cells
    n = length(v)
    cnt = Dict(l => count(==(l), v) for l in unique(v))
    m   = maximum(values(cnt))
    lab = minimum(l for (l, c) in cnt if c == m)      # deterministic tie-break
    agr = m / n
    count(==(m), collect(values(cnt))) > 1 && (global n_tie += 1)
    agr <= 0.6 && (global n_amb += 1)

    c = clamp((agr - 0.5) / 0.5, 0.0, 1.0)
    j, i = fi[k[2]], ki[k[1]]
    img[j, i]   = blend(HUE[lab], c)
    pfold[j, i] = get(cnt, 2, 0) / n
    get(cnt, 1, 0) / n >= SLITHER_MARK && push!(slith, k)
end

@printf("%s · %d cells · %d runs\n", folder, length(cells), sum(length, values(cells)))
@printf("  exact ties (colour was sort-order dependent) : %d (%.1f%%)\n", n_tie, 100n_tie / length(cells))
@printf("  agreement ≤ 0.6 (at/below chance)            : %d (%.1f%%)\n", n_amb, 100n_amb / length(cells))
@printf("  cells with ≥%.0f%% slithering seeds            : %d\n", 100SLITHER_MARK, length(slith))

# ── Figure ───────────────────────────────────────────────────────────────────
# Plotted in log10 space on linear axes: even spacing across decades, ticks
# relabelled to the original values.
logκ, logf = log10.(κs), log10.(fs)
ticks(v) = (range(minimum(v), maximum(v), length = 5),
            [@sprintf("%.3g", 10^x) for x in range(minimum(v), maximum(v), length = 5)])
xr = (minimum(logf), maximum(logf))
yr = (minimum(logκ), maximum(logκ))

fig = Figure(size = (1340, 620))

ax1 = Axis(fig[1, 1], xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
           title = "Phase — hue = state, saturation = seed agreement",
           xticks = ticks(logf), yticks = ticks(logκ))
image!(ax1, xr, yr, img; interpolate = false)
limits!(ax1, xr..., yr...)

# Legend teaches the encoding: each phase shown decided AND ambiguous.
Legend(fig[2, 1],
       vcat([PolyElement(color = blend(HUE[l], 1.0)) for l in (3, 2, 1)],
            [PolyElement(color = blend(HUE[3], 0.0))]),
       vcat([NAMES[l] for l in (3, 2, 1)], ["50/50 — undecided"]),
       orientation = :horizontal, framevisible = false, labelsize = 11, nbanks = 1)

ax2 = Axis(fig[1, 2], xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
           title = "P(fold) — fraction of seeds ending folded",
           xticks = ticks(logf), yticks = ticks(logκ))
hm = heatmap!(ax2, logf, logκ, pfold;
              colormap = cgrad([RGBAf(HUE[3]..., 1), RGBAf(GREY..., 1), RGBAf(HUE[2]..., 1)]),
              colorrange = (0, 1))
if !isempty(slith)
    scatter!(ax2, [log10(c[2]) for c in slith], [log10(c[1]) for c in slith];
             marker = :circle, markersize = 7, color = (:white, 0.0),
             strokecolor = RGBAf(HUE[1]..., 1), strokewidth = 1.4)
end
limits!(ax2, xr..., yr...)
Colorbar(fig[1, 3], hm, label = "P(fold)   0 = every seed glides · 1 = every seed folds",
         width = 14)
Label(fig[2, 2], "○ = ≥$(round(Int, 100SLITHER_MARK))% of seeds slithering",
      fontsize = 11, tellwidth = false)

out = joinpath(PROJECT_ROOT_PDC, "results", folder, "phase_diagram_confidence.png")
save(out, fig; px_per_unit = 3)
CairoMakie.activate!(type = "pdf"); save(replace(out, ".png" => ".pdf"), fig); CairoMakie.activate!(type = "png")
println("✓ figure → $(relpath(out, PROJECT_ROOT_PDC))")
