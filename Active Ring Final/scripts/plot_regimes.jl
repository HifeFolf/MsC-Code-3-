# plot_regimes.jl
# Renders the regime phase diagram from a cached regime_map.csv — no trajectory
# rescan (that costs ~10 min for 10k runs and must not be repeated per plot tweak).
#
# Panel C deliberately shows f_bistable, NOT the raw transition rate.  The raw
# rate counts the INITIAL folding event, which every absorbing run has, so it
# lights up the whole folded region and reads as switching where there is none:
# the high-f0 band shows rate 0.94/100 s but f_bi = 0.006 with occupancy 0.81 —
# rings that folded once and merely jitter across the cut.  Requiring transitions
# in BOTH directions removes that artifact.
#
# Usage:  julia --project scripts/plot_regimes.jl [folder]

using CairoMakie, Printf, Statistics
CairoMakie.activate!(type = "png")

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_phi_long"
csv = joinpath(ROOT, "results", folder, "regime_map.csv")
isfile(csv) || error("Missing $csv — run scripts/phase_diagram_regimes.jl first")

# Regime is RECLASSIFIED here rather than taken from the scan's majority vote.
# Majority requires f_bi >= 0.5, but the true maximum anywhere is 0.53 — so the
# rule counted 13 cells at 10 seeds and only 11 at 30, because sharper estimates
# REMOVED the chance crossings that had pushed cells over 0.5.  It gets stricter
# as precision improves, which is backwards.  A cell where >= BI_MIN of seeds are
# seen switching BOTH ways inside a window shorter than the dwell time is strong
# evidence, so that is the criterion.
const BI_MIN = 0.20
rows = let raw = [split(l,",") for l in readlines(csv)[2:end] if !isempty(strip(l))]
    map(raw) do r
        f_qui = parse(Float64, r[6]); f_bi = parse(Float64, r[7]); f_abs = parse(Float64, r[8])
        reg  = f_bi >= BI_MIN ? 2 : (f_abs > f_qui ? 3 : 1)
        conf = reg == 2 ? clamp(f_bi/0.5, 0, 1) : max(f_qui, f_abs)
        (κ=parse(Float64,r[1]), f0=parse(Float64,r[2]), occ=parse(Float64,r[5]),
         f_bi=f_bi, reg=reg, conf=conf)
    end
end

κs = sort(unique(r.κ for r in rows)); fs = sort(unique(r.f0 for r in rows))
ki = Dict(v=>i for (i,v) in enumerate(κs)); fi = Dict(v=>i for (i,v) in enumerate(fs))
logκ, logf = log10.(κs), log10.(fs)
tk(v) = (range(minimum(v),maximum(v),length=5),
         [@sprintf("%.3g",10^x) for x in range(minimum(v),maximum(v),length=5)])
xr = (minimum(logf), maximum(logf)); yr = (minimum(logκ), maximum(logκ))

HUE = [(0.27,0.47,0.72), (0.80,0.13,0.44), (0.95,0.55,0.10)]
GREY = (0.85,0.85,0.85)
RN = ["Quiescent (never folds)", "Bistable — ≥20% of seeds switch both ways", "Absorbing (folds, stays)"]

OCC = fill(NaN, length(fs), length(κs)); BI = fill(NaN, length(fs), length(κs))
IMG = fill(RGBAf(0,0,0,0), length(fs), length(κs))
for r in rows
    j, i = fi[r.f0], ki[r.κ]
    OCC[j,i] = r.occ; BI[j,i] = r.f_bi
    c = clamp((r.conf-0.4)/0.6, 0.0, 1.0); h = HUE[r.reg]
    IMG[j,i] = RGBAf(c*h[1]+(1-c)*GREY[1], c*h[2]+(1-c)*GREY[2], c*h[3]+(1-c)*GREY[3], 1)
end

fig = Figure(size = (1800, 580))
ax1 = Axis(fig[1,1], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="A   Occupancy — fraction of TIME folded", xticks=tk(logf), yticks=tk(logκ))
hm = heatmap!(ax1, logf, logκ, OCC;
              colormap=cgrad([RGBAf(HUE[1]...,1), RGBAf(GREY...,1), RGBAf(HUE[3]...,1)]),
              colorrange=(0,1))
Colorbar(fig[1,2], hm, label="fraction of time folded", width=13)

ax2 = Axis(fig[1,3], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="B   Regime (saturation = seed agreement)", xticks=tk(logf), yticks=tk(logκ))
image!(ax2, xr, yr, IMG; interpolate=false); limits!(ax2, xr..., yr...)
Legend(fig[2,3], [PolyElement(color=RGBAf(HUE[c]...,1)) for c in 1:3], RN,
       orientation=:horizontal, framevisible=false, labelsize=10)

ax3 = Axis(fig[1,4], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="C   Two-way switching (seeds folding AND unfolding)",
           xticks=tk(logf), yticks=tk(logκ))
hm3 = heatmap!(ax3, logf, logκ, BI; colormap=[:gray92,:gold,:crimson], colorrange=(0,0.6))
Colorbar(fig[1,5], hm3, label="fraction of seeds switching both ways", width=13)
Label(fig[2,1],
  "28 s window · bistable region is a LOWER BOUND (dwell times 15-75 s exceed it) · " *
  "panel C excludes the initial fold, which the raw transition rate would count",
  fontsize=10, tellwidth=false)

# ── Export ───────────────────────────────────────────────────────────────────
# Default Makie output is 1 px per unit — 1800x580 here, far below the ~300 DPI a
# journal wants.  px_per_unit=3 gives 5400x1740 for raster use, and a PDF carries
# the same figure as vector art so it scales without resampling.
base = joinpath(ROOT, "results", folder, "phase_diagram_regimes")
save(base * ".png", fig; px_per_unit = 3)
CairoMakie.activate!(type = "pdf")
save(base * ".pdf", fig)
CairoMakie.activate!(type = "png")
@printf("✓ %s.png  (%.0fx%.0f px)\n", relpath(base, ROOT), 1800*3, 580*3)
@printf("✓ %s.pdf  (vector)\n", relpath(base, ROOT))
