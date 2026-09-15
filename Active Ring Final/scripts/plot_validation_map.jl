# plot_validation_map.jl
# The regime phase diagram with the rendered validation runs marked and numbered,
# and a legend keying each number to its movie file.
#
# Purpose: make the movies checkable against the map.  A reader (or a referee)
# can point at a numbered cell, open the corresponding file, and see whether the
# behaviour matches the label — which is the only way a phase diagram earns trust.
#
# Reads only cached CSVs; no trajectory scan.
#
# Usage:  julia --project scripts/plot_validation_map.jl [folder]

using CairoMakie, Printf, Statistics

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_phi_long"
const BI_MIN = 0.20

# Makie has no :monospace symbol; resolve a real family, fall back to regular.
const MONO = try (Makie.to_font("DejaVu Sans Mono"); "DejaVu Sans Mono") catch; :regular end

rd(p) = [split(l, ",") for l in readlines(p) if !isempty(strip(l))]
reg_rows = rd(joinpath(ROOT, "results", folder, "regime_map.csv"))[2:end]
rows = map(reg_rows) do r
    f_qui = parse(Float64, r[6]); f_bi = parse(Float64, r[7]); f_abs = parse(Float64, r[8])
    reg  = f_bi >= BI_MIN ? 2 : (f_abs > f_qui ? 3 : 1)
    conf = reg == 2 ? clamp(f_bi/0.5, 0, 1) : max(f_qui, f_abs)
    (κ = parse(Float64, r[1]), f0 = parse(Float64, r[2]), reg = reg, conf = conf)
end
marks = rd(joinpath(ROOT, "results", "campaign", "validate_list.txt"))

κs = sort(unique(r.κ for r in rows)); fs = sort(unique(r.f0 for r in rows))
ki = Dict(v=>i for (i,v) in enumerate(κs)); fi = Dict(v=>i for (i,v) in enumerate(fs))
logκ, logf = log10.(κs), log10.(fs)
tk(v) = (range(minimum(v),maximum(v),length=5),
         [@sprintf("%.3g",10^x) for x in range(minimum(v),maximum(v),length=5)])
xr = (minimum(logf), maximum(logf)); yr = (minimum(logκ), maximum(logκ))

HUE = [(0.27,0.47,0.72), (0.80,0.13,0.44), (0.95,0.55,0.10)]
GREY = (0.85,0.85,0.85)
IMG = fill(RGBAf(0,0,0,0), length(fs), length(κs))
for r in rows
    c = clamp((r.conf-0.4)/0.6, 0.0, 1.0); h = HUE[r.reg]
    IMG[fi[r.f0], ki[r.κ]] = RGBAf(c*h[1]+(1-c)*GREY[1], c*h[2]+(1-c)*GREY[2], c*h[3]+(1-c)*GREY[3], 1)
end

fig = Figure(size = (1560, 760))
ax = Axis(fig[1,1], xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
          title = "Validation runs on the regime map", xticks = tk(logf), yticks = tk(logκ))
image!(ax, xr, yr, IMG; interpolate = false)
# Pad by ~0.7 of a cell: markers on cells at the axis boundary (kappa=1, f0=120)
# are otherwise drawn half outside the frame and clipped.
dx = (xr[2]-xr[1])/max(1,length(fs)-1); dy = (yr[2]-yr[1])/max(1,length(κs)-1)
limits!(ax, xr[1]-0.7dx, xr[2]+0.7dx, yr[1]-0.7dy, yr[2]+0.7dy)

# numbered markers
xs = [log10(parse(Float64, m[4])) for m in marks]
ys = [log10(parse(Float64, m[3])) for m in marks]
scatter!(ax, xs, ys; marker = :circle, markersize = 22,
         color = (:white, 0.92), strokecolor = :black, strokewidth = 1.6)
for (i, (x, y)) in enumerate(zip(xs, ys))
    text!(ax, x, y; text = string(i), align = (:center, :center),
          fontsize = 12, font = :bold, color = :black)
end

Legend(fig[2,1],
       [PolyElement(color = RGBAf(HUE[c]...,1)) for c in 1:3],
       ["Quiescent (never folds)", "Bistable (≥20% of seeds switch)", "Absorbing (folds, stays)"],
       orientation = :horizontal, framevisible = false, labelsize = 11)

# key: number → parameters → movie file
lines = String[]
push!(lines, rpad("#", 3) * rpad("category", 9) * rpad("κ", 9) * rpad("f₀", 10) *
             rpad("Φ", 8) * "movie file")
push!(lines, "─"^92)
for (i, m) in enumerate(marks)
    cat, run, k, f, p = m[1], m[2], m[3], m[4], m[5]
    push!(lines, rpad(string(i), 3) * rpad(cat, 9) * rpad(k, 9) * rpad(f, 10) * rpad(p, 8) *
                 "$(cat)_kappa$(k)_f0$(f)_phi$(p).mp4")
end
Label(fig[1:2, 2], join(lines, "\n"); font = MONO, fontsize = 11,
      halign = :left, justification = :left, valign = :center, tellwidth = false)
colsize!(fig.layout, 1, Relative(0.46))

base = joinpath(ROOT, "results", "campaign", "validation_map")
save(base * ".png", fig; px_per_unit = 3)
CairoMakie.activate!(type = "pdf"); save(base * ".pdf", fig); CairoMakie.activate!(type = "png")
println("✓ ", relpath(base, ROOT), ".png / .pdf")
