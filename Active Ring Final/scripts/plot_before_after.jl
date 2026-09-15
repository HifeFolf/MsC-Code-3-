# plot_before_after.jl
# The single most persuasive slide: the same physics, mapped the way it was at
# the start of the project and the way it is now.
#
#   BEFORE  Results_N17_circle_indep — 1 seed per cell, 7.5 s, categorical label
#           from the `route` classifier.  Every cell forced into one of three
#           classes; a coin-flip cell looks identical to a decided one.
#   AFTER   Results_N17_phi_long — 30 seeds per cell, 30 s, regime from transition
#           DIRECTION, with saturation carrying seed agreement.
#
# Both panels are drawn here from cached CSVs on identical axes, so the
# comparison is like-for-like rather than two differently-sized old PNGs.
#
# Usage:  julia --project scripts/plot_before_after.jl

using CairoMakie, Printf, Statistics

const ROOT = joinpath(@__DIR__, "..")
const BI_FRAC=0.15; const INWARD_DEV=-8.0; const RG_FRAC=0.70; const B_SLITHER=0.04
const BI_MIN = 0.20
rd(p)=[split(l,",") for l in readlines(p)[2:end] if !isempty(strip(l))]
num(v,i)=(x=tryparse(Float64,v[i]); x===nothing ? NaN : x)

# ── BEFORE: one seed, 7.5 s, route classifier ───────────────────────────────
f1 = joinpath(ROOT,"results","Results_N17_circle_indep","phase_features.csv")
hdr1 = String.(split(readlines(f1)[1], ",")); ix1(n)=findfirst(==(n),hdr1)
before = map(rd(f1)) do r
    lab = min(num(r,ix1("f_in")), num(r,ix1("f_out"))) > BI_FRAC ? 1 :
          ((num(r,ix1("dev")) < INWARD_DEV || num(r,ix1("Rg_med")) < RG_FRAC*(num(r,ix1("N"))/(2π))) ? 2 :
           ((isnan(num(r,ix1("bstd"))) ? 0.0 : num(r,ix1("bstd"))) > B_SLITHER ? 1 : 3))
    (κ=num(r,ix1("kappa")), f0=num(r,ix1("f0")), lab=lab)
end

# ── AFTER: 30 seeds, 30 s, regime map ───────────────────────────────────────
after = map(rd(joinpath(ROOT,"results","Results_N17_phi_long","regime_map.csv"))) do r
    fq=parse(Float64,r[6]); fb=parse(Float64,r[7]); fa=parse(Float64,r[8])
    reg = fb>=BI_MIN ? 2 : (fa>fq ? 3 : 1)
    (κ=parse(Float64,r[1]), f0=parse(Float64,r[2]), reg=reg,
     conf = reg==2 ? clamp(fb/0.5,0,1) : max(fq,fa))
end

κs = sort(unique(r.κ for r in before)); fs = sort(unique(r.f0 for r in before))
ki=Dict(v=>i for (i,v) in enumerate(κs)); fi=Dict(v=>i for (i,v) in enumerate(fs))
logκ, logf = log10.(κs), log10.(fs)
tk(v)=(range(minimum(v),maximum(v),length=5),[@sprintf("%.3g",10^x) for x in range(minimum(v),maximum(v),length=5)])
xr=(minimum(logf),maximum(logf)); yr=(minimum(logκ),maximum(logκ))

OLD = Dict(1=>(0.80,0.13,0.44), 2=>(0.95,0.55,0.10), 3=>(0.27,0.47,0.72))
HUE = [(0.27,0.47,0.72),(0.80,0.13,0.44),(0.95,0.55,0.10)]; GREY=(0.85,0.85,0.85)
IMGb = fill(RGBAf(0,0,0,0), length(fs), length(κs))
IMGa = fill(RGBAf(0,0,0,0), length(fs), length(κs))
for r in before
    (haskey(fi,r.f0) && haskey(ki,r.κ)) || continue
    IMGb[fi[r.f0], ki[r.κ]] = RGBAf(OLD[r.lab]..., 1)
end
for r in after
    (haskey(fi,r.f0) && haskey(ki,r.κ)) || continue
    c=clamp((r.conf-0.4)/0.6,0,1); h=HUE[r.reg]
    IMGa[fi[r.f0], ki[r.κ]] = RGBAf(c*h[1]+(1-c)*GREY[1], c*h[2]+(1-c)*GREY[2], c*h[3]+(1-c)*GREY[3],1)
end

fig = Figure(size=(1560,720))
for (col,(IMG,ttl,sub)) in enumerate([
      (IMGb, "BEFORE   1 seed · 7.5 s · forced label",
             "every cell gets a class — a coin flip looks like a certainty"),
      (IMGa, "AFTER   30 seeds · 30 s · regime + confidence",
             "class from transition DIRECTION; pale = seeds disagree")])
    ax = Axis(fig[1,col], xlabel="Active force  f₀", ylabel = col==1 ? "Bending stiffness  κ" : "",
              title=ttl, subtitle=sub, xticks=tk(logf), yticks=tk(logκ))
    image!(ax, xr, yr, IMG; interpolate=false); limits!(ax, xr..., yr...)
end
Legend(fig[2,1:2],
  [PolyElement(color=RGBAf(HUE[c]...,1)) for c in 1:3],
  ["Quiescent / glider (never folds)","Bistable (switches both ways)","Absorbing / crescent (folds, stays)"],
  orientation=:horizontal, framevisible=false, labelsize=12)
base = joinpath(ROOT,"results","campaign","before_after")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("✓ ", relpath(base,ROOT), ".png / .pdf")
