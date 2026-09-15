# plot_cut_map.jl
# The regime phase diagram with the well-cut sweeps drawn on it and numbered, so
# each -ln P(Rg) panel can be placed against the position it was measured at.
#
#   LEFT   full diagram, with the zoom box marked
#   RIGHT  zoomed to the cuts, each line numbered 1-6
#
# Cuts are DETECTED from cells_wellcuts.csv rather than restated here: a fixed
# value appears with all 9 swept values (9 cells), a swept value appears only at
# the 3 fixed ones. So the figure cannot drift out of step with what was run.
#
# Note the vertical cuts extend to kappa = 0.5, BELOW the mapped grid (kappa >= 1).
# Those points are real runs but sit outside the surveyed region, so the background
# is simply absent there — drawn, not hidden.
#
# Usage:  julia --project scripts/plot_cut_map.jl

using CairoMakie, Printf, Statistics

const ROOT = joinpath(@__DIR__, "..")
const BI_MIN = 0.20
rd(p) = [split(l,",") for l in readlines(p)[2:end] if !isempty(strip(l))]

# ── background: regime map ───────────────────────────────────────────────────
rows = map(rd(joinpath(ROOT,"results","Results_N17_phi_long","regime_map.csv"))) do r
    fq=parse(Float64,r[6]); fb=parse(Float64,r[7]); fa=parse(Float64,r[8])
    reg = fb>=BI_MIN ? 2 : (fa>fq ? 3 : 1)
    (κ=parse(Float64,r[1]), f0=parse(Float64,r[2]), reg=reg,
     conf = reg==2 ? clamp(fb/0.5,0,1) : max(fq,fa))
end
κs = sort(unique(r.κ for r in rows)); fs = sort(unique(r.f0 for r in rows))
ki=Dict(v=>i for (i,v) in enumerate(κs)); fi=Dict(v=>i for (i,v) in enumerate(fs))
HUE=[(0.27,0.47,0.72),(0.80,0.13,0.44),(0.95,0.55,0.10)]; GREY=(0.85,0.85,0.85)
IMG = fill(RGBAf(0,0,0,0), length(fs), length(κs))
for r in rows
    c=clamp((r.conf-0.4)/0.6,0,1); h=HUE[r.reg]
    IMG[fi[r.f0],ki[r.κ]] = RGBAf(c*h[1]+(1-c)*GREY[1], c*h[2]+(1-c)*GREY[2], c*h[3]+(1-c)*GREY[3],1)
end
xr=(log10(minimum(fs)), log10(maximum(fs))); yr=(log10(minimum(κs)), log10(maximum(κs)))

# ── detect the cuts from the cell list ───────────────────────────────────────
cut = rd(joinpath(ROOT,"results","campaign","cells_wellcuts.csv"))
cκ = [parse(Float64,r[1]) for r in cut]; cf = [parse(Float64,r[2]) for r in cut]
nper(v, all) = count(==(v), all)
fixF = sort(unique(f for f in cf if nper(f,cf) == 9))      # vertical cuts
fixK = sort(unique(k for k in cκ if nper(k,cκ) == 9))      # horizontal cuts
κspan = (minimum(cκ), maximum(cκ)); fspan = (minimum(cf), maximum(cf))
@printf("detected %d vertical cuts (fixed f0 = %s)\n", length(fixF), join(fixF,", "))
@printf("detected %d horizontal cuts (fixed kappa = %s)\n", length(fixK), join(fixK,", "))

const CA = RGBAf(0.05,0.35,0.22,1)    # vertical  (fixed f0)
const CB = RGBAf(0.35,0.10,0.45,1)    # horizontal (fixed kappa)

# numbering: 1..3 vertical, 4..6 horizontal
labels = String[]
for (i,f) in enumerate(fixF); push!(labels, @sprintf("%d   f₀ = %.2f  fixed,  κ swept %.2f → %.1f", i, f, κspan...)); end
for (j,k) in enumerate(fixK); push!(labels, @sprintf("%d   κ = %.3f fixed,  f₀ swept %.2f → %.1f", j+length(fixF), k, fspan...)); end

function draw_cuts!(ax; marks=true, nums=true)
    for (i,f) in enumerate(fixF)
        x = log10(f)
        lines!(ax, [x,x], [log10(κspan[1]), log10(κspan[2])]; color=CA, linewidth=2.6)
        if marks
            ks = sort(unique(cκ[cf .== f]))
            scatter!(ax, fill(x,length(ks)), log10.(ks); color=CA, markersize=7)
        end
        if nums
            scatter!(ax,[x],[log10(κspan[2])+0.055]; marker=:circle, markersize=19,
                     color=(:white,0.95), strokecolor=CA, strokewidth=1.6)
            text!(ax,x,log10(κspan[2])+0.055; text=string(i), align=(:center,:center),
                  fontsize=11, font=:bold, color=CA)
        end
    end
    for (j,k) in enumerate(fixK)
        y = log10(k)
        lines!(ax, [log10(fspan[1]), log10(fspan[2])], [y,y]; color=CB, linewidth=2.6, linestyle=:dash)
        if marks
            f0s = sort(unique(cf[cκ .== k]))
            scatter!(ax, log10.(f0s), fill(y,length(f0s)); color=CB, markersize=7)
        end
        if nums
            scatter!(ax,[log10(fspan[2])+0.055],[y]; marker=:circle, markersize=19,
                     color=(:white,0.95), strokecolor=CB, strokewidth=1.6)
            text!(ax,log10(fspan[2])+0.055,y; text=string(j+length(fixF)),
                  align=(:center,:center), fontsize=11, font=:bold, color=CB)
        end
    end
end

tk(lo,hi) = (range(lo,hi,length=5), [@sprintf("%.3g",10^x) for x in range(lo,hi,length=5)])
fig = Figure(size=(1560,700))

# LEFT: full diagram + zoom box
ax1 = Axis(fig[1,1], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="Full diagram — cuts occupy the lower-left corner",
           xticks=tk(xr...), yticks=tk(yr...))
image!(ax1, xr, yr, IMG; interpolate=false)
zx = (log10(fspan[1])-0.06, log10(fspan[2])+0.12); zy = (log10(κspan[1])-0.06, log10(κspan[2])+0.14)
lines!(ax1, [zx[1],zx[2],zx[2],zx[1],zx[1]], [zy[1],zy[1],zy[2],zy[2],zy[1]];
       color=:black, linewidth=1.8)
draw_cuts!(ax1; marks=false, nums=false)
limits!(ax1, xr[1]-0.03, xr[2]+0.03, min(yr[1],zy[1])-0.03, yr[2]+0.03)

# RIGHT: zoom
ax2 = Axis(fig[1,2], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="Where each cut sits  (points = sampled cells)",
           xticks=tk(zx...), yticks=tk(zy...))
image!(ax2, xr, yr, IMG; interpolate=false)
hlines!(ax2, [log10(minimum(κs))]; color=(:black,0.35), linewidth=1, linestyle=:dot)
text!(ax2, zx[1]+0.04, log10(minimum(κs))-0.035; text="below: no phase-diagram data (grid starts at κ = 1)",
      fontsize=9, color=(:black,0.55), align=(:left,:top))
draw_cuts!(ax2)
limits!(ax2, zx..., zy...)

Legend(fig[2,1:2],
  [LineElement(color=CA,linewidth=2.6), LineElement(color=CB,linewidth=2.6,linestyle=:dash),
   PolyElement(color=RGBAf(HUE[1]...,1)), PolyElement(color=RGBAf(HUE[2]...,1)),
   PolyElement(color=RGBAf(HUE[3]...,1))],
  ["Cuts 1-3: fixed f₀, κ swept", "Cuts 4-6: fixed κ, f₀ swept",
   "Quiescent", "Bistable", "Absorbing"],
  orientation=:horizontal, framevisible=false, labelsize=11)
Label(fig[3,1:2], join(labels, "      "); fontsize=11, tellwidth=false)

base = joinpath(ROOT,"results","campaign","cut_map")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("✓ ", relpath(base,ROOT), ".png / .pdf")
