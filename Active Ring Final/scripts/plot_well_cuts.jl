# plot_well_cuts.jl
# Effective double wells along the six cuts, numbered to match cut_map.png.
#
# Why cuts rather than a Phi series:  Panel B of bistability_evidence.png plots the
# well against Phi, which conflates f0 and kappa.  Since Phi50 ~ kappa^0.41 * kBT^-0.60,
# Phi is NOT the control parameter, so two cells at equal Phi are not equivalent.
# Holding one axis fixed and sweeping the other separates the two dependencies.
#
#   Panels 1-3  fixed f0, kappa swept   -> expect tilt folded (low kappa) -> open (high kappa)
#   Panels 4-6  fixed kappa, f0 swept   -> expect tilt open (low f0) -> folded (high f0)
#
# Per cell, Rg is pooled over all 10 seeds (post burn-in), histogrammed, and plotted
# as -ln P.  Each curve is shifted so ITS OWN minimum sits at zero, so the comparison
# is relative well depth, not absolute probability.
#
# Cells far from the transition show a SINGLE well — the ring never visits the other
# state.  That is the tilt going to completion, not a failure of the measurement.
#
# Usage:  julia --project scripts/plot_well_cuts.jl [folder]

using CairoMakie, Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_wellcuts"
const BURN_S = 5.0
const RG_LO, RG_HI, NBIN = 1.15, 2.95, 64

dir = joinpath(ROOT,"results",folder)
runs = sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
@printf("pooling Rg from %d runs…\n", length(runs))

pool = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
for rn in runs
    par=joinpath(dir,rn,"params.toml"); obs=joinpath(dir,rn,"observables.jld2")
    (isfile(par)&&isfile(obs)) || continue
    ep=TOML.parsefile(par)["params"]
    k=(round(Float64(ep["kappa"]),digits=4), round(Float64(ep["f0"]),digits=4))
    d=jldopen(obs,"r") do f
        haskey(f,"Rg") ? (t=f["times"], Rg=f["Rg"]) : nothing
    end
    d === nothing && continue
    i0=something(findfirst(>=(BURN_S), d.t),1)
    append!(get!(pool,k,Float64[]), @view d.Rg[i0:end])
end
@printf("  %d cells pooled\n", length(pool))

# potential from pooled Rg, shifted to its own minimum
edges = range(RG_LO, RG_HI, length=NBIN+1); ctr = (edges[1:end-1].+edges[2:end])./2
w = step(edges)
function well(v)
    h = zeros(Int, NBIN)
    for x in v
        (RG_LO <= x <= RG_HI) || continue
        h[clamp(Int(fld(x-RG_LO,w))+1,1,NBIN)] += 1
    end
    s = sum(h); s == 0 && return (Float64[], Float64[])
    p = h ./ s; k = p .> 0
    y = -log.(p[k]); (ctr[k], y .- minimum(y))
end

# detect cuts from the cell list, exactly as plot_cut_map.jl does
cut = [split(l,",") for l in readlines(joinpath(ROOT,"results","campaign","cells_wellcuts.csv"))[2:end] if !isempty(strip(l))]
cκ=[round(parse(Float64,r[1]),digits=4) for r in cut]; cf=[round(parse(Float64,r[2]),digits=4) for r in cut]
fixF = sort(unique(f for f in cf if count(==(f),cf)==9))
fixK = sort(unique(k for k in cκ if count(==(k),cκ)==9))

fig = Figure(size=(1680,980))
Label(fig[0,1:3], "Effective double well along each cut — numbers match cut_map.png";
      fontsize=17, font=:bold, tellwidth=false)

function panel(pos, num, cells, swept, fixedlabel, sweptlabel, cmapname)
    ax = Axis(fig[pos...], xlabel="Radius of gyration  Rg", ylabel="−ln P(Rg)  [arb.]",
              title=@sprintf("%d   %s", num, fixedlabel))
    cg = cgrad(cmapname, max(2,length(swept)); categorical=true)
    for (i,s) in enumerate(swept)
        key = cells[i]
        haskey(pool,key) || continue
        x,y = well(pool[key]); isempty(x) && continue
        lines!(ax, x, y; color=cg[i], linewidth=2.0)
    end
    ylims!(ax, -0.3, 9.0); xlims!(ax, RG_LO, RG_HI)
    ax
end

for (i,f) in enumerate(fixF)
    ks = sort(unique(cκ[cf .== f]))
    panel((1,i), i, [(k,f) for k in ks], ks,
          @sprintf("f₀ = %.2f fixed · κ swept %.2f → %.1f", f, minimum(ks), maximum(ks)),
          "κ", :viridis)
end
for (j,k) in enumerate(fixK)
    f0s = sort(unique(cf[cκ .== k]))
    panel((2,j), j+length(fixF), [(k,f) for f in f0s], f0s,
          @sprintf("κ = %.3f fixed · f₀ swept %.1f → %.1f", k, minimum(f0s), maximum(f0s)),
          "f₀", :plasma)
end

Colorbar(fig[1,4]; colormap=:viridis, limits=(minimum(cκ),maximum(cκ)), scale=log10,
         label="κ  (panels 1–3)", width=13)
Colorbar(fig[2,4]; colormap=:plasma, limits=(minimum(cf),maximum(cf)), scale=log10,
         label="f₀  (panels 4–6)", width=13)
Label(fig[3,1:4],
  "each curve shifted to its own minimum · Rg ≈ 1.35 = folded, ≈ 2.71 = open · " *
  "a single well means the ring never visited the other state";
  fontsize=11, tellwidth=false)

base = joinpath(ROOT,"results","campaign","well_cuts")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("✓ ", relpath(base,ROOT), ".png / .pdf")
