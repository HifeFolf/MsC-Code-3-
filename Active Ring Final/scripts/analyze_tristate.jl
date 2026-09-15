# analyze_tristate.jl
# The "folded" state is two states. This maps all three.
#
# Splitting folded dwells by the sign of Kconn gave two populations with the same
# Rg (~1.35-1.40) but ~12x different lifetimes (4.1 s vs 50.4 s). Rg alone cannot
# tell them apart, so every Rg-only analysis (regime map, dwell kinetics, the
# double wells) merged them. Here states are defined on (Rg, Kconn):
#
#   OPEN      Rg > 1.1*thr
#   FOLD-OUT  Rg < 0.9*thr and Kconn > +K0     (curled outward)
#   FOLD-IN   Rg < 0.9*thr and Kconn < -K0     (crumpled inward)
#   anything else is a deadband: no opinion, the previous state holds
#
# A candidate state must hold MIN_DWELL_S before a transition is registered.
# Rates are count(a->b) / total time spent in a.
#
# On panel A: in 2D the drift of a driven system generally has a rotational
# (non-gradient) part, so NO potential exists. -ln P(Rg, Kconn) shows where the
# basins are; it is not a landscape to read barrier heights from.
#
# Usage:  julia --project scripts/analyze_tristate.jl [folder]

using CairoMakie, Printf, Statistics, TOML, JLD2
const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_bistable_long"
const BURN_S = 5.0
const MIN_DWELL_S = 0.5
const K0 = 30.0
const ST = [:O, :Fo, :Fi]
const SHORT = Dict(:O=>"open", :Fo=>"fold-out", :Fi=>"fold-in")

function scan(dir)
    RG=Float64[]; KC=Float64[]; thr=NaN
    dw=Dict(s=>Float64[] for s in ST); occ=Dict(s=>0.0 for s in ST)
    T=Dict{Tuple{Symbol,Symbol},Int}(); nrun=0
    for rn in sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
        par=joinpath(dir,rn,"params.toml"); obs=joinpath(dir,rn,"observables.jld2")
        (isfile(par)&&isfile(obs)) || continue
        N=Int(TOML.parsefile(par)["params"]["N"]); thr=0.70*(N/(2π))
        d=jldopen(f->(t=f["times"],Rg=f["Rg"],K=rad2deg.(f["Kconn"])), obs, "r")
        i0=something(findfirst(>=(BURN_S),d.t),1)
        t=d.t[i0:end]; Rg=d.Rg[i0:end]; K=d.K[i0:end]
        length(t) < 20 && continue
        dt=(t[end]-t[1])/(length(t)-1); need=max(2,round(Int,MIN_DWELL_S/dt))
        raw(j) = Rg[j] > 1.1thr ? :O :
                 (Rg[j] < 0.9thr ? (K[j] > K0 ? :Fo : (K[j] < -K0 ? :Fi : :none)) : :none)
        j1 = findfirst(j->raw(j)!==:none, eachindex(Rg))
        j1 === nothing && continue
        st=raw(j1); tin=t[j1]; rl=0; pend=:none
        segs=Tuple{Symbol,Float64,Float64}[]
        for j in j1:length(Rg)
            s=raw(j)
            if s===:none || s===st
                rl=0; pend=:none; continue
            end
            if s===pend; rl+=1; else; pend=s; rl=1; end
            if rl >= need
                push!(segs,(st,tin,t[j])); T[(st,s)]=get(T,(st,s),0)+1
                st=s; tin=t[j]; rl=0; pend=:none
            end
        end
        push!(segs,(st,tin,t[end]))
        for (k,(s,a,b)) in enumerate(segs)
            occ[s]+=b-a
            (k==1 || k==length(segs)) && continue
            push!(dw[s], b-a)
        end
        append!(RG,Rg); append!(KC,K); nrun+=1
    end
    (RG=RG,KC=KC,dw=dw,occ=occ,T=T,nrun=nrun,thr=thr)
end

r = scan(joinpath(ROOT,"results",folder))
@printf("%s: %d runs, %d frames\n\n", folder, r.nrun, length(r.RG))
tot = sum(values(r.occ))
println("Per-state dwell statistics (complete dwells only):")
@printf("  %-10s %6s %10s %7s %11s\n","state","n","mean (s)","CV","occupancy")
for s in ST
    v=r.dw[s]
    length(v) >= 3 ?
      @printf("  %-10s %6d %10.2f %7.2f %10.1f%%\n", SHORT[s], length(v), mean(v), std(v)/mean(v), 100r.occ[s]/tot) :
      @printf("  %-10s %6d %10s %7s %10.1f%%\n", SHORT[s], length(v), "-", "-", 100r.occ[s]/tot)
end
println("\nTransition network   count  (rate = count / time in FROM state)")
@printf("  %-10s", "from\to"); for b in ST; @printf("%24s", SHORT[b]); end; println()
for a in ST
    @printf("  %-10s", SHORT[a])
    for b in ST
        a==b && (@printf("%24s","-"); continue)
        c=get(r.T,(a,b),0)
        @printf("%24s", @sprintf("%d  (%.4f /s)", c, r.occ[a]>0 ? c/r.occ[a] : NaN))
    end
    println()
end
open(joinpath(ROOT,"results",folder,"tristate_transitions.csv"),"w") do io
    println(io,"from,to,count,rate_per_s,time_in_from_s")
    for a in ST, b in ST
        a==b && continue
        c=get(r.T,(a,b),0)
        @printf(io,"%s,%s,%d,%.6f,%.2f\n",SHORT[a],SHORT[b],c,r.occ[a]>0 ? c/r.occ[a] : NaN,r.occ[a])
    end
end

fig = Figure(size=(1820,600))
rlo,rhi = quantile(r.RG,0.002), quantile(r.RG,0.998)
nx,ny = 60,72
ex=range(rlo,rhi,length=nx+1); ey=range(-180,180,length=ny+1)
H=zeros(nx,ny)
for (x,y) in zip(r.RG,r.KC)
    (rlo<=x<=rhi) || continue
    H[clamp(Int(fld(x-rlo,step(ex)))+1,1,nx), clamp(Int(fld(y+180,step(ey)))+1,1,ny)] += 1
end
P=H./sum(H); F=fill(NaN,nx,ny)
for i in 1:nx, j in 1:ny; P[i,j]>0 && (F[i,j]=-log(P[i,j])); end
F .-= minimum(filter(isfinite,F))
cx=(ex[1:end-1].+ex[2:end])./2; cy=(ey[1:end-1].+ey[2:end])./2
ax1=Axis(fig[1,1], xlabel="Radius of gyration  Rg", ylabel="Kconn  [deg]",
         title="A   Three basins in (Rg, Kconn)")
hm=heatmap!(ax1,cx,cy,F; colormap=:viridis, colorrange=(0,9), nan_color=:white)
Colorbar(fig[1,2],hm,label="−ln P   (basins only — not a potential)", width=12)
masks = Dict(:O  => r.RG .> 1.1r.thr,
             :Fo => (r.RG .< 0.9r.thr) .& (r.KC .>  K0),
             :Fi => (r.RG .< 0.9r.thr) .& (r.KC .< -K0))
for s in ST
    m=masks[s]; any(m) || continue
    text!(ax1, median(r.RG[m]), median(r.KC[m]); text=SHORT[s], color=:white,
          font=:bold, fontsize=14, align=(:center,:center))
end

ax2=Axis(fig[1,3], title="B   Transition network", xlabel="to", ylabel="from",
         xticks=(1:3,[SHORT[s] for s in ST]), yticks=(1:3,[SHORT[s] for s in ST]),
         yreversed=true)
M=[a==b ? NaN : Float64(get(r.T,(a,b),0)) for a in ST, b in ST]
heatmap!(ax2, 1:3, 1:3, permutedims(M); colormap=:Blues, nan_color=:gray92)
for (i,a) in enumerate(ST), (j,b) in enumerate(ST)
    a==b && continue
    c=get(r.T,(a,b),0)
    text!(ax2, j, i; text=@sprintf("%d\n%.3f /s", c, r.occ[a]>0 ? c/r.occ[a] : 0.0),
          align=(:center,:center), fontsize=13)
end

ax3=Axis(fig[1,4], xlabel="dwell time  (s)", ylabel="survival  P(T > t)", yscale=log10,
         title="C   Dwell survival, per state")
cols=Dict(:O=>:steelblue, :Fo=>:darkorange, :Fi=>:crimson)
for s in ST
    v=sort(r.dw[s]); length(v) < 5 && continue
    lines!(ax3, v, 1 .- (0:length(v)-1)./length(v); color=cols[s], linewidth=2.5,
           label=@sprintf("%s  n=%d  CV=%.2f", SHORT[s], length(v), std(v)/mean(v)))
end
axislegend(ax3; position=:rt, framevisible=false, labelsize=11)

base=joinpath(ROOT,"results",folder,"tristate")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("\n✓ ", relpath(base,ROOT), ".png / .pdf  ·  tristate_transitions.csv")
