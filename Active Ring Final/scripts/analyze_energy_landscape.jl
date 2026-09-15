# analyze_energy_landscape.jl
# Three things -ln P(Rg) is not, computed properly.
#
#  1. REAL POTENTIAL ENERGY.  <U | Rg> from the analytic potentials, split into
#     bond / bend / WCA.  This is an energy in simulation units (kBT = 0.1).
#     The ACTIVE force is non-conservative and contributes no potential — its
#     absence is exactly why -ln P is not a free energy.
#
#  2. KRAMERS-MOYAL EFFECTIVE POTENTIAL.  For dx/dt = D1(x) + sqrt(2 D2(x)) xi,
#         D1(x) = <dRg | Rg=x> / dt          (drift)
#         D2(x) = <dRg^2 | Rg=x> / (2 dt)    (diffusion)
#         U_eff(x) = -integral D1/D2 dx
#     This makes NO Boltzmann assumption, so it is valid in a driven steady state,
#     and its barriers are the ones that actually set the rates.
#
#  3. THE DISTORTION.  Theory says  -ln P = U_eff + ln D2 + const.  So -ln P equals
#     the kinetic potential only where D2 is flat.  Plotting all three tests that
#     directly — if U_eff + ln D2 overlays -ln P, the reconstruction is consistent.
#
# Usage:  julia --project scripts/analyze_energy_landscape.jl [folder] [kappa] [f0]

using CairoMakie, Printf, Statistics, TOML, JLD2
const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "src", "EngineRing.jl")); using .EngineRing
include(joinpath(ROOT, "scripts", "ring_energy.jl")); using .RingEnergy

folder = length(ARGS)>=1 ? ARGS[1] : "Results_N17_bistable_long"
κsel   = length(ARGS)>=2 ? parse(Float64,ARGS[2]) : 1.0
f0sel  = length(ARGS)>=3 ? parse(Float64,ARGS[3]) : 2.2143464
const BURN_S = 5.0
const NBIN = 44

dir = joinpath(ROOT,"results",folder)
Rg_all=Float64[]; dR_all=Float64[]; Rg_pair=Float64[]
Ub=Float64[]; Ue=Float64[]; Uw=Float64[]; RgU=Float64[]
dt = NaN; nrun = 0
for rn in sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
    par=joinpath(dir,rn,"params.toml"); obs=joinpath(dir,rn,"observables.jld2")
    trj=joinpath(dir,rn,"traj.jld2")
    (isfile(par)&&isfile(obs)&&isfile(trj)) || continue
    ep=TOML.parsefile(par)["params"]
    (isapprox(Float64(ep["kappa"]),κsel;rtol=1e-4) && isapprox(Float64(ep["f0"]),f0sel;rtol=1e-4)) || continue
    p = EngineRing.Params(N=Int(ep["N"]), l=ep["l"], ks=ep["ks"], κ=ep["kappa"],
        theta_max=ep["theta_max"], theta_buf=ep["theta_buf"], k_stop=ep["k_stop"],
        sigma=ep["sigma"], epsilon=ep["epsilon"], topology=Symbol(ep["topology"]),
        f0=ep["f0"], mu=ep["mu"], kBT=ep["kBT"], dt=ep["dt"])
    d = jldopen(f->(t=f["times"], Rg=f["Rg"]), obs, "r")
    pos = jldopen(f->f["positions"], trj, "r")
    i0 = something(findfirst(>=(BURN_S), d.t), 1)
    global dt = (d.t[end]-d.t[1])/(length(d.t)-1)
    R = @view d.Rg[i0:end]
    append!(Rg_all, R)
    append!(Rg_pair, @view R[1:end-1]); append!(dR_all, diff(R))
    for k in i0:size(pos,3)
        r = @view pos[:,:,k]
        push!(Ub, RingEnergy.bond_energy(r,p)); push!(Ue, RingEnergy.bend_energy(r,p))
        push!(Uw, RingEnergy.wca_energy(r,p)); push!(RgU, d.Rg[k])
    end
    global nrun += 1
end
nrun == 0 && error("no runs at kappa=$κsel f0=$f0sel in $folder")
@printf("cell κ=%.3f f₀=%.4f  ·  %d runs  ·  %d frames  ·  dt = %.3f s\n",
        κsel, f0sel, nrun, length(Rg_all), dt)

lo, hi = quantile(Rg_all,0.002), quantile(Rg_all,0.998)
edges = range(lo,hi,length=NBIN+1); ctr=(edges[1:end-1].+edges[2:end])./2; w=step(edges)
bin(x) = (x < lo || x > hi) ? 0 : clamp(Int(fld(x-lo,w))+1, 1, NBIN)

# -ln P
cnt = zeros(Int,NBIN); for x in Rg_all; b=bin(x); b>0 && (cnt[b]+=1); end
P = cnt ./ sum(cnt); okP = P .> 0
mlnP = -log.(P[okP])

# Kramers-Moyal
s1=zeros(NBIN); s2=zeros(NBIN); nb=zeros(Int,NBIN)
for (x,dx) in zip(Rg_pair,dR_all)
    b=bin(x); b==0 && continue
    s1[b]+=dx; s2[b]+=dx*dx; nb[b]+=1
end
okK = nb .>= 200
D1 = (s1 ./ max.(nb,1)) ./ dt
D2 = (s2 ./ max.(nb,1)) ./ (2dt)
# U_eff = -int D1/D2 dx  (trapezoid over the well-sampled bins)
idx = findall(okK)
Ueff = zeros(length(idx)); integ = -D1[idx] ./ D2[idx]
for i in 2:length(idx)
    Ueff[i] = Ueff[i-1] + 0.5*(integ[i]+integ[i-1])*(ctr[idx[i]]-ctr[idx[i-1]])
end
lnD2 = log.(D2[idx])
recon = Ueff .+ lnD2

# <U | Rg>
eb=zeros(NBIN); ee=zeros(NBIN); ew=zeros(NBIN); ne=zeros(Int,NBIN)
for (x,a,b,c) in zip(RgU,Ub,Ue,Uw)
    k=bin(x); k==0 && continue
    eb[k]+=a; ee[k]+=b; ew[k]+=c; ne[k]+=1
end
okE = ne .>= 200
mb=eb[okE]./ne[okE]; me=ee[okE]./ne[okE]; mw=ew[okE]./ne[okE]
@printf("mean energies over the window:  U_bond %.1f   U_bend %.2f   U_wca %.4f   (kBT = 0.1)\n",
        mean(Ub), mean(Ue), mean(Uw))

sh(v) = v .- minimum(v)
fig = Figure(size=(1680,560))
ax1 = Axis(fig[1,1], xlabel="Rg", ylabel="[arb.]",
           title="A   −ln P is NOT the kinetic potential")
lines!(ax1, ctr[okP], sh(mlnP); linewidth=2.6, label="−ln P(Rg)")
lines!(ax1, ctr[idx], sh(Ueff); linewidth=2.6, label="U_eff  = −∫ D₁/D₂ dRg")
lines!(ax1, ctr[idx], sh(recon); linewidth=2.0, linestyle=:dash, label="U_eff + ln D₂  (should overlay −ln P)")
axislegend(ax1; position=:ct, framevisible=false, labelsize=10)

ax2 = Axis(fig[1,2], xlabel="Rg", ylabel="D₂  (drift D₁ dashed)",
           title="B   Diffusion is NOT flat in Rg")
lines!(ax2, ctr[idx], D2[idx]; linewidth=2.6, color=:seagreen, label="D₂ (diffusion)")
lines!(ax2, ctr[idx], D1[idx]; linewidth=2.0, color=:crimson, linestyle=:dash, label="D₁ (drift)")
hlines!(ax2, [0.0]; color=(:black,0.3), linewidth=1)
axislegend(ax2; position=:rt, framevisible=false, labelsize=10)

ax3 = Axis(fig[1,3], xlabel="Rg", ylabel="⟨U | Rg⟩  [sim units]",
           title="C   Real potential energy, by term")
lines!(ax3, ctr[okE], sh(mb); linewidth=2.6, label=@sprintf("bond  (mean %.0f)", mean(Ub)))
lines!(ax3, ctr[okE], sh(me); linewidth=2.6, label=@sprintf("bend  (mean %.2f)", mean(Ue)))
lines!(ax3, ctr[okE], sh(mw); linewidth=2.6, label=@sprintf("WCA   (mean %.3f)", mean(Uw)))
axislegend(ax3; position=:ct, framevisible=false, labelsize=10)
Label(fig[2,1:3], @sprintf("κ = %.2f, f₀ = %.3f (Φ = %.2f) · %d runs · %d frames · each curve shifted to its own minimum · active force is non-conservative and has no potential",
      κsel, f0sel, f0sel/κsel, nrun, length(Rg_all)); fontsize=11, tellwidth=false)

base = joinpath(ROOT,"results","campaign","energy_landscape")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("✓ ", relpath(base,ROOT), ".png / .pdf")
