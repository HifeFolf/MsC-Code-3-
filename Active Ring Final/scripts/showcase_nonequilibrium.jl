# showcase_nonequilibrium.jl
# The claim, with the evidence a reader needs to check it, and nothing else.
#
# CLAIM.  The folded state is not a minimum of the ring's energy.  It costs tens of
# kBT MORE than the open state, and the ring still spends much of its time there.
# In equilibrium that is impossible; the drive is what creates and holds the state.
#
# Nothing here uses -ln P as an energy, no Kramers-Moyal, no Markov assumption, no
# binning choice that changes the answer.  Only two measured quantities:
#
#   <U | Rg>  the ring's own potential energy, computed from the SAME formulas the
#             engine differentiates for its forces (ring_energy.jl), checked here
#             by finite-differencing U against those forces.
#   P(folded)/P(open)  how the ring actually divides its time, from the dwell
#             tracker used throughout.
#
# Boltzmann says the second must equal exp(-dU/kBT).  It does not, by ~30 orders of
# magnitude, and the passive control (f0 = 0, same stiffness, same temperature)
# shows the folded state simply does not exist without the drive.
#
# Outputs (campaign folder): showcase_nonequilibrium.png/.pdf, showcase_nonequilibrium.csv
#
# Usage:  julia --project scripts/showcase_nonequilibrium.jl

using CairoMakie, Printf, Statistics, TOML, JLD2, Random
const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT,"src","EngineRing.jl")); using .EngineRing
include(joinpath(ROOT,"src","InitialConditions.jl")); using .InitialConditions
include(joinpath(ROOT,"scripts","ring_energy.jl")); using .RingEnergy

const PASSIVE = "Results_N17_passive"
const ACTIVE  = "Results_N17_bistable_long"
const BURN_S  = 5.0
const RG_FRAC = 0.70
const MIN_DWELL_S = 0.5
const KREF, FREF = 1.0, 2.2143464          # the cell shown in panels A and B

mkparams(ep) = EngineRing.Params(N=Int(ep["N"]), l=ep["l"], ks=ep["ks"], κ=ep["kappa"],
    theta_max=ep["theta_max"], theta_buf=ep["theta_buf"], k_stop=ep["k_stop"],
    sigma=ep["sigma"], epsilon=ep["epsilon"], topology=Symbol(ep["topology"]),
    f0=ep["f0"], mu=ep["mu"], kBT=ep["kBT"], dt=ep["dt"])

# ── CHECK 1: is our energy function the engine's energy function? ───────────
# -dU/dr computed numerically must equal the conservative forces the engine uses.
function force_check(p)
    Random.seed!(7)
    worst = 0.0
    for _ in 1:3
        r = InitialConditions.perfect_circle_ic(p, fill(:passive, p.N)).r .+ 0.05 .* randn(2, p.N)
        F = EngineRing.bond_forces(r,p); EngineRing.bending_forces!(F,r,p); EngineRing.wca_forces!(F,r,p)
        h = 1e-6
        for i in 1:p.N, d in 1:2
            rp = copy(r); rp[d,i] += h; rm = copy(r); rm[d,i] -= h
            num = -(RingEnergy.total_energy(rp,p) - RingEnergy.total_energy(rm,p))/(2h)
            worst = max(worst, abs(num - F[d,i])/max(abs(F[d,i]), 1.0))
        end
    end
    worst
end

# ── state tracker, identical to analyze_dwell.jl ────────────────────────────
function segments(t, Rg, thr, need)
    segs = Tuple{Symbol,Float64,Float64}[]
    state = Rg[1] < thr ? :folded : :open
    t_enter = t[1]; rl = 0; pend = :none
    for (j,x) in enumerate(Rg)
        s = x < 0.9thr ? :folded : (x > 1.1thr ? :open : :none)
        (s == :none || s == state) && (rl = 0; pend = :none; continue)
        (s == pend) ? (rl += 1) : (pend = s; rl = 1)
        if rl >= need
            push!(segs, (state, t_enter, t[j])); state = s; t_enter = t[j]; rl = 0; pend = :none
        end
    end
    push!(segs, (state, t_enter, t[end]))
    segs
end

# ── scan a campaign: energies split by state, occupancy, fold events ────────
function scan(folder; want_traj=true)
    dir = joinpath(ROOT,"results",folder)
    isdir(dir) || return nothing
    cells = Dict{Tuple{Float64,Float64},Any}()
    p_any = nothing
    for rn in sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
        par=joinpath(dir,rn,"params.toml"); obs=joinpath(dir,rn,"observables.jld2")
        trj=joinpath(dir,rn,"traj.jld2")
        (isfile(par) && isfile(obs)) || continue
        ep = TOML.parsefile(par)["params"]; p = mkparams(ep); p_any = p
        key = (round(p.κ,digits=4), round(p.f0,digits=4))
        thr = RG_FRAC*(p.N/(2π))
        d = jldopen(f->(t=f["times"], Rg=f["Rg"]), obs, "r")
        i0 = something(findfirst(>=(BURN_S), d.t), 1)
        t = d.t[i0:end]; Rg = d.Rg[i0:end]
        length(t) < 20 && continue
        dt = (t[end]-t[1])/(length(t)-1)
        c = get!(cells, key, Dict(:U=>Dict(:folded=>Float64[], :open=>Float64[]),
                                  :T=>Dict(:folded=>0.0, :open=>0.0), :events=>0,
                                  :Rg=>Float64[], :trace=>nothing, :minRg=>Inf, :p=>p, :nrun=>0))
        c[:nrun] += 1
        append!(c[:Rg], Rg); c[:minRg] = min(c[:minRg], minimum(Rg))
        segs = segments(t, Rg, thr, max(2, round(Int, MIN_DWELL_S/dt)))
        for (si,(st,ta,tb)) in enumerate(segs)
            c[:T][st] += tb - ta
            si < length(segs) && (c[:events] += 1)      # every segment but the last ended in a switch
        end
        c[:trace] === nothing && (c[:trace] = (t, Rg))
        if want_traj && isfile(trj)
            pos = jldopen(f->f["positions"], trj, "r")
            for k in i0:5:min(size(pos,3), length(d.Rg))
                x = d.Rg[k]
                st = x < 0.9thr ? :folded : (x > 1.1thr ? :open : :none)
                st === :none && continue
                push!(c[:U][st], RingEnergy.total_energy(pos[:,:,k], p))
            end
        end
    end
    (cells=cells, p=p_any)
end

println("scanning passive control…");  pas = scan(PASSIVE)
println("scanning active campaign…");  act = scan(ACTIVE)
pas === nothing && error("no passive campaign at results/$PASSIVE — run it first")

p0 = act.p
err = force_check(p0)
@printf("\nCHECK 1  energy vs engine forces: worst relative mismatch %.2e  (finite difference)\n", err)

# ── CHECK 2: is the thermostat the temperature we claim? ────────────────────
# With the motors off, every internal force cancels in the sum, so the ring's centre
# of mass is a free Brownian particle:  D_com = mu*kBT/N,  EXACTLY — Euler-Maruyama
# integrates free diffusion with no discretisation error, unlike the stiff bond modes.
# The run rigidly shifts the ring by L when it leaves the box, so undo those jumps.
function unwrap!(c, L)
    off = 0.0
    for i in 2:length(c)
        d = c[i] - c[i-1] + off
        abs(d) > L/2 && (off -= sign(d)*L)
        c[i] += off
    end
    c
end
function com_diffusion(folder)
    dir = joinpath(ROOT,"results",folder)
    lags = [2,5,10,20,50,100]; s = zeros(length(lags)); n = zeros(Int,length(lags)); dtf = NaN
    for rn in sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
        trj=joinpath(dir,rn,"traj.jld2"); obs=joinpath(dir,rn,"observables.jld2")
        mt =joinpath(dir,rn,"metadata.json")
        (isfile(trj)&&isfile(obs)&&isfile(mt)) || continue
        m = match(r"\"L\":\s*([0-9.]+)", read(mt,String)); m === nothing && continue
        L = parse(Float64, m.captures[1])
        pos = jldopen(f->f["positions"], trj, "r")
        t   = jldopen(f->f["times"], obs, "r"); dtf = t[2]-t[1]
        cx = unwrap!(vec(sum(pos[1,:,:],dims=1))./size(pos,2), L)
        cy = unwrap!(vec(sum(pos[2,:,:],dims=1))./size(pos,2), L)
        for (li,lg) in enumerate(lags), i in 1:lg:length(cx)-lg
            s[li] += (cx[i+lg]-cx[i])^2 + (cy[i+lg]-cy[i])^2; n[li] += 1
        end
    end
    all(iszero, n) && return nothing
    (lags=lags .* dtf, D=(s ./ max.(n,1)) ./ (4 .* lags .* dtf), n=n)
end
cd_ = com_diffusion(PASSIVE)
if cd_ !== nothing
    Dth = p0.mu * p0.kBT / p0.N
    @printf("CHECK 2  thermostat, passive ring: centre-of-mass diffusion\n")
    @printf("         theory  D = μ·kBT/N = %.6f   (exact for a free coordinate)\n", Dth)
    for (τ,D,n) in zip(cd_.lags, cd_.D, cd_.n)
        @printf("         lag %5.2f s   D = %.6f   (%.1f%% off, %d samples)\n", τ, D, 100*(D/Dth-1), n)
    end
end

# ── passive control: did it ever fold? ──────────────────────────────────────
pv       = collect(values(pas.cells))
tot_T    = sum(c[:T][:folded] + c[:T][:open] for c in pv)
tot_fold = sum(c[:T][:folded] for c in pv)
tot_ev   = sum(c[:events] for c in pv)
minRg    = minimum(c[:minRg] for c in pv)
npas     = sum(c[:nrun] for c in pv)
thr = RG_FRAC*(p0.N/(2π))
@printf("\nCHECK 3  passive control (f₀ = 0): %d runs, %.0f s of simulation\n", npas, tot_T)
@printf("         time folded %.1f s (%.4f%%) · transitions %d · smallest Rg ever reached %.2f (fold threshold %.2f)\n",
        tot_fold, 100*tot_fold/tot_T, tot_ev, minRg, thr)

# ── the measurement: energy gap vs occupancy, per cell ──────────────────────
rows = NamedTuple[]
for (key,c) in act.cells
    nf, no = length(c[:U][:folded]), length(c[:U][:open])
    (nf < 200 || no < 200) && continue
    Tf, To = c[:T][:folded], c[:T][:open]
    (Tf <= 0 || To <= 0) && continue
    dU = (mean(c[:U][:folded]) - mean(c[:U][:open])) / c[:p].kBT      # in kBT
    push!(rows, (kappa=key[1], f0=key[2], dU=dU, ratio=Tf/To, boltz=exp(-dU),
                 occ=Tf/(Tf+To), nf=nf, no=no))
end
sort!(rows, by=r->r.dU)
open(joinpath(ROOT,"results","campaign","showcase_nonequilibrium.csv"),"w") do io
    println(io, "kappa,f0,dU_folded_minus_open_kBT,measured_ratio,boltzmann_ratio,occ_folded,n_folded_frames,n_open_frames")
    for r in rows
        @printf(io,"%.4f,%.4f,%.2f,%.4g,%.4g,%.4f,%d,%d\n",
                r.kappa,r.f0,r.dU,r.ratio,r.boltz,r.occ,r.nf,r.no)
    end
end
@printf("\nTHE MEASUREMENT  %d cells where the ring visits both states\n", length(rows))
@printf("         energy cost of folding: median %.0f kBT  [%.0f … %.0f]\n",
        median([r.dU for r in rows]), minimum([r.dU for r in rows]), maximum([r.dU for r in rows]))
@printf("         measured folded/open time ratio: median %.2f\n", median([r.ratio for r in rows]))
@printf("         Boltzmann would give:            median %.1e\n", median([r.boltz for r in rows]))
@printf("         discrepancy: %.0f orders of magnitude\n",
        median([log10(r.ratio/r.boltz) for r in rows]))

# ── figure ──────────────────────────────────────────────────────────────────
refkey = (round(KREF,digits=4), round(FREF,digits=4))
ac = get(act.cells, refkey, nothing)
pc = get(pas.cells, (round(KREF,digits=4), 0.0), nothing)
fig = Figure(size=(1620,560))
Label(fig[0,1:3], "The folded state costs energy and the ring occupies it anyway — it is made by the drive";
      fontsize=17, font=:bold, tellwidth=false)

# A — energy and probability over Rg, with the passive ring's range marked
if ac !== nothing
    ax = Axis(fig[1,1], xlabel="Radius of gyration  Rg", ylabel="⟨U | Rg⟩   [kBT]",
              title="A   the energy says open; the ring sits folded")
    Rg = ac[:Rg]; lo, hi = quantile(Rg,0.002), quantile(Rg,0.998)
    edges = range(lo,hi,length=41); ctr=(edges[1:end-1].+edges[2:end])./2; w=step(edges)
    bin(x) = (x<lo||x>hi) ? 0 : clamp(Int(fld(x-lo,w))+1,1,40)
    cnt = zeros(Int,40); for x in Rg; b=bin(x); b>0 && (cnt[b]+=1); end
    P = cnt ./ sum(cnt); ok = P .> 0
    mlnP = -log.(P[ok]); mlnP .-= minimum(mlnP)
    # <U|Rg> from the saved trajectories of this cell
    dir = joinpath(ROOT,"results",ACTIVE); sU=zeros(40); nU=zeros(Int,40)
    for rn in sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
        par=joinpath(dir,rn,"params.toml"); isfile(par) || continue
        ep=TOML.parsefile(par)["params"]
        (isapprox(ep["kappa"],KREF;rtol=1e-3) && isapprox(ep["f0"],FREF;rtol=1e-3)) || continue
        p=mkparams(ep); obs=joinpath(dir,rn,"observables.jld2"); trj=joinpath(dir,rn,"traj.jld2")
        (isfile(obs)&&isfile(trj)) || continue
        d=jldopen(f->(t=f["times"],Rg=f["Rg"]),obs,"r"); pos=jldopen(f->f["positions"],trj,"r")
        i0=something(findfirst(>=(BURN_S),d.t),1)
        for k in i0:5:min(size(pos,3),length(d.Rg))
            b=bin(d.Rg[k]); b==0 && continue
            sU[b] += RingEnergy.total_energy(pos[:,:,k],p)/p.kBT; nU[b]+=1
        end
    end
    okU = nU .>= 50; U = sU[okU]./nU[okU]; U .-= minimum(U)
    # -ln P is NOT an energy and its range is ~30x smaller, so it gets its own axis;
    # the point of the panel is that the two curves disagree in shape, not in value
    ax2 = Axis(fig[1,1], yaxisposition=:right, ylabel="−ln P(Rg)   (not an energy)",
               ylabelcolor=:black, rightspinecolor=:black)
    hidexdecorations!(ax2); hidespines!(ax2, :l, :t, :b); linkxaxes!(ax, ax2)
    if pc !== nothing
        pr = pc[:Rg]
        vspan!(ax, quantile(pr,0.001), quantile(pr,0.999); color=(:steelblue,0.15))
        text!(ax, quantile(pr,0.5), maximum(U)*0.42; text="the passive ring\nnever left here",
              align=(:center,:center), fontsize=10, color=:steelblue)
    end
    l1 = lines!(ax,  ctr[okU], U;    color=:crimson, linewidth=3)
    l2 = lines!(ax2, ctr[ok],  mlnP; color=:black,   linewidth=2.5)
    vlines!(ax, [thr]; color=(:black,0.3), linestyle=:dash)
    text!(ax, ctr[okU][1], maximum(U)*0.88;
          text=@sprintf("folding costs\n%.0f kBT", maximum(U)), align=(:left,:center),
          fontsize=11, color=:crimson)
    axislegend(ax, [l1, l2], ["⟨U | Rg⟩   the ring's own energy", "−ln P(Rg)   where it spends its time"];
               position=:rt, framevisible=false, labelsize=10)
end

# B — raw traces, passive vs active
let ax = Axis(fig[1,2], xlabel="time [s]", ylabel="Rg",
              title=@sprintf("B   one run each, κ = %.1f", KREF))
    if pc !== nothing && pc[:trace] !== nothing
        t,R = pc[:trace]; lines!(ax, t, R; color=:steelblue, linewidth=1.1, label="motors off (f₀ = 0)")
    end
    if ac !== nothing && ac[:trace] !== nothing
        t,R = ac[:trace]; lines!(ax, t, R; color=:crimson, linewidth=1.1,
                                 label=@sprintf("motors on (f₀ = %.1f)", FREF))
    end
    hlines!(ax, [thr]; color=(:black,0.45), linestyle=:dash)
    text!(ax, 3, thr+0.03; text="fold threshold", fontsize=10, color=(:black,0.55))
    axislegend(ax; position=:rb, framevisible=false, labelsize=10)
end

# C — Boltzmann vs measured, every cell
let ax = Axis(fig[1,3], xlabel="energy cost of folding  [kBT]", ylabel="folded / open  time ratio",
              yscale=log10, title="C   what equilibrium would give vs what happens")
    scatter!(ax, [r.dU for r in rows], [max(r.boltz,1e-200) for r in rows];
             color=:steelblue, markersize=10, label="Boltzmann,  e^(−ΔU/kBT)")
    scatter!(ax, [r.dU for r in rows], [r.ratio for r in rows];
             color=:crimson, markersize=10, label="measured")
    hlines!(ax, [1.0]; color=(:black,0.3), linestyle=:dash)
    let r = rows[length(rows)÷2], gap = log10(r.ratio/r.boltz)
        lines!(ax, [r.dU, r.dU], [r.boltz, r.ratio]; color=(:black,0.45), linewidth=1.5)
        text!(ax, r.dU + 4, sqrt(r.boltz*r.ratio); text=@sprintf("%.0f orders\nof magnitude", gap),
              align=(:left,:center), fontsize=11)
    end
    axislegend(ax; position=:lb, framevisible=false, labelsize=10)
end

Label(fig[2,1:3], @sprintf(
  "energy computed from the engine's own potentials (agrees with its forces to %.0e) · passive control: %d runs × 300 s, %.0f s total, %d folding events, closest approach Rg %.2f vs threshold %.2f · active: %s",
  err, npas, tot_T, tot_ev, minRg, thr, ACTIVE); fontsize=11, tellwidth=false)

base = joinpath(ROOT,"results","campaign","showcase_nonequilibrium")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("\n✓ ", relpath(base,ROOT), ".png / .pdf  and  .csv")
