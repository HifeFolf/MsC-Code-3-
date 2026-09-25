# explain_fold_asymmetry.jl
# Why does the ring spend ~75% of its time in the inward fold and ~2% in the
# outward fold, when both have the same size?
#
# Geometry first: the sign of Kconn is whether the junction is CONVEX or CONCAVE.
# The ring's turning angles sum to +360 deg, so a positive Kconn is a convex bend
# (fold-out: the junction sits at the tip of a hairpin) and a negative Kconn is a
# concave dent (fold-in: the junction is pushed into the ring, which wraps round it).
#
# Candidate mechanism being TESTED here: the inward fold jams — the arms are pressed
# into contact and bead repulsion locks them — while the outward fold is held by
# nothing. Measured per state, on every STRIDE-th frame:
#   contacts        non-bonded bead pairs inside the WCA cutoff
#   U_wca, U_bend   repulsion and bending energy
#   dRg/dt active   the rate at which the ACTIVE forces alone change the ring's size
#                   = mu * sum_i F_i . dRg/dr_i,   dRg/dr_i = (r_i - r_cm)/(N Rg)
#                   negative = activity is closing the ring, positive = opening it
#   dRg/dt elastic  the same for bonds + bending + repulsion
#
# Usage:  julia --project scripts/explain_fold_asymmetry.jl [folder]

using CairoMakie, Printf, Statistics, TOML, JLD2
const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT,"src","EngineRing.jl")); using .EngineRing
include(joinpath(ROOT,"src","InitialConditions.jl")); using .InitialConditions
include(joinpath(ROOT,"scripts","ring_energy.jl")); using .RingEnergy

folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_bistable_long"
const STRIDE = 10
const K0 = 30.0
const ST = [:O, :Fo, :Fi]
const SHORT = Dict(:O=>"open", :Fo=>"fold-out", :Fi=>"fold-in")

mkparams(ep) = EngineRing.Params(N=Int(ep["N"]), l=ep["l"], ks=ep["ks"], κ=ep["kappa"],
    theta_max=ep["theta_max"], theta_buf=ep["theta_buf"], k_stop=ep["k_stop"],
    sigma=ep["sigma"], epsilon=ep["epsilon"], topology=Symbol(ep["topology"]),
    f0=ep["f0"], mu=ep["mu"], kBT=ep["kBT"], dt=ep["dt"])

function contacts(r, p)
    N=p.N; rc2=(2.0^(1/6)*p.sigma)^2; c=0
    for i in 1:N-1, j in i+1:N
        min(j-i, N-(j-i)) == 1 && continue
        dx=r[1,j]-r[1,i]; dy=r[2,j]-r[2,i]
        dx*dx+dy*dy < rc2 && (c += 1)
    end
    c
end

function run_all(dir)
    sums = Dict(s => zeros(5) for s in ST)       # contacts, U_wca, U_bend, v_act, v_el
    percell = Dict{Tuple{Float64,Float64,Symbol},Vector{Float64}}()   # (kappa,f0,state) => sums..., n
    cnt  = Dict(s => 0 for s in ST)
    RGs  = Dict(s => Float64[] for s in ST); KCs = Dict(s => Float64[] for s in ST)
    cand = Dict(s => Any[] for s in ST)
    bt = Symbol[]
    for rn in sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
        par=joinpath(dir,rn,"params.toml"); obs=joinpath(dir,rn,"observables.jld2"); trj=joinpath(dir,rn,"traj.jld2")
        (isfile(par) && isfile(obs) && isfile(trj)) || continue
        p = mkparams(TOML.parsefile(par)["params"])
        bt = InitialConditions.antipolar_bead_types(p.N)
        thr = 0.70*(p.N/(2π))
        d = jldopen(f->(t=f["times"], Rg=f["Rg"], K=rad2deg.(f["Kconn"])), obs, "r")
        pos = jldopen(f->f["positions"], trj, "r")
        i0 = something(findfirst(>=(5.0), d.t), 1)
        for k in i0:STRIDE:size(pos,3)
            Rg = d.Rg[k]; K = d.K[k]
            s = Rg > 1.1thr ? :O : (Rg < 0.9thr ? (K > K0 ? :Fo : (K < -K0 ? :Fi : :none)) : :none)
            s === :none && continue
            r = pos[:,:,k]
            Fa = zeros(2,p.N); EngineRing.active_forces!(Fa, r, p, bt)
            Fc = EngineRing.bond_forces(r,p); EngineRing.bending_forces!(Fc,r,p); EngineRing.wca_forces!(Fc,r,p)
            rcv = r .- sum(r,dims=2) ./ p.N
            sums[s] .+= [contacts(r,p), RingEnergy.wca_energy(r,p), RingEnergy.bend_energy(r,p),
                         p.mu*sum(Fa .* rcv)/(p.N*Rg), p.mu*sum(Fc .* rcv)/(p.N*Rg)]
            cnt[s] += 1
            pc = get!(percell, (p.κ, p.f0, s), zeros(6))
            pc[1:5] .+= [contacts(r,p), RingEnergy.wca_energy(r,p), RingEnergy.bend_energy(r,p),
                         p.mu*sum(Fa .* rcv)/(p.N*Rg), p.mu*sum(Fc .* rcv)/(p.N*Rg)]; pc[6] += 1
            push!(RGs[s], Rg); push!(KCs[s], K)
            (cnt[s] % 25 == 0 && length(cand[s]) < 400) && push!(cand[s], (Rg, K, copy(r), copy(Fa), contacts(r,p)))
        end
    end
    sums, cnt, RGs, KCs, cand, bt, percell
end

dir = joinpath(ROOT,"results",folder)
sums, cnt, RGs, KCs, cand, bt, percell = run_all(dir)
println("Averages per state (every $(STRIDE)th frame, all 401 runs):\n")
@printf("  %-9s %8s %9s %8s %8s %15s %15s\n", "state", "frames", "contacts", "U_wca", "U_bend",
        "dRg/dt active", "dRg/dt elastic")
for s in ST
    n = cnt[s]; n == 0 && continue
    m = sums[s] ./ n
    @printf("  %-9s %8d %9.2f %8.3f %8.2f %15.4f %15.4f\n", SHORT[s], n, m[1], m[2], m[3], m[4], m[5])
end
println("\n  dRg/dt: negative = pushing the ring closed, positive = pushing it open  (units: Rg per s)")


# ── within-cell comparison: removes the confound that the two folds are sampled
#    from different (kappa, f0) mixes. Only cells with >= MIN_FR frames of each fold.
const MIN_FR = 30
const KBT = 0.1
names = ["contacts", "U_wca [kBT]", "U_bend [kBT]", "dRg/dt active", "dRg/dt elastic"]
cells = unique([(k[1],k[2]) for k in keys(percell)])
paired = [(c, percell[(c...,:Fi)], percell[(c...,:Fo)]) for c in cells
          if haskey(percell,(c...,:Fi)) && haskey(percell,(c...,:Fo)) &&
             percell[(c...,:Fi)][6] >= MIN_FR && percell[(c...,:Fo)][6] >= MIN_FR]
println("\nWithin-cell comparison, fold-in minus fold-out ($(length(paired)) cells with ≥$(MIN_FR) frames of each):")
@printf("  %-16s %14s %22s\n", "quantity", "median diff", "cells where in > out")
for q in 1:5
    sc = q in (2,3) ? 1/KBT : 1.0
    d = [sc*(a[q]/a[6] - b[q]/b[6]) for (_,a,b) in paired]
    @printf("  %-16s %+14.3f %16d / %d\n", names[q], median(d), count(>(0), d), length(d))
end
# ── representative snapshot per state: candidate nearest that state's median ─
COL = Dict(:active=>RGBAf(0.85,0.33,0.10,1), :active_rev=>RGBAf(0.20,0.45,0.70,1), :passive=>RGBAf(0.1,0.1,0.1,1))
fig = Figure(size=(1500,560))
for (col, s) in enumerate(ST)
    isempty(cand[s]) && continue
    mR, mK = median(RGs[s]), median(KCs[s])
    best = cand[s][argmin([abs(c[1]-mR)/0.1 + abs(c[2]-mK)/20 for c in cand[s]])]
    Rg, K, r, Fa, nc = best
    q = r .- sum(r,dims=2) ./ size(r,2)
    ax = Axis(fig[1,col], aspect=DataAspect(),
              title=@sprintf("%s    Rg %.2f · Kconn %+.0f° · %d contacts", SHORT[s], Rg, K, nc))
    hidedecorations!(ax); hidespines!(ax)
    lines!(ax, vcat(q[1,:],q[1,1]), vcat(q[2,:],q[2,1]); color=(:gray40,0.8), linewidth=2.2)
    sc = 0.55 / max(maximum(sqrt.(sum(Fa.^2,dims=1))), 1e-12)
    for i in 1:size(q,2)
        bt[i] == :passive && continue
        arrows!(ax, [q[1,i]], [q[2,i]], [sc*Fa[1,i]], [sc*Fa[2,i]]; color=COL[bt[i]], linewidth=1.8)
    end
    for i in 1:size(q,2)
        scatter!(ax, [q[1,i]], [q[2,i]]; color=COL[bt[i]], markersize=bt[i]==:passive ? 22 : 14,
                 strokecolor=:white, strokewidth=1)
    end
    L = maximum(abs.(q)) + 0.4; limits!(ax, -L, L, -L, L)
end
Label(fig[2,1:3], "orange = active (drives CCW) · blue = active_rev (drives CW) · black = passive junction · " *
      "arrows = active force on each bead · snapshot nearest each state's median (Rg, Kconn)";
      fontsize=12, tellwidth=false)
out = joinpath(dir, "fold_shapes")
save(out*".png", fig; px_per_unit=3)
println("\n✓ ", relpath(out, ROOT), ".png")
