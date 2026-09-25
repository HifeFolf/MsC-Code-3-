# markov_check.jl
# The last untested assumption under routes 2 and 3.
#
# Kramers-Moyal takes D1 = <dRg>/tau and D2 = <dRg^2>/2tau from a FINITE sampling
# lag tau = 0.05 s, and treats Rg as Markovian at that lag.  Three tests:
#
#  0. IS IT A DIFFUSION AT ALL?  For a diffusion <dRg^2> grows like tau.  Fit
#     <dRg^2> ~ tau^p at the two wells and at the barrier top over tau = 0.05-0.4 s.
#     p = 1 diffusive; p < 1 confined (the displacement saturates inside a well);
#     p -> 2 ballistic (increments stay correlated across the whole lag, i.e. the
#     coordinate carries memory of where it was heading).
#
#  1. LAG DEPENDENCE.  Re-estimate D1, D2 and the barrier at tau = m*dt for
#     m = 1,2,3,4,6,8 and extrapolate to tau -> 0.  The dependence turns out to be
#     strongly curved, so the extrapolation uses the three SHORTEST lags only; a
#     line through all six would not describe the small-tau end at all.
#
#  2. CHAPMAN-KOLMOGOROV.  For a Markov process the two-step transition matrix
#     equals the one-step matrix squared:  P(2 tau) = P(tau)^2.  Compare them over
#     coarse Rg bins, measuring the occupancy-weighted total-variation distance
#     per row.  A raw distance means nothing on its own, so the same distance is
#     computed between two independent halves of the data (odd vs even runs):
#     that is the noise floor this test can possibly reach.  CK distance at or
#     near the floor = no detectable memory; well above it = memory.
#
# Caveat kept in mind: binning a continuous process is itself a coarse-graining,
# so this detects memory beyond the bin resolution, not memory of any kind.
#
# Outputs (campaign folder): markov_check.png/.pdf, markov_check.csv
#
# Usage:  julia --project scripts/markov_check.jl [kappa f0]   (reference cell for the plots)

using CairoMakie, Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
const FOLDERS = ["Results_N17_wellcuts", "Results_N17_bistable_long"]
const BURN_S  = 5.0
const RG_FRAC = 0.70
const NBIN_KM = 28      # bins for D1/D2, as in analyze_kramers_rates.jl
const MINB    = 40
const NBIN_CK = 12      # coarser bins for the transition matrices
const LAGS    = [1,2,3,4,6,8]
κref  = length(ARGS) >= 2 ? parse(Float64,ARGS[1]) : 1.0
f0ref = length(ARGS) >= 2 ? parse(Float64,ARGS[2]) : 2.2143

# ── load Rg per cell, kept per run so no pair crosses a run boundary ─────────
series = Dict{Tuple{Float64,Float64},Vector{Vector{Float64}}}()
dtg = NaN; thr = NaN
for folder in FOLDERS
    dir = joinpath(ROOT,"results",folder); isdir(dir) || continue
    rs = sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
    @printf("scanning %d runs in %s…\n", length(rs), folder)
    for rn in rs
        par=joinpath(dir,rn,"params.toml"); obs=joinpath(dir,rn,"observables.jld2")
        (isfile(par)&&isfile(obs)) || continue
        ep = TOML.parsefile(par)["params"]
        key = (round(Float64(ep["kappa"]),digits=4), round(Float64(ep["f0"]),digits=4))
        global thr = RG_FRAC*(Int(ep["N"])/(2π))
        d = jldopen(f->haskey(f,"Rg") ? (t=f["times"],Rg=f["Rg"]) : nothing, obs, "r")
        d === nothing && continue
        i0 = something(findfirst(>=(BURN_S), d.t), 1)
        length(d.t) - i0 < 200 && continue
        global dtg = (d.t[end]-d.t[1])/(length(d.t)-1)
        push!(get!(series, key, Vector{Float64}[]), collect(d.Rg[i0:end]))
    end
end
@printf("  %d cells, dt = %.3f s\n", length(series), dtg)

binner(lo,hi,n) = (x -> (x < lo || x > hi) ? 0 : clamp(Int(fld((x-lo)/((hi-lo)/n)))+1, 1, n))

# ── D1, D2, U_eff and the barrier at lag m ──────────────────────────────────
function km_at_lag(vs, m, nbin, minb)
    all = reduce(vcat, vs)
    lo, hi = quantile(all,0.002), quantile(all,0.998)
    hi-lo < 0.5 && return nothing
    w = (hi-lo)/nbin; ctr = [lo + (i-0.5)*w for i in 1:nbin]
    bin(x) = (x<lo||x>hi) ? 0 : clamp(Int(fld(x-lo,w))+1,1,nbin)
    s1=zeros(nbin); s2=zeros(nbin); nb=zeros(Int,nbin)
    for v in vs, i in 1:length(v)-m
        b = bin(v[i]); b==0 && continue
        dx = v[i+m]-v[i]; s1[b]+=dx; s2[b]+=dx*dx; nb[b]+=1
    end
    ok = nb .>= minb
    best = 0:-1; cur = 0
    for i in 1:nbin
        cur = ok[i] ? cur+1 : 0
        cur > length(best) && (best = (i-cur+1):i)
    end
    length(best) < 9 && return nothing
    τ = m*dtg
    x = ctr[best]
    D1 = (s1[best] ./ nb[best]) ./ τ
    D2 = (s2[best] ./ nb[best]) ./ (2τ)
    U = zeros(length(x)); g = -D1 ./ D2
    for i in 2:length(x); U[i] = U[i-1] + 0.5*(g[i]+g[i-1])*(x[i]-x[i-1]); end
    n = length(x)
    bt = 2 + argmax(@view U[3:n-2])
    li = argmin(@view U[1:bt]); ri = bt - 1 + argmin(@view U[bt:end])
    (x[li] > 0.9thr || x[ri] < 1.1thr) && return nothing
    (x=x, D1=D1, D2=D2, U=U, bar_f=U[bt]-U[li], bar_o=U[bt]-U[ri], τ=τ,
     msd = (s2[best] ./ nb[best]),            # <dRg^2> per bin, before dividing by tau
     xwell_f=x[li], xtop=x[bt], xwell_o=x[ri])
end

# <dRg^2> at a fixed Rg, as a function of lag: slope 1 in log-log = diffusive,
# slope 2 = ballistic (the increments stay correlated over the whole lag)
msd_at(r, x0) = r.msd[argmin(abs.(r.x .- x0))]
function loglogslope(τs, ys)
    m = isfinite.(ys) .& (ys .> 0)
    sum(m) < 3 && return NaN
    lx = log.(τs[m]); ly = log.(ys[m])
    sum((lx .- mean(lx)) .* (ly .- mean(ly))) / sum((lx .- mean(lx)).^2)
end

# ── transition matrices over coarse bins ────────────────────────────────────
function tmat(vs, m, lo, hi, n)
    w = (hi-lo)/n; bin(x) = (x<lo||x>hi) ? 0 : clamp(Int(fld(x-lo,w))+1,1,n)
    N = zeros(n,n)
    for v in vs, i in 1:length(v)-m
        a = bin(v[i]); b = bin(v[i+m]); (a>0 && b>0) && (N[a,b] += 1)
    end
    N
end
function rownorm(N)
    P = copy(N); r = vec(sum(N, dims=2))
    for i in axes(P,1); r[i] > 0 && (P[i,:] ./= r[i]); end
    P, r
end
# occupancy-weighted mean total-variation distance between two transition matrices
function tvdist(N1, N2)
    P1, r1 = rownorm(N1); P2, r2 = rownorm(N2)
    w = r1 .+ r2; s = 0.0; W = 0.0
    for i in axes(P1,1)
        (r1[i] < 50 || r2[i] < 50) && continue
        s += w[i] * 0.5*sum(abs, P1[i,:] .- P2[i,:]); W += w[i]
    end
    W == 0 ? NaN : s/W
end

function ck_test(vs, m, nbin)
    all = reduce(vcat, vs)
    lo, hi = quantile(all,0.002), quantile(all,0.998)
    N1 = tmat(vs, m, lo, hi, nbin); N2 = tmat(vs, 2m, lo, hi, nbin)
    P1, r1 = rownorm(N1)
    sq = P1*P1
    # CK discrepancy: measured P(2tau) vs P(tau)^2, weighted by occupancy
    s = 0.0; W = 0.0; P2, r2 = rownorm(N2)
    for i in axes(P2,1)
        (r1[i] < 50 || r2[i] < 50) && continue
        s += r2[i] * 0.5*sum(abs, P2[i,:] .- sq[i,:]); W += r2[i]
    end
    ck = W == 0 ? NaN : s/W
    # noise floor: same quantity between two independent halves of the data
    odd = vs[1:2:end]; even = vs[2:2:end]
    floor = (length(odd) >= 2 && length(even) >= 2) ?
            tvdist(tmat(odd, 2m, lo, hi, nbin), tmat(even, 2m, lo, hi, nbin)) : NaN
    (ck=ck, floor=floor)
end

# ── run both tests on every cell with enough data ───────────────────────────
rows = NamedTuple[]
for (key, vs) in series
    sum(length, vs) < 40000 && continue
    base = km_at_lag(vs, 1, NBIN_KM, MINB); base === nothing && continue
    bars = Tuple{Float64,Float64,Float64}[]        # (tau, bar_f, bar_o)
    msds = Tuple{Float64,Float64,Float64,Float64}[] # (tau, msd at well_f, top, well_o)
    for m in LAGS
        r = km_at_lag(vs, m, NBIN_KM, MINB); r === nothing && continue
        push!(bars, (r.τ, r.bar_f, r.bar_o))
        push!(msds, (r.τ, msd_at(r, base.xwell_f), msd_at(r, base.xtop), msd_at(r, base.xwell_o)))
    end
    length(bars) < 4 && continue
    # extrapolate on the SHORTEST lags only: the lag dependence is visibly curved,
    # so a line through all six would not be an extrapolation of the small-tau end
    fit0(y, n) = begin
        xs = [b[1] for b in bars][1:n]; ys = y[1:n]; mx = mean(xs); my = mean(ys)
        my - (sum((xs .- mx).*(ys .- my))/sum((xs .- mx).^2))*mx
    end
    bf = [b[2] for b in bars]; bo = [b[3] for b in bars]
    τs = [m[1] for m in msds]
    ck = ck_test(vs, 1, NBIN_CK)
    push!(rows, (kappa=key[1], f0=key[2], nfr=sum(length,vs),
                 bar_f=base.bar_f, bar_o=base.bar_o,
                 bar0_f=fit0(bf,3), bar0_o=fit0(bo,3),
                 bar0_f_all=fit0(bf,length(bf)), bar0_o_all=fit0(bo,length(bo)),
                 exp_wf=loglogslope(τs, [m[2] for m in msds]),
                 exp_top=loglogslope(τs, [m[3] for m in msds]),
                 exp_wo=loglogslope(τs, [m[4] for m in msds]),
                 ck=ck.ck, floor=ck.floor, bars=bars, msds=msds))
end
@printf("\n%d cells pass both tests' data requirements\n\n", length(rows))

biasf = [(r.bar_f - r.bar0_f)/r.bar0_f for r in rows if isfinite(r.bar0_f) && r.bar0_f > 0.3]
biaso = [(r.bar_o - r.bar0_o)/r.bar0_o for r in rows if isfinite(r.bar0_o) && r.bar0_o > 0.3]
@printf("0. IS Rg EVEN DIFFUSIVE?  <dRg²> ~ tau^p over %.2f–%.2f s   [p=1 diffusive, p=2 ballistic]\n",
        LAGS[1]*dtg, LAGS[end]*dtg)
for (nm, f) in (("folded well", r->r.exp_wf), ("barrier top", r->r.exp_top), ("open well", r->r.exp_wo))
    e = [f(r) for r in rows if isfinite(f(r))]
    isempty(e) && continue
    @printf("   %-12s p = %.2f  [10-90%%: %.2f … %.2f]  n=%d\n",
            nm, median(e), quantile(e,0.1), quantile(e,0.9), length(e))
end

@printf("\n1. LAG DEPENDENCE — barrier at tau=%.3f s vs the tau->0 intercept\n", dtg)
@printf("   folded barrier: median bias %+.1f%%   [10-90%%: %+.1f%% … %+.1f%%]  n=%d\n",
        100median(biasf), 100quantile(biasf,0.1), 100quantile(biasf,0.9), length(biasf))
@printf("   open   barrier: median bias %+.1f%%   [10-90%%: %+.1f%% … %+.1f%%]  n=%d\n",
        100median(biaso), 100quantile(biaso,0.1), 100quantile(biaso,0.9), length(biaso))

cks = [r.ck for r in rows if isfinite(r.ck)]; fls = [r.floor for r in rows if isfinite(r.floor)]
rat = [r.ck/r.floor for r in rows if isfinite(r.ck) && isfinite(r.floor) && r.floor > 0]
@printf("\n2. CHAPMAN-KOLMOGOROV — P(2tau) vs P(tau)^2, total-variation distance\n")
@printf("   CK distance   median %.4f   [10-90%%: %.4f … %.4f]\n", median(cks), quantile(cks,0.1), quantile(cks,0.9))
@printf("   noise floor   median %.4f   [10-90%%: %.4f … %.4f]\n", median(fls), quantile(fls,0.1), quantile(fls,0.9))
@printf("   ratio CK/floor median %.2f  [10-90%%: %.2f … %.2f]   (1 = indistinguishable from noise)\n",
        median(rat), quantile(rat,0.1), quantile(rat,0.9))

open(joinpath(ROOT,"results","campaign","markov_check.csv"), "w") do io
    println(io, "kappa,f0,frames,bar_folded,bar_folded_tau0,bar_open,bar_open_tau0,msd_exp_foldwell,msd_exp_top,msd_exp_openwell,ck_distance,noise_floor,ck_over_floor")
    for r in sort(rows, by=x->(x.f0,x.kappa))
        @printf(io, "%.4f,%.4f,%d,%.4g,%.4g,%.4g,%.4g,%.3f,%.3f,%.3f,%.5g,%.5g,%.4g\n",
                r.kappa, r.f0, r.nfr, r.bar_f, r.bar0_f, r.bar_o, r.bar0_o,
                r.exp_wf, r.exp_top, r.exp_wo, r.ck, r.floor, r.ck/r.floor)
    end
end

# ── figure: reference cell in detail + the two summaries ────────────────────
ref = argmin([abs(log(r.kappa/κref)) + abs(log(r.f0/f0ref)) for r in rows])
rr = rows[ref]
@printf("\nreference cell for the plots: κ = %.3f, f₀ = %.3f\n", rr.kappa, rr.f0)
vs = series[(rr.kappa, rr.f0)]
fig = Figure(size=(1800,1030))
Label(fig[0,1:3], "Markov check — is Rg a diffusion at our sampling lag?";
      fontsize=17, font=:bold, tellwidth=false)
cg = cgrad(:viridis, length(LAGS); categorical=true)

ax1 = Axis(fig[1,1], xlabel="Rg", ylabel="D₁  [Rg/s]",
           title=@sprintf("A   drift at lag τ  (κ=%.2f, f₀=%.2f)", rr.kappa, rr.f0))
ax2 = Axis(fig[1,2], xlabel="Rg", ylabel="D₂  [Rg²/s]", title="B   diffusion at lag τ")
for (i,m) in enumerate(LAGS)
    r = km_at_lag(vs, m, NBIN_KM, MINB); r === nothing && continue
    lines!(ax1, r.x, r.D1; color=cg[i], linewidth=2, label=@sprintf("%.2f s", r.τ))
    lines!(ax2, r.x, r.D2; color=cg[i], linewidth=2, label=@sprintf("%.2f s", r.τ))
end
hlines!(ax1, [0.0]; color=(:black,0.3))
axislegend(ax1; position=:rt, framevisible=false, labelsize=9, nbanks=2)
axislegend(ax2; position=:rt, framevisible=false, labelsize=9, nbanks=2)

# C — the decisive one: how the mean square change of Rg grows with the lag
let ax = Axis(fig[1,3], xlabel="sampling lag τ  [s]", ylabel="⟨ΔRg²⟩", xscale=log10, yscale=log10,
              title="C   ⟨ΔRg²⟩ vs lag: slope 1 = diffusive, 2 = ballistic")
    τs = [m[1] for m in rr.msds]
    for (k,col,nm,e) in ((2,:crimson,"folded well",rr.exp_wf), (3,:goldenrod1,"barrier top",rr.exp_top),
                         (4,:seagreen,"open well",rr.exp_wo))
        ys = [m[k] for m in rr.msds]
        lines!(ax, τs, ys; color=col, linewidth=2.5, label=@sprintf("%s  (p = %.2f)", nm, e))
        scatter!(ax, τs, ys; color=col, markersize=8)
    end
    let y0 = rr.msds[1][3]
        lines!(ax, τs, y0 .* (τs./τs[1]);      color=(:black,0.45), linestyle=:dash, linewidth=2, label="slope 1 (diffusive)")
        lines!(ax, τs, y0 .* (τs./τs[1]).^2;   color=(:black,0.30), linestyle=:dot,  linewidth=2, label="slope 2 (ballistic)")
    end
    axislegend(ax; position=:lt, framevisible=false, labelsize=9)
end

ax3 = Axis(fig[2,1], xlabel="sampling lag τ  [s]", ylabel="barrier  [effective-noise units]",
           title="D   barrier vs lag, extrapolated to τ → 0")
for (nm,k,col) in (("folded→open",2,:crimson), ("open→folded",3,:seagreen))
    for r in rows
        lines!(ax3, [b[1] for b in r.bars], [b[k] for b in r.bars]; color=(col,0.25), linewidth=1)
    end
    lines!(ax3, [b[1] for b in rr.bars], [b[k] for b in rr.bars]; color=col, linewidth=3, label=nm)
end
scatter!(ax3, [0.0, 0.0], [rr.bar0_f, rr.bar0_o]; color=:black, markersize=11, marker=:star5,
         label="τ → 0 (reference cell)")
axislegend(ax3; position=:lt, framevisible=false, labelsize=10)

ax4 = Axis(fig[2,2], xlabel="noise floor (odd vs even runs)", ylabel="CK distance  |P(2τ) − P(τ)²|",
           title="E   Chapman–Kolmogorov against what noise alone gives")
scatter!(ax4, [r.floor for r in rows], [r.ck for r in rows]; color=:steelblue, markersize=9)
let m = maximum(max(r.ck, r.floor) for r in rows)*1.08
    lines!(ax4, [0,m], [0,m]; color=(:black,0.5), linestyle=:dash)
    text!(ax4, m*0.55, m*0.60; text="no detectable memory", fontsize=10, color=(:black,0.6))
    limits!(ax4, 0, m, 0, m)
end
# F — the same scaling exponent for every cell, not just the reference one
let ax = Axis(fig[2,3], ylabel="exponent p in ⟨ΔRg²⟩ ~ τᵖ", xticks=(1:3, ["folded\nwell","barrier\ntop","open\nwell"]),
              title="F   every cell, not just the reference one")
    for (i,(f,col)) in enumerate(((r->r.exp_wf,:crimson), (r->r.exp_top,:goldenrod1), (r->r.exp_wo,:seagreen)))
        ys = [f(r) for r in rows if isfinite(f(r))]
        xs = i .+ 0.16 .* (rand(length(ys)) .- 0.5)
        scatter!(ax, xs, ys; color=(col,0.75), markersize=8)
        lines!(ax, [i-0.3, i+0.3], fill(median(ys),2); color=:black, linewidth=2.5)
    end
    hlines!(ax, [1.0]; color=(:black,0.55), linestyle=:dash)
    hlines!(ax, [2.0]; color=(:black,0.35), linestyle=:dot)
    text!(ax, 3.3, 1.02; text="diffusive", fontsize=10, color=(:black,0.6), align=(:right,:bottom))
    text!(ax, 3.3, 1.97; text="ballistic", fontsize=10, color=(:black,0.6), align=(:right,:top))
    limits!(ax, 0.5, 3.5, 0.3, 2.15)
end

Label(fig[3,1:3], @sprintf("%d cells · 300 s runs · D₁/D₂ in %d bins · transition matrices in %d bins · lags τ = %.2f–%.2f s · black bars = medians",
      length(rows), NBIN_KM, NBIN_CK, dtg, LAGS[end]*dtg); fontsize=11, tellwidth=false)

base = joinpath(ROOT,"results","campaign","markov_check")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("\n✓ ", relpath(base,ROOT), ".png / .pdf  and  markov_check.csv")
