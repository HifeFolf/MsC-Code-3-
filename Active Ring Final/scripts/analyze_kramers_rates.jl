# analyze_kramers_rates.jl
# Route 3: barriers from switching rates, and a test of route 2.
#
# Cells come from the 300 s campaigns (well-cuts: 45 cells on six lines;
# bistable_long: 40 cells on the bistable locus; the two share no cell).
#
#   MEASURED RATE.  Segment Rg into folded/open dwells with the same hysteretic
#   tracker as analyze_dwell.jl.  The escape rate out of a state is the standard
#   censored-exponential MLE
#       k = (completed escapes from the state) / (total time spent in the state)
#   which counts the truncated first/last segments in the exposure but not as
#   events.  CV of the COMPLETE dwells tests the exponential (Kramers) assumption.
#
#   PREDICTED BARRIER.  From the same Rg series, Kramers-Moyal as in
#   analyze_energy_landscape.jl: D1 = <dRg>/dt, D2 = <dRg^2>/2dt per bin,
#   U_eff = -integral D1/D2.  Barrier = U_eff(top) - U_eff(that state's well).
#   -ln P gives a second, competing barrier estimate for the same cell.
#
#   PREDICTED RATE.  The exact mean first-passage time over the U_eff landscape
#   (1/D2 in the inner integral), and its high-barrier limit, the Kramers rate with
#   D2 at the well.  See barriers() for the boundaries.
#
# THE TEST.  Kramers says k = A exp(-dU), so ln k against the barrier should have
# slope -1 if U_eff is the potential that sets the rates.  Fitting the slope for
# both estimators says which one is right, across many cells rather than the single
# cell route 2 was built on.  Caveat: A is not identical across cells (it depends on
# D2 and on the curvatures), so scatter about the line is expected.
#
# The transition region is thinly visited, so the number of usable cells depends on
# the binning.  SENSITIVITY is scanned and printed before the figure is made.
#
# Outputs (campaign folder): kramers_rates.csv, kramers_rates.png/.pdf
#
# Usage:  julia --project scripts/analyze_kramers_rates.jl [folder ...]

using CairoMakie, Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
folders = isempty(ARGS) ? ["Results_N17_wellcuts", "Results_N17_bistable_long"] : ARGS
const RG_FRAC     = 0.70
const MIN_DWELL_S = 0.5
const BURN_S      = 5.0
const MIN_EVENTS  = 5        # min escapes before a rate is trusted
const NBIN        = 28       # default binning for the figure
const MINB        = 40       # min pairs in a bin for a usable D1/D2

# ── per-run segmentation, identical rules to analyze_dwell.jl ────────────────
function segment(t, Rg, thr, need)
    segs = Tuple{Symbol,Float64,Float64}[]
    state = Rg[1] < thr ? :folded : :open
    t_enter = t[1]; rl = 0; pend = :none
    for (j, x) in enumerate(Rg)
        s = x < 0.9thr ? :folded : (x > 1.1thr ? :open : :none)
        (s == :none || s == state) && (rl = 0; pend = :none; continue)
        (s == pend) ? (rl += 1) : (pend = s; rl = 1)
        if rl >= need
            push!(segs, (state, t_enter, t[j]))
            state = s; t_enter = t[j]; rl = 0; pend = :none
        end
    end
    push!(segs, (state, t_enter, t[end]))
    segs
end

mutable struct Cell
    exposure::Dict{Symbol,Float64}
    events::Dict{Symbol,Int}
    durs::Dict{Symbol,Vector{Float64}}
    Rg::Vector{Float64}; Rgp::Vector{Float64}; dR::Vector{Float64}
    dt::Float64; nrun::Int
end
Cell() = Cell(Dict(:folded=>0.0,:open=>0.0), Dict(:folded=>0,:open=>0),
              Dict(:folded=>Float64[],:open=>Float64[]),
              Float64[], Float64[], Float64[], NaN, 0)

# ── gather ──────────────────────────────────────────────────────────────────
cells = Dict{Tuple{String,Float64,Float64},Cell}()
thr_global = NaN
for folder in folders
    dir = joinpath(ROOT, "results", folder)
    isdir(dir) || error("missing $dir")
    runs = sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
    @printf("scanning %d runs in %s…\n", length(runs), folder)
    for rn in runs
        par = joinpath(dir,rn,"params.toml"); obs = joinpath(dir,rn,"observables.jld2")
        (isfile(par) && isfile(obs)) || continue
        ep = TOML.parsefile(par)["params"]
        key = (folder, round(Float64(ep["kappa"]),digits=4), round(Float64(ep["f0"]),digits=4))
        global thr_global = RG_FRAC * (Int(ep["N"]) / (2π))
        d = jldopen(f -> haskey(f,"Rg") ? (t=f["times"], Rg=f["Rg"]) : nothing, obs, "r")
        d === nothing && continue
        i0 = something(findfirst(>=(BURN_S), d.t), 1)
        t = d.t[i0:end]; Rg = d.Rg[i0:end]
        length(t) < 20 && continue
        dt = (t[end]-t[1])/(length(t)-1)
        c = get!(cells, key, Cell()); c.dt = dt; c.nrun += 1
        append!(c.Rg, Rg); append!(c.Rgp, @view Rg[1:end-1]); append!(c.dR, diff(Rg))
        segs = segment(t, Rg, thr_global, max(2, round(Int, MIN_DWELL_S/dt)))
        for (si,(st,ta,tb)) in enumerate(segs)
            c.exposure[st] += tb - ta
            si == length(segs) && continue      # censored: exposure only, no event
            c.events[st] += 1
            si == 1 && continue                 # censored on the left: no duration
            push!(c.durs[st], tb - ta)
        end
    end
end
@printf("  %d cells total\n", length(cells))

# ── Kramers-Moyal and -ln P barriers for one cell ───────────────────────────
function barriers(c::Cell, thr, nbin, minb)
    length(c.Rg) < 5000 && return nothing
    lo, hi = quantile(c.Rg,0.002), quantile(c.Rg,0.998)
    hi - lo < 0.5 && return nothing
    edges = range(lo,hi,length=nbin+1); ctr=(edges[1:end-1].+edges[2:end])./2; w=step(edges)
    bin(x) = (x<lo||x>hi) ? 0 : clamp(Int(fld(x-lo,w))+1,1,nbin)
    cnt=zeros(Int,nbin); s1=zeros(nbin); s2=zeros(nbin); nb=zeros(Int,nbin)
    for x in c.Rg; b=bin(x); b>0 && (cnt[b]+=1); end
    for (x,dx) in zip(c.Rgp,c.dR); b=bin(x); b==0 && continue; s1[b]+=dx; s2[b]+=dx*dx; nb[b]+=1; end
    ok = nb .>= minb
    best = 0:-1; cur = 0                      # longest contiguous usable stretch
    for i in 1:nbin
        cur = ok[i] ? cur+1 : 0
        cur > length(best) && (best = (i-cur+1):i)
    end
    length(best) < 9 && return nothing
    x = ctr[best]
    (minimum(x) > 0.9thr || maximum(x) < 1.1thr) && return nothing   # must span both states
    D1 = (s1[best] ./ nb[best]) ./ c.dt
    D2 = (s2[best] ./ nb[best]) ./ (2*c.dt)
    integ = -D1 ./ D2
    U = zeros(length(x))
    for i in 2:length(x); U[i] = U[i-1] + 0.5*(integ[i]+integ[i-1])*(x[i]-x[i-1]); end
    P = cnt[best] ./ sum(cnt[best])
    any(==(0), P) && return nothing
    mP = -log.(P)
    n = length(x); i1, i2 = 3, n-2
    i2 <= i1 && return nothing
    bt = i1 - 1 + argmax(@view U[i1:i2])
    li = argmin(@view U[1:bt]); ri = bt - 1 + argmin(@view U[bt:end])
    (x[li] > 0.9thr || x[ri] < 1.1thr) && return nothing        # wells on the right sides
    (U[bt]-U[li] < 0.3 || U[bt]-U[ri] < 0.3) && return nothing  # not actually two wells
    # Absolute Kramers prediction, no fitted parameter:
    #   k = (D2(well)/2pi) sqrt(|U''(well) U''(top)|) exp(-dU)
    # D2 belongs at the WELL for U_eff: the backward equation D1 T' + D2 T'' = -1 gives
    #   T = int_well^abs dy e^{U(y)} int^y dz e^{-U(z)}/D2(z)
    # with 1/D2 in the inner (well-dominated) integral.  D2(top) pairs with G = -ln P
    # instead; the old D2(top)-with-U_eff value is kept as kpred_top for comparison only.
    # curvature by least-squares parabola over +-2 bins
    function curv(i)
        lo = max(1,i-2); hi = min(length(x),i+2)
        hi-lo < 2 && return NaN
        xs = x[lo:hi] .- x[i]
        coef = [xs.^2 xs ones(length(xs))] \ U[lo:hi]   # NB: not `c`, which is the Cell
        2*coef[1]
    end
    cw_f, cw_o, ct = curv(li), curv(ri), curv(bt)
    pre(cw, i) = (D2[i]/(2π)) * sqrt(abs(cw*ct))
    # Exact mean first-passage time over the usable bins, by trapezoid.  Start at the
    # well; reflect at the far edge of the usable range; absorb at the OTHER state's
    # tracker threshold (1.1thr for folded escape, 0.9thr for open), which is where the
    # measured rate counts an escape.  Computed two ways that are algebraically equal:
    #   A: outer e^{U},       inner e^{-U}/D2
    #   B: outer e^{G}/D2,    inner e^{-G},     G = U + ln D2
    # They must agree to rounding; if not, the bins or limits are wrong.
    # C uses the histogram -ln P for G: a physics check of -ln P = U_eff + ln D2.
    function mfpt(xs, outer, inner, i0, ia)
        cum = zeros(length(xs))
        for i in 2:ia; cum[i] = cum[i-1] + 0.5*(inner[i]+inner[i-1])*(xs[i]-xs[i-1]); end
        g = outer .* cum
        sum(0.5*(g[i]+g[i-1])*(xs[i]-xs[i-1]) for i in i0+1:ia)
    end
    G = U .+ log.(D2)
    ways = ((exp.(U), exp.(-U)./D2), (exp.(G)./D2, exp.(-G)), (exp.(mP)./D2, exp.(-mP)))
    ia_f = findfirst(>=(1.1thr), x)                 # folded escape: absorb at 1.1thr
    Tf = [mfpt(x, o, w, li, ia_f) for (o, w) in ways]
    rv(v) = reverse(v)                              # open escape: mirror so it runs left→right
    xr = -rv(x); n_ = length(x)
    ia_o = n_ + 1 - findlast(<=(0.9thr), x)         # absorb at 0.9thr
    To = [mfpt(xr, rv(o), rv(w), n_ + 1 - ri, ia_o) for (o, w) in ways]
    for T in (Tf, To)
        abs(T[1] - T[2]) <= 1e-9 * abs(T[1]) ||
            error(@sprintf("exact MFPT disagrees between U_eff and G forms: %.6g vs %.6g", T[1], T[2]))
    end
    (barU=(folded=U[bt]-U[li], open=U[bt]-U[ri]),
     barP=(folded=mP[bt]-mP[li], open=mP[bt]-mP[ri]),
     kpred=(folded=pre(cw_f,li)*exp(-(U[bt]-U[li])), open=pre(cw_o,ri)*exp(-(U[bt]-U[ri]))),
     kpred_top=(folded=pre(cw_f,bt)*exp(-(U[bt]-U[li])), open=pre(cw_o,bt)*exp(-(U[bt]-U[ri]))),
     kexact=(folded=1/Tf[1], open=1/To[1]),
     kexact_lnp=(folded=1/Tf[3], open=1/To[3]),
     xtop=x[bt], nbins=length(x))
end

function build_rows(nbin, minb)
    rows = NamedTuple[]
    for (key, c) in cells
        fol, κ, f0 = key
        b = barriers(c, thr_global, nbin, minb)
        occ = c.exposure[:folded] / (c.exposure[:folded] + c.exposure[:open])
        for st in (:folded, :open)
            k  = c.events[st] / max(c.exposure[st], eps())
            cv = length(c.durs[st]) >= 3 ? std(c.durs[st])/mean(c.durs[st]) : NaN
            push!(rows, (folder=fol, kappa=κ, f0=f0, phi=f0/κ, state=st, nrun=c.nrun,
                         events=c.events[st], exposure=c.exposure[st], rate=k,
                         mean_dwell=isempty(c.durs[st]) ? NaN : mean(c.durs[st]),
                         cv=cv, occ_folded=occ,
                         bar_ueff = b === nothing ? NaN : getfield(b.barU, st),
                         bar_mlnp = b === nothing ? NaN : getfield(b.barP, st),
                         k_pred   = b === nothing ? NaN : getfield(b.kpred, st),
                         k_pred_top = b === nothing ? NaN : getfield(b.kpred_top, st),
                         k_exact  = b === nothing ? NaN : getfield(b.kexact, st),
                         k_exact_lnp = b === nothing ? NaN : getfield(b.kexact_lnp, st),
                         nbins = b === nothing ? 0 : b.nbins))
        end
    end
    sort!(rows, by=r->(r.folder, r.f0, r.kappa, r.state))
    rows
end

function fitline(x, y)
    n = length(x); n < 3 && return (NaN, NaN, NaN)
    mx, my = mean(x), mean(y)
    s = sum((x .- mx) .* (y .- my)) / sum((x .- mx).^2); a = my - s*mx
    (a, s, 1 - sum((y .- (a .+ s .* x)).^2) / sum((y .- my).^2))
end
usable_of(rows) = [r for r in rows if r.events >= MIN_EVENTS && isfinite(r.bar_ueff)]

# ── sensitivity of the whole test to the binning ────────────────────────────
println("\nbinning sensitivity (the transition region is thinly visited):")
@printf("  %5s %5s %7s %22s %22s\n", "nbin", "minb", "usable", "U_eff slope (R²)", "−lnP slope (R²)")
for (nb_, mb_) in ((40,80),(36,60),(28,40),(24,30),(20,20))
    rs = build_rows(nb_, mb_); us = usable_of(rs)
    length(us) < 3 && (@printf("  %5d %5d %7d %22s %22s\n", nb_, mb_, length(us), "-", "-"); continue)
    lnk = [log(r.rate) for r in us]
    _, sU, rU = fitline([r.bar_ueff for r in us], lnk)
    _, sP, rP = fitline([r.bar_mlnp for r in us], lnk)
    @printf("  %5d %5d %7d %14.2f (%.2f) %14.2f (%.2f)\n", nb_, mb_, length(us), sU, rU, sP, rP)
end

rows = build_rows(NBIN, MINB)
usable = usable_of(rows)
open(joinpath(ROOT,"results","campaign","kramers_rates.csv"), "w") do io
    println(io, "folder,kappa,f0,phi,state,nrun,events,exposure_s,rate_per_s,rate_pred_per_s,rate_pred_top_per_s,rate_exact_per_s,rate_exact_lnp_per_s,mean_dwell_s,cv,occ_folded,bar_ueff,bar_mlnp,nbins")
    for r in rows
        @printf(io, "%s,%.4f,%.4f,%.4f,%s,%d,%d,%.2f,%.5g,%.5g,%.5g,%.5g,%.5g,%.4g,%.4g,%.4f,%.4g,%.4g,%d\n",
                r.folder, r.kappa, r.f0, r.phi, r.state, r.nrun, r.events, r.exposure,
                r.rate, r.k_pred, r.k_pred_top, r.k_exact, r.k_exact_lnp,
                r.mean_dwell, r.cv, r.occ_folded, r.bar_ueff, r.bar_mlnp, r.nbins)
    end
end

@printf("\nat nbin=%d minb=%d: %d of %d (cell, direction) pairs usable\n", NBIN, MINB, length(usable), length(rows))
@printf("%7s %7s %8s %7s %9s %10s %10s %10s %9s %6s %9s %9s\n",
        "kappa","f0","dir","events","rate/s","exact/s","D2well/s","D2top/s","dwell_s","CV","bar_Ueff","bar_-lnP")
for r in sort(usable, by=r->(r.f0,r.kappa))
    @printf("%7.3f %7.3f %8s %7d %9.4f %10.4f %10.4f %10.4f %9.2f %6.2f %9.2f %9.2f\n",
            r.kappa, r.f0, r.state == :folded ? "fold→op" : "open→fo",
            r.events, r.rate, r.k_exact, r.k_pred, r.k_pred_top, r.mean_dwell, r.cv, r.bar_ueff, r.bar_mlnp)
end
lnk = [log(r.rate) for r in usable]
aU, sU, rU = fitline([r.bar_ueff for r in usable], lnk)
aP, sP, rP = fitline([r.bar_mlnp for r in usable], lnk)
@printf("\nln k vs barrier (all):  U_eff slope %+.2f (R² %.2f)   −lnP slope %+.2f (R² %.2f)   [Kramers predicts −1]\n",
        sU, rU, sP, rP)

# the two directions must be fitted apart: they are different states with different
# kinetics, and the CV test below says only one of them is memoryless
println("\nby direction:")
@printf("  %-12s %5s %19s %19s\n", "direction", "n", "U_eff slope (R²)", "−lnP slope (R²)")
for (st, nm) in ((:folded,"folded→open"), (:open,"open→folded"))
    us = [r for r in usable if r.state == st]
    length(us) < 3 && continue
    lk = [log(r.rate) for r in us]
    _, s1, q1 = fitline([r.bar_ueff for r in us], lk)
    _, s2, q2 = fitline([r.bar_mlnp for r in us], lk)
    @printf("  %-12s %5d %11.2f (%.2f) %11.2f (%.2f)\n", nm, length(us), s1, q1, s2, q2)
end
println("\nmeasured / predicted rate, median [10–90% across cells]:")
for (st, nm) in ((:folded,"folded→open"), (:open,"open→folded"))
    us = [r for r in usable if r.state == st]
    length(us) < 3 && continue
    for (lab, get) in (("exact MFPT (U_eff)", r->r.k_exact), ("exact MFPT (hist −ln P)", r->r.k_exact_lnp),
                       ("Kramers, D2(well)", r->r.k_pred), ("Kramers, D2(top) [old]", r->r.k_pred_top))
        ratios = [r.rate/get(r) for r in us if isfinite(get(r)) && get(r) > 0]
        isempty(ratios) && continue
        @printf("  %-12s %-26s %6.2f  [%.2f – %.2f]  n=%d\n", nm, lab, median(ratios),
                quantile(ratios,0.1), quantile(ratios,0.9), length(ratios))
    end
end
for st in (:folded, :open)
    cvs = [r.cv for r in usable if r.state==st && isfinite(r.cv)]
    isempty(cvs) && continue
    @printf("CV of dwells, %-6s escape: median %.2f  (n=%d cells)   [1.0 = exponential]\n",
            st == :folded ? "folded" : "open", median(cvs), length(cvs))
end

# ── figure ──────────────────────────────────────────────────────────────────
wc = [r for r in rows if r.folder == "Results_N17_wellcuts"]
cutF = sort(unique(r.f0 for r in wc if count(x->x.f0==r.f0, wc) >= 18))
cutK = sort(unique(r.kappa for r in wc if count(x->x.kappa==r.kappa, wc) >= 18))
fig = Figure(size=(1800,1060))
Label(fig[0,1:3], "Route 3 — switching rates, and the barrier that predicts them";
      fontsize=17, font=:bold, tellwidth=false)

function ratepanel(pos, title, xlabel, lab, cuts, getx, match)
    ax = Axis(fig[pos...], xlabel=xlabel, ylabel="escape rate  [1/s]", yscale=log10,
              xscale=log10, title=title)
    cg = cgrad(:viridis, max(2,length(cuts)); categorical=true)
    for (i,cv) in enumerate(cuts), (st,ls,mk) in ((:folded,:solid,:circle),(:open,:dash,:utriangle))
        pts = sort([r for r in wc if match(r,cv) && r.state==st && r.events >= MIN_EVENTS], by=getx)
        length(pts) < 2 && continue
        lines!(ax, getx.(pts), [r.rate for r in pts]; color=cg[i], linestyle=ls, linewidth=2,
               label = st == :folded ? @sprintf("%s = %.2f", lab, cv) : nothing)
        scatter!(ax, getx.(pts), [r.rate for r in pts]; color=cg[i], marker=mk, markersize=10)
    end
    axislegend(ax; position=:lt, framevisible=false, labelsize=9)
    ax
end
ratepanel((1,1), "A   fixed f₀, κ swept   (solid: folded→open · dashed: open→folded)",
          "κ", "f₀", cutF, r->r.kappa, (r,v)->r.f0==v)
ratepanel((1,2), "B   fixed κ, f₀ swept   (solid: folded→open · dashed: open→folded)",
          "f₀", "κ", cutK, r->r.f0, (r,v)->r.kappa==v)

# C — does the single-cell result (−ln P overstates the barrier) hold across cells?
let ax = Axis(fig[1,3], xlabel="barrier from U_eff", ylabel="barrier from −ln P",
              title="C   −ln P vs the kinetic barrier, 29 cells")
    for (st,col,mk,nm) in ((:folded,:crimson,:circle,"folded→open"),(:open,:seagreen,:utriangle,"open→folded"))
        pts = [r for r in usable if r.state==st]
        isempty(pts) && continue
        scatter!(ax, [r.bar_ueff for r in pts], [r.bar_mlnp for r in pts]; color=col,
                 marker=mk, markersize=10,
                 label=@sprintf("%s (median ratio %.2f)", nm, median([r.bar_mlnp/r.bar_ueff for r in pts])))
    end
    lims = [0, maximum(max(r.bar_ueff, r.bar_mlnp) for r in usable)*1.05]
    lines!(ax, lims, lims; color=(:black,0.5), linestyle=:dash)
    text!(ax, lims[2]*0.62, lims[2]*0.66; text="equal", fontsize=10, color=(:black,0.6))
    axislegend(ax; position=:lt, framevisible=false, labelsize=10)
end

ax3 = Axis(fig[2,1], xlabel="barrier  [effective-noise units]", ylabel="ln (escape rate / s⁻¹)",
           title="D   Kramers test: ln k vs barrier, slope should be −1")
for (nm, get, col, aa, sl, r2) in (("U_eff (Kramers–Moyal)", r->r.bar_ueff, :darkorange, aU, sU, rU),
                                   ("−ln P", r->r.bar_mlnp, :steelblue, aP, sP, rP))
    xs = [get(r) for r in usable]
    scatter!(ax3, xs, lnk; color=col, markersize=9,
             label=@sprintf("%s: slope %+.2f, R² %.2f", nm, sl, r2))
    xr = [minimum(xs), maximum(xs)]
    lines!(ax3, xr, aa .+ sl .* xr; color=col, linewidth=2)
end
let xs = [r.bar_ueff for r in usable]
    xr = [minimum(xs), maximum(xs)]
    lines!(ax3, xr, mean(lnk) + mean(xs) .- xr; color=(:black,0.45), linestyle=:dot,
           linewidth=2, label="slope −1 (Kramers)")
end
axislegend(ax3; position=:lb, framevisible=false, labelsize=10)

# E — the test with no fitted parameter: Kramers rate from U_eff, D₂ and curvatures
let ax = Axis(fig[2,2], xlabel="predicted rate  [1/s]", ylabel="measured rate  [1/s]",
              xscale=log10, yscale=log10,
              title="E   Predicted rate, no free parameter   (filled: exact MFPT · hollow: Kramers, D₂ at well)")
    pk = [r for r in usable if isfinite(r.k_pred) && r.k_pred > 0 && isfinite(r.k_exact) && r.k_exact > 0]
    for (st,col,mk,nm) in ((:folded,:crimson,:circle,"folded→open"),(:open,:seagreen,:utriangle,"open→folded"))
        pts = [r for r in pk if r.state==st]
        isempty(pts) && continue
        scatter!(ax, [r.k_exact for r in pts], [r.rate for r in pts]; color=col, marker=mk,
                 markersize=10,
                 label=@sprintf("%s exact (median %.2f×)", nm, median([r.rate/r.k_exact for r in pts])))
        scatter!(ax, [r.k_pred for r in pts], [r.rate for r in pts]; color=:transparent, marker=mk,
                 markersize=10, strokecolor=col, strokewidth=1.5,
                 label=@sprintf("%s Kramers (median %.2f×)", nm, median([r.rate/r.k_pred for r in pts])))
    end
    lo = minimum(min(r.k_pred, r.k_exact, r.rate) for r in pk)*0.6
    hi = maximum(max(r.k_pred, r.k_exact, r.rate) for r in pk)*1.6
    lines!(ax, [lo,hi], [lo,hi]; color=(:black,0.55), linestyle=:dash)
    band!(ax, [lo,hi], [lo/3,hi/3], [lo*3,hi*3]; color=(:black,0.07))
    text!(ax, hi*0.35, hi*0.75; text="exact", fontsize=10, color=(:black,0.6))
    axislegend(ax; position=:lt, framevisible=false, labelsize=10)
end

ax4 = Axis(fig[2,3], xlabel="escapes observed", ylabel="CV of dwell times", xscale=log10,
           title="F   Is escape memoryless?   CV = 1 means exponential")
for (st,col,mk,nm) in ((:folded,:crimson,:circle,"folded→open"),(:open,:seagreen,:utriangle,"open→folded"))
    pts = [r for r in usable if r.state==st && isfinite(r.cv)]
    isempty(pts) && continue
    scatter!(ax4, [r.events for r in pts], [r.cv for r in pts]; color=col, marker=mk,
             markersize=10, label=@sprintf("%s (median %.2f)", nm, median([r.cv for r in pts])))
end
hlines!(ax4, [1.0]; color=(:black,0.5), linestyle=:dash)
ylims!(ax4, 0, 2.2)
axislegend(ax4; position=:rt, framevisible=false, labelsize=10)
Label(fig[3,1:3],
  @sprintf("%s · %d cells · 300 s runs · rate = escapes / time in state (censored MLE) · barriers from the same Rg series · %d bins, ≥%d pairs/bin",
           join(folders, " + "), length(cells), NBIN, MINB); fontsize=11, tellwidth=false)

base = joinpath(ROOT,"results","campaign","kramers_rates")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("\n✓ ", relpath(base,ROOT), ".png / .pdf  and  kramers_rates.csv")
