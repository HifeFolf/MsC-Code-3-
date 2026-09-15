# phase_diagram_regimes.jl
# A phase diagram built from TIME-RESOLVED behaviour rather than the final frame.
#
# Why this supersedes the P(fold) map:
#   In the switching region the ring hops between states ergodically — at one cell
#   (κ=1, Φ=2.21) all ten seeds visit BOTH states and spend 21-63% of their time
#   folded, yet the state at any chosen instant is a coin flip.  So "P(fold)" read
#   off the final frame is not a basin probability; it is a one-sample estimate of
#   OCCUPANCY.  Measuring occupancy directly is both the right quantity and far
#   less noisy.
#
#   It also separates three regimes the final-frame map renders identically:
#     QUIESCENT  never folds within the window
#     BISTABLE   folds AND unfolds — visits both states reversibly
#     ABSORBING  folds and never returns
#
# Per run, Rg is tracked against the fold threshold with a hysteresis band
# (0.9/1.1×thr, min dwell MIN_DWELL_S) so fluctuation chatter is not counted.
#
# ⚠️ The bistable region measured this way is a LOWER BOUND: mean dwell times on
# the locus are 15-75 s (from Results_N17_bistable_long), so a 30 s window misses
# slow switchers, which are then misread as ABSORBING.  Stated on the figure.
#
# Usage:  julia --project scripts/phase_diagram_regimes.jl [folder]

using CairoMakie, Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_phi_long"
const RG_FRAC = 0.70
const MIN_DWELL_S = 0.5
const BURN_S = 2.0

dir = joinpath(ROOT, "results", folder)
runs = sort(filter(d -> occursin(r"^run_\d+$", d), readdir(dir)))
@printf("scanning %d runs from %s…\n", length(runs), folder)

cells = Dict{Tuple{Float64,Float64}, Vector{NamedTuple}}()
for (i, rn) in enumerate(runs)
    rp = joinpath(dir, rn)
    obs = joinpath(rp,"observables.jld2"); par = joinpath(rp,"params.toml")
    (isfile(obs) && isfile(par)) || continue
    ep = TOML.parsefile(par)["params"]
    N = Int(ep["N"]); κ = Float64(ep["kappa"]); f0 = Float64(ep["f0"])
    thr = RG_FRAC*(N/(2π))
    d = jldopen(obs,"r") do f
        haskey(f,"Rg") || return nothing
        (t = haskey(f,"times") ? f["times"] : collect(1:length(f["Rg"])), Rg = f["Rg"])
    end
    d === nothing && continue
    i0 = something(findfirst(≥(BURN_S), d.t), 1)
    t = @view d.t[i0:end]; Rg = @view d.Rg[i0:end]
    length(t) < 20 && continue
    dt = (t[end]-t[1])/(length(t)-1); need = max(2, round(Int, MIN_DWELL_S/dt))
    occ = count(<(thr), Rg)/length(Rg)
    st = Rg[1] < 0.9thr ? -1 : 1; downs = ups = 0; rl = 0; pend = 0
    for x in Rg
        s = x < 0.9thr ? -1 : (x > 1.1thr ? 1 : 0)
        (s == 0 || s == st) && (rl = 0; pend = 0; continue)
        (s == pend) ? (rl += 1) : (pend = s; rl = 1)
        if rl >= need
            s == -1 ? (downs += 1) : (ups += 1); st = s; rl = 0; pend = 0
        end
    end
    push!(get!(cells, (κ, f0), NamedTuple[]),
          (occ = occ, downs = downs, ups = ups, span = t[end]-t[1]))
    i % 2000 == 0 && @printf("  %d / %d\n", i, length(runs))
end

# ── per-cell regime ──────────────────────────────────────────────────────────
# 1 QUIESCENT  2 BISTABLE  3 ABSORBING ; regime = the majority behaviour
rows = NamedTuple[]
for (k, v) in cells
    n = length(v)
    # A run already FOLDED when the window opens records zero transitions.  Testing
    # only `downs==0` would call that "never folds" — the median collapse here is
    # 0.12 s, so most runs fold before the burn-in ends.  Classify on OCCUPANCY.
    isbi(r) = r.downs > 0 && r.ups > 0
    f_bi   = count(isbi, v)/n
    f_abs  = count(r -> !isbi(r) && r.occ >  0.5, v)/n
    f_qui  = count(r -> !isbi(r) && r.occ <= 0.5, v)/n
    fr = (f_qui, f_bi, f_abs)
    reg = argmax(fr)
    rate = sum(r.downs + r.ups for r in v) / sum(r.span for r in v)   # per second
    push!(rows, (κ = k[1], f0 = k[2], Φ = k[2]/k[1], n = n,
                 occ = mean(r.occ for r in v), f_bi = f_bi, f_abs = f_abs,
                 f_qui = f_qui, reg = reg, conf = maximum(fr), rate = rate))
end
sort!(rows, by = r -> (r.κ, r.f0))
open(joinpath(dir, "regime_map.csv"), "w") do io
    println(io, "kappa,f0,phi,n,occupancy,f_quiescent,f_bistable,f_absorbing,regime,confidence,rate_per_s")
    for r in rows
        @printf(io, "%.8g,%.8g,%.4f,%d,%.4f,%.3f,%.3f,%.3f,%d,%.3f,%.5f\n",
                r.κ, r.f0, r.Φ, r.n, r.occ, r.f_qui, r.f_bi, r.f_abs, r.reg, r.conf, r.rate)
    end
end
RN = ["Quiescent (never folds)", "Bistable (folds AND unfolds)", "Absorbing (folds, stays)"]
for c in 1:3
    @printf("  %-30s %5d cells\n", RN[c], count(r -> r.reg == c, rows))
end

# ── figure ───────────────────────────────────────────────────────────────────
κs = sort(unique(r.κ for r in rows)); fs = sort(unique(r.f0 for r in rows))
ki = Dict(v=>i for (i,v) in enumerate(κs)); fi = Dict(v=>i for (i,v) in enumerate(fs))
logκ, logf = log10.(κs), log10.(fs)
tk(v) = (range(minimum(v),maximum(v),length=5),
         [@sprintf("%.3g",10^x) for x in range(minimum(v),maximum(v),length=5)])
xr = (minimum(logf), maximum(logf)); yr = (minimum(logκ), maximum(logκ))

OCC = fill(NaN, length(fs), length(κs)); RATE = fill(NaN, length(fs), length(κs))
IMG = fill(RGBAf(0,0,0,0), length(fs), length(κs))
HUE = [(0.27,0.47,0.72), (0.80,0.13,0.44), (0.95,0.55,0.10)]   # quiescent / bistable / absorbing
GREY = (0.85,0.85,0.85)
for r in rows
    j, i = fi[r.f0], ki[r.κ]
    OCC[j,i] = r.occ; RATE[j,i] = r.rate
    c = clamp((r.conf - 0.4)/0.6, 0.0, 1.0)
    h = HUE[r.reg]
    IMG[j,i] = RGBAf(c*h[1]+(1-c)*GREY[1], c*h[2]+(1-c)*GREY[2], c*h[3]+(1-c)*GREY[3], 1)
end

fig = Figure(size = (1800, 560))
ax1 = Axis(fig[1,1], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="A  Occupancy — fraction of TIME folded", xticks=tk(logf), yticks=tk(logκ))
hm = heatmap!(ax1, logf, logκ, OCC;
              colormap=cgrad([RGBAf(HUE[1]...,1), RGBAf(GREY...,1), RGBAf(HUE[3]...,1)]),
              colorrange=(0,1))
Colorbar(fig[1,2], hm, label="fraction of time folded", width=13)

ax2 = Axis(fig[1,3], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="B  Regime (saturation = seed agreement)", xticks=tk(logf), yticks=tk(logκ))
image!(ax2, xr, yr, IMG; interpolate=false); limits!(ax2, xr..., yr...)
Legend(fig[2,3], [PolyElement(color=RGBAf(HUE[c]...,1)) for c in 1:3], RN,
       orientation=:horizontal, framevisible=false, labelsize=10)

ax3 = Axis(fig[1,4], xlabel="Active force  f₀", ylabel="Bending stiffness  κ",
           title="C  Switching rate (transitions per 100 s)", xticks=tk(logf), yticks=tk(logκ))
hm3 = heatmap!(ax3, logf, logκ, RATE .* 100; colormap=:magma, colorrange=(0,10))
Colorbar(fig[1,5], hm3, label="transitions / 100 s", width=13)
Label(fig[2,1:2], "window = $(round(Int, median(r.span for v in values(cells) for r in v))) s · " *
      "bistable region is a LOWER BOUND (dwell times 15-75 s exceed the window)",
      fontsize=10, tellwidth=false)

out = joinpath(dir, "phase_diagram_regimes.png")
save(out, fig)
println("✓ figure → $(relpath(out, ROOT))")
