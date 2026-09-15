# analyze_kbt_transition.jl
# Does the folding transition move with temperature?
#
# Fits P(fold) vs log10 Φ with a binomial-ML logistic, SEPARATELY per kBT, and
# regresses the fitted midpoint on kBT.  This is the test that separates the two
# explanations for the measured Φ₅₀ ∝ κ^0.385 drift, which cannot be separated at
# fixed kBT because kBT/κ ∝ 1/κ is then perfectly collinear with κ:
#
#   slope ≈ 0  → temperature-independent; the κ drift is a genuine κ effect and
#                fl_origin.jl's constant-Fl_c derivation needs revisiting.
#   slope < 0  → thermally activated; the control parameter involves kBT/κ, which
#                would also explain the ~20× gap to linear_stability.jl's Fl_c≈75
#                (noise-assisted folding far below the deterministic threshold).
#
# A logistic fit is used rather than a threshold crossing: the "first Φ with
# P ≥ 0.5" estimator is biased low and outlier-sensitive, and previously produced
# a spurious "midpoint is pinned at Φ≈3" reading.
#
# Usage:  julia --project scripts/analyze_kbt_transition.jl [folder] [cutoff_s]

using Printf, Statistics

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_kbt_pilot"
csv    = length(ARGS) >= 2 ? ARGS[2] : "phase_features.csv"

# ── Classifier: mirrors clf_route in phase_reliability.jl exactly ────────────
const BI_FRAC = 0.15; const INWARD_DEV = -8.0
const RG_CURLED_FRAC = 0.70; const B_SLITHER = 0.04
function route(f_in, f_out, dev, Rg, bstd, N)
    min(f_in, f_out) > BI_FRAC && return 1
    (dev < INWARD_DEV || Rg < RG_CURLED_FRAC * (N / (2π))) && return 2
    return (isnan(bstd) ? 0.0 : bstd) > B_SLITHER ? 1 : 3
end

path = joinpath(ROOT, "results", folder, csv)
isfile(path) || error("Missing $path")
lines = readlines(path)
hdr = String.(split(lines[1], ","))
ix(n) = findfirst(==(n), hdr)
c = (κ=ix("kappa"), f=ix("f0"), b=ix("kbt"), s=ix("seed"), N=ix("N"),
     fi=ix("f_in"), fo=ix("f_out"), dv=ix("dev"), rg=ix("Rg_med"),
     bs=ix("bstd"), tc=ix("t_col"), te=ix("t_end"))
c.b === nothing && error("No kbt column — rebuild the cache with --force")

seen = Set{NTuple{4,Float64}}()
recs = NamedTuple[]
num(v, i) = (x = tryparse(Float64, v[i]); x === nothing ? NaN : x)
for ln in lines[2:end]
    isempty(strip(ln)) && continue
    v = split(ln, ",")
    length(v) < length(hdr) && continue
    key = (num(v,c.κ), num(v,c.f), num(v,c.b), num(v,c.s))
    key in seen && continue                      # drop the duplicated checkpoint run
    push!(seen, key)
    lab = route(num(v,c.fi), num(v,c.fo), num(v,c.dv), num(v,c.rg), num(v,c.bs), num(v,c.N))
    push!(recs, (κ=key[1], f=key[2], kbt=key[3], lab=lab,
                 tcol=num(v,c.tc), tend=num(v,c.te)))
end
@printf("%s: %d runs (%d duplicates dropped)\n\n", folder, length(recs), length(lines)-1-length(recs))

# ── Logistic P(fold) = 1/(1+exp(-(x-mu)/s)), x = log10 Phi, binomial ML ──────
function fit(pts)
    best = (Inf, NaN, NaN)
    for mu in -1.0:0.005:2.0, s in 0.01:0.005:0.80
        ll = 0.0
        for (x, k, n) in pts
            p = clamp(1 / (1 + exp(-(x - mu) / s)), 1e-9, 1 - 1e-9)
            ll -= k*log(p) + (n-k)*log(1-p)
        end
        ll < best[1] && (best = (ll, mu, s))
    end
    best
end

kbts = sort(unique(r.kbt for r in recs))
@printf("%9s %7s %10s %9s %13s | %10s %11s\n",
        "kBT","cells","Phi_50","width s","P range","med t_col","fold>25s")
fitted = Tuple{Float64,Float64}[]
for kb in kbts
    sub = [r for r in recs if r.kbt == kb]
    cells = Dict{Float64,Vector{Int}}()
    for r in sub; push!(get!(cells, r.f/r.κ, Int[]), r.lab); end
    pts = [(log10(Φ), count(==(2), v), length(v)) for (Φ, v) in cells]
    ps  = [k/n for (x,k,n) in pts]
    fold = [r for r in sub if r.lab == 2 && !isnan(r.tcol)]
    late = isempty(fold) ? NaN :
           100count(r -> r.tcol > r.tend - 5, fold) / length(fold)
    if minimum(ps) > 0.25 || maximum(ps) < 0.75
        @printf("%9.3f %7d %10s %9s %5.2f..%5.2f | %10s %10s\n",
                kb, length(pts), "unbracketed", "-", minimum(ps), maximum(ps),
                isempty(fold) ? "-" : @sprintf("%.2f s", median(r.tcol for r in fold)),
                isnan(late) ? "-" : @sprintf("%.1f%%", late))
        continue
    end
    (ll, mu, s) = fit(pts)
    push!(fitted, (kb, mu))
    @printf("%9.3f %7d %10.2f %9.3f %5.2f..%5.2f | %10.2f s %10.1f%%\n",
            kb, length(pts), 10^mu, s, minimum(ps), maximum(ps),
            median(r.tcol for r in fold), late)
end

# ── Summarise ────────────────────────────────────────────────────────────────
# A power-law fit is NOT assumed: the pilot shows Phi_50(kBT) is U-shaped, so a
# log-log slope averages the two arms to ~0 and would falsely read as
# "temperature independent".  Report the shape instead.
if length(fitted) >= 3
    kb = first.(fitted); mu = last.(fitted); Φ50 = 10 .^ mu
    lo = argmin(Φ50)
    @printf("\nPhi_50 range: %.2f (at kBT=%.3f) .. %.2f (at kBT=%.3f)  — spread %.1fx\n",
            minimum(Φ50), kb[lo], maximum(Φ50), kb[argmax(Φ50)], maximum(Φ50)/minimum(Φ50))
    mono = all(diff(Φ50) .> 0) || all(diff(Φ50) .< 0)
    if mono
        x = log10.(kb); b = cov(x, mu)/var(x)
        @printf("  monotonic — Phi_50 ~ kBT^%+.3f (r = %+.4f)\n", b, cor(x, mu))
    else
        @printf("  NON-MONOTONIC: minimum at kBT = %.3f. A single power law does not apply;\n", kb[lo])
        @printf("  do NOT summarise this with a log-log slope.\n")
    end
    println(maximum(Φ50)/minimum(Φ50) > 1.5 ?
        "  => transition is STRONGLY temperature dependent." :
        "  => transition is roughly temperature independent.")
end
