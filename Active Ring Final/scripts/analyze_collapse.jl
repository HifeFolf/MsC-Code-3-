# analyze_collapse.jl
# Is kBT/κ the second control parameter?
#
# The κ-sweep at fixed kBT=0.1 measured Φ₅₀ ∝ κ^0.385.  Two readings are possible
# and are perfectly confounded when kBT is held fixed (kBT/κ ∝ 1/κ):
#   (a) a genuine κ dependence, temperature irrelevant;
#   (b) a thermally activated transition controlled by τ = kBT/κ.
#
# This decides between them by holding τ FIXED and varying κ.  The two readings
# make quantitatively different predictions for how much Φ₅₀ may vary across κ
# at fixed τ, over a κ span of R:
#   (a) κ-only     → Φ₅₀ varies by R^0.385   (R=4  ⇒  1.72×)
#   (b) τ-collapse → Φ₅₀ varies by 1.00×
# so the measured ratio lands on one or the other.
#
# Usage:  julia --project scripts/analyze_collapse.jl [folder] [features.csv]

using Printf, Statistics

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_collapse"
csv    = length(ARGS) >= 2 ? ARGS[2] : "phase_features.csv"
const KAPPA_EXPONENT = 0.385      # measured at fixed kBT=0.1 on Results_N17_phi_long

# Mirrors clf_route in phase_reliability.jl exactly.
const BI_FRAC = 0.15; const INWARD_DEV = -8.0
const RG_CURLED_FRAC = 0.70; const B_SLITHER = 0.04
function route(f_in, f_out, dev, Rg, bstd, N)
    min(f_in, f_out) > BI_FRAC && return 1
    (dev < INWARD_DEV || Rg < RG_CURLED_FRAC * (N / (2π))) && return 2
    return (isnan(bstd) ? 0.0 : bstd) > B_SLITHER ? 1 : 3
end

path = joinpath(ROOT, "results", folder, csv)
isfile(path) || error("Missing $path")
lines = readlines(path); hdr = String.(split(lines[1], ","))
ix(n) = findfirst(==(n), hdr)
num(v, i) = (x = tryparse(Float64, v[i]); x === nothing ? NaN : x)
ix("kbt") === nothing && error("No kbt column — rebuild the cache with --force")

seen = Set{NTuple{4,Float64}}(); recs = NamedTuple[]
for ln in lines[2:end]
    isempty(strip(ln)) && continue
    v = split(ln, ","); length(v) < length(hdr) && continue
    key = (num(v,ix("kappa")), num(v,ix("f0")), num(v,ix("kbt")), num(v,ix("seed")))
    key in seen && continue; push!(seen, key)
    push!(recs, (κ=key[1], f=key[2], kbt=key[3],
                 lab=route(num(v,ix("f_in")), num(v,ix("f_out")), num(v,ix("dev")),
                           num(v,ix("Rg_med")), num(v,ix("bstd")), num(v,ix("N"))),
                 tcol=num(v,ix("t_col")), tend=num(v,ix("t_end"))))
end
@printf("%s: %d runs\n", folder, length(recs))

function fit(pts)
    best = (Inf, NaN, NaN)
    for mu in -1.0:0.005:2.0, s in 0.01:0.005:0.80
        ll = 0.0
        for (x, k, n) in pts
            p = clamp(1/(1+exp(-(x-mu)/s)), 1e-9, 1-1e-9)
            ll -= k*log(p) + (n-k)*log(1-p)
        end
        ll < best[1] && (best = (ll, mu, s))
    end
    best
end

# group by (kappa, kBT)
groups = Dict{Tuple{Float64,Float64}, Vector{NamedTuple}}()
for r in recs; push!(get!(groups, (r.κ, r.kbt), NamedTuple[]), r); end
taus = sort(unique(round(k[2]/k[1], sigdigits=4) for k in keys(groups)), rev=true)

results = Dict{Float64, Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}}()
const SLOPES = Float64[]
for τ in taus
    rows = Tuple{Float64,Float64,Float64,Float64,Float64}[]
    for (key, sub) in sort(collect(groups), by = x -> x[1])
        κ, kb = key
        round(kb/κ, sigdigits=4) == τ || continue
        cells = Dict{Float64,Vector{Int}}()
        for r in sub; push!(get!(cells, r.f/r.κ, Int[]), r.lab); end
        pts = [(log10(Φ), count(==(2), v), length(v)) for (Φ, v) in cells]
        ps = [k/n for (x,k,n) in pts]
        (minimum(ps) > 0.25 || maximum(ps) < 0.75) && continue    # unbracketed
        (ll, mu, s) = fit(pts)
        fold = [r for r in sub if r.lab == 2 && !isnan(r.tcol)]
        late = isempty(fold) ? 0.0 : 100count(r -> r.tcol > r.tend - 5, fold)/length(fold)
        push!(rows, (κ, kb, 10^mu, s, late))
    end
    !isempty(rows) && (results[τ] = rows)
end

for τ in sort(collect(keys(results)), rev=true)
    rows = results[τ]
    @printf("\n=== tau = kBT/kappa = %.4g ===\n", τ)
    @printf("  %7s %9s %9s %8s %10s\n", "kappa", "kBT", "Phi_50", "width", "late folds")
    for (κ, kb, Φ, s, late) in rows
        @printf("  %7.2f %9.4g %9.2f %8.3f %9.1f%%\n", κ, kb, Φ, s, late)
    end
    # SIGNED exponent, not just spread magnitude.  The first version compared
    # |spread| only and called a BACKWARDS-running dependence "kappa-only":
    # at fixed kBT, Phi_50 RISES with kappa; at fixed tau it FALLS.  Sign matters.
    if length(rows) >= 3
        Φv = [r[3] for r in rows]; κv = [r[1] for r in rows]
        x = log10.(κv); y = log10.(Φv)
        b = cov(x, y) / var(x)
        @printf("  d(log Phi_50)/d(log kappa) at fixed tau = %+.3f\n", b)
        @printf("     tau-collapse predicts %+.3f  ·  kappa-only predicts %+.3f\n",
                0.0, KAPPA_EXPONENT)
        push!(SLOPES, b)
    end
end

# Overall verdict
if !isempty(SLOPES)
    m = sum(SLOPES)/length(SLOPES)
    @printf("\nMean d(log Phi_50)/d(log kappa) at fixed tau, over %d tau levels : %+.3f\n",
            length(SLOPES), m)
    @printf("  tau-collapse (kBT/kappa) predicts  0.000\n")
    @printf("  kappa-only               predicts %+.3f\n", KAPPA_EXPONENT)
    if abs(m) < 0.10
        println("  => VERDICT: consistent with the tau = kBT/kappa COLLAPSE.")
    elseif m > 0
        println("  => VERDICT: same sign as kappa-only; the collapse does NOT hold.")
    else
        @printf("  => VERDICT: NEGATIVE — opposite sign to kappa-only. Neither pure hypothesis\n")
        @printf("     fits; the sign flip brackets an intermediate variable kBT/kappa^c with\n")
        @printf("     c = %.2f  (from a + b*c = 0 given a = %.3f at fixed kBT).\n",
                KAPPA_EXPONENT/(KAPPA_EXPONENT - m), KAPPA_EXPONENT)
    end
end
