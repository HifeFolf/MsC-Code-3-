# analyze_dwell.jl
# Dwell-time statistics and an effective double-well, for the bistable locus.
#
# What this decides, and why it matters for a paper:
#   Repeated Rg-threshold crossings alone do NOT prove two-state switching — a
#   chaotically wandering ring can cross a cut repeatedly without occupying two
#   states.  The discriminator is the DWELL-TIME distribution:
#     • exponential (CV ≈ 1) ⇒ memoryless escape from a well = Kramers, and the
#       rate gives a barrier;
#     • strongly non-exponential (CV ≪ 1 or ≫ 1) ⇒ not simple two-state kinetics.
#
# Censoring: the first and last dwell of every run are truncated by the recording
# window, so they are EXCLUDED from the distribution (they would bias the mean
# downward).  They are still counted in total occupancy for rate estimates, which
# is the standard treatment.
#
# Outputs (in the campaign folder):
#   dwell_times.csv          one row per COMPLETE dwell
#   dwell_summary.csv        per-Phi-bin statistics
#   dwell_analysis.png       effective potential + survival curves
#
# Usage:  julia --project scripts/analyze_dwell.jl [folder]

using CairoMakie, Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_bistable_long"
const RG_CURLED_FRAC = 0.70
const MIN_DWELL_S    = 0.5      # debounce: a state must hold this long to count
const BURN_S         = 5.0      # ignore the initial relaxation

dir = joinpath(ROOT, "results", folder)
isdir(dir) || error("Missing $dir")
runs = sort(filter(d -> occursin(r"^run_\d+$", d), readdir(dir)))
isempty(runs) && error("No runs in $dir")

# ── Pass 1: per-run state segmentation ───────────────────────────────────────
dwells  = NamedTuple[]                 # complete dwells only
occup   = Dict(:open => 0.0, :folded => 0.0)   # total time, for rates
ntrans  = Dict(:to_folded => 0, :to_open => 0)
rgpool  = Float64[]
println("Scanning $(length(runs)) runs…")
for (i, rn) in enumerate(runs)
    rp = joinpath(dir, rn)
    obs = joinpath(rp, "observables.jld2"); par = joinpath(rp, "params.toml")
    (isfile(obs) && isfile(par)) || continue
    ep = TOML.parsefile(par)["params"]
    N = Int(ep["N"]); κ = Float64(ep["kappa"]); f0 = Float64(ep["f0"])
    thr = RG_CURLED_FRAC * (N / (2π))
    d = jldopen(obs, "r") do f
        haskey(f, "Rg") || return nothing
        (t = haskey(f,"times") ? f["times"] : collect(1:length(f["Rg"])), Rg = f["Rg"])
    end
    d === nothing && continue
    i0 = something(findfirst(≥(BURN_S), d.t), 1)
    t = @view d.t[i0:end]; Rg = @view d.Rg[i0:end]
    length(t) < 20 && continue
    append!(rgpool, Rg)
    dt = (t[end]-t[1])/(length(t)-1)
    need = max(2, round(Int, MIN_DWELL_S/dt))

    state = Rg[1] < thr ? :folded : :open
    t_enter = t[1]; rl = 0; pend = :none
    segs = Tuple{Symbol,Float64,Float64}[]        # (state, t_in, t_out)
    for (j, x) in enumerate(Rg)
        s = x < 0.9thr ? :folded : (x > 1.1thr ? :open : :none)
        (s == :none || s == state) && (rl = 0; pend = :none; continue)
        (s == pend) ? (rl += 1) : (pend = s; rl = 1)
        if rl >= need
            push!(segs, (state, t_enter, t[j]))
            state == :open ? (ntrans[:to_folded] += 1) : (ntrans[:to_open] += 1)
            state = s; t_enter = t[j]; rl = 0; pend = :none
        end
    end
    push!(segs, (state, t_enter, t[end]))          # final (censored) segment
    for (si, (st, ta, tb)) in enumerate(segs)
        occup[st] += tb - ta
        # complete = not the first and not the last segment of the run
        (si == 1 || si == length(segs)) && continue
        push!(dwells, (κ = κ, f0 = f0, Φ = f0/κ, state = st, dur = tb - ta))
    end
    i % 100 == 0 && @printf("  %d / %d\n", i, length(runs))
end
@printf("\ncomplete dwells: %d  (open %d, folded %d)\n", length(dwells),
        count(d -> d.state == :open, dwells), count(d -> d.state == :folded, dwells))

open(joinpath(dir, "dwell_times.csv"), "w") do io
    println(io, "kappa,f0,phi,state,duration_s")
    for d in dwells
        @printf(io, "%.8g,%.8g,%.4f,%s,%.4f\n", d.κ, d.f0, d.Φ, d.state, d.dur)
    end
end

# ── Per-Phi-bin statistics ───────────────────────────────────────────────────
bins = [(2.0,2.6),(2.6,3.2),(3.2,4.0),(4.0,5.1)]
println("\nDwell statistics by Phi bin  (CV = std/mean; exponential => CV ~ 1)")
@printf("  %-10s %-8s %6s %9s %8s %10s\n","Phi","state","n","mean(s)","CV","rate(1/s)")
open(joinpath(dir, "dwell_summary.csv"), "w") do io
    println(io, "phi_lo,phi_hi,state,n,mean_s,cv,median_s")
    for (lo,hi) in bins, st in (:open, :folded)
        v = [d.dur for d in dwells if lo <= d.Φ < hi && d.state == st]
        length(v) < 5 && continue
        cv = std(v)/mean(v)
        @printf("  %-10s %-8s %6d %9.2f %8.2f %10.4f\n",
                "$lo-$hi", st, length(v), mean(v), cv, 1/mean(v))
        @printf(io, "%.2f,%.2f,%s,%d,%.4f,%.4f,%.4f\n", lo, hi, st, length(v), mean(v), cv, median(v))
    end
end
@printf("\ntotal occupancy: open %.0f s, folded %.0f s\n", occup[:open], occup[:folded])
@printf("transitions: open→folded %d, folded→open %d\n", ntrans[:to_folded], ntrans[:to_open])

# ── Figure ───────────────────────────────────────────────────────────────────
fig = Figure(size = (1200, 480))
# effective potential from the pooled Rg histogram
lo, hi = quantile(rgpool, 0.001), quantile(rgpool, 0.999)
edges = range(lo, hi, length = 90); ctr = (edges[1:end-1] .+ edges[2:end]) ./ 2
h = zeros(Int, length(ctr)); w = step(edges)
for x in rgpool
    (lo <= x <= hi) || continue
    k = clamp(Int(fld(x - lo, w)) + 1, 1, length(h)); h[k] += 1
end
p = h ./ sum(h); keep = p .> 0
ax1 = Axis(fig[1,1], xlabel = "Radius of gyration  Rg",
           ylabel = "−ln P(Rg)   [effective potential, arb.]",
           title = "Effective double well")
lines!(ax1, ctr[keep], -log.(p[keep]); linewidth = 2.5)
scatter!(ax1, ctr[keep], -log.(p[keep]); markersize = 4)

ax2 = Axis(fig[1,2], xlabel = "dwell time  (s)", ylabel = "survival  P(T > t)",
           yscale = log10, title = "Dwell-time survival — straight ⇒ exponential")
for (st, col) in ((:open, :steelblue), (:folded, :darkorange))
    v = sort([d.dur for d in dwells if d.state == st])
    length(v) < 5 && continue
    s = 1 .- (0:length(v)-1) ./ length(v)
    ok = s .> 0
    lines!(ax2, v[ok], s[ok]; color = col, linewidth = 2.5,
           label = "$st (n=$(length(v)), CV=$(round(std(v)/mean(v), digits=2)))")
end
axislegend(ax2; position = :rt, framevisible = false, labelsize = 10)

out = joinpath(dir, "dwell_analysis.png")
save(out, fig)
println("\n✓ figure → $(relpath(out, ROOT))")
