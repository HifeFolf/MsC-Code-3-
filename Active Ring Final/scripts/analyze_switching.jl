# analyze_switching.jl
# Is the folded (crescent) state ABSORBING, or genuinely bistable?
#
# This decides which word a paper may use.  "Multistability"/"bistability" claims
# that two states coexist and the system can occupy either — ideally with
# transitions BOTH ways.  If instead every fold is permanent, what the seed-split
# cells show is stochastic BASIN SELECTION into one absorbing state, with the
# glider merely long-lived.  Both are publishable; they are not the same claim.
#
# Method: per run, track Rg against the fold threshold with a hysteresis band
# (down at 0.9*thr, up at 1.1*thr) so fluctuation chatter near the cut is not
# counted as a transition.  Then count sustained crossings in each direction.
#
# Usage:  julia --project scripts/analyze_switching.jl [folder] [pmin] [pmax]

using Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_phi_long"
pmin   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.2
pmax   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.8
const RG_CURLED_FRAC = 0.70
const MIN_DWELL_S    = 0.5     # a state must hold this long to count as entered

rd(p) = [split(l, ",") for l in readlines(p)[2:end] if !isempty(strip(l))]
lab = rd(joinpath(ROOT, "results", folder, "phase_runlabels_route.csv"))
cells = Dict{Tuple{Float64,Float64}, Vector{Tuple{String,Int}}}()
for r in lab
    k = (parse(Float64, r[4]), parse(Float64, r[5]))
    push!(get!(cells, k, Tuple{String,Int}[]), (String(r[2]), parse(Int, r[6])))
end
# ambiguous band = cells whose seeds genuinely split
band = [k for (k, v) in cells
        if (p = count(x -> x[2] == 2, v)/length(v); pmin <= p <= pmax)]
@printf("%s: %d cells with P(fold) in [%.1f, %.1f] — %d runs\n",
        folder, length(band), pmin, pmax, sum(length(cells[k]) for k in band))

tot = folded = ever_up = multi = 0
updurations = Float64[]
for k in band, (run, _) in cells[k]
    rp = joinpath(ROOT, "results", folder, run)
    obs = joinpath(rp, "observables.jld2"); par = joinpath(rp, "params.toml")
    (isfile(obs) && isfile(par)) || continue
    N = Int(TOML.parsefile(par)["params"]["N"])
    thr = RG_CURLED_FRAC * (N / (2π))
    d = jldopen(obs, "r") do f
        haskey(f, "Rg") || return nothing
        (t = haskey(f, "times") ? f["times"] : collect(1:length(f["Rg"])), Rg = f["Rg"])
    end
    d === nothing && continue
    global tot += 1
    dt = length(d.t) > 1 ? (d.t[end] - d.t[1])/(length(d.t)-1) : 1.0
    need = max(2, round(Int, MIN_DWELL_S/dt))
    # hysteretic state track: -1 folded, +1 open
    state = d.Rg[1] < 0.9thr ? -1 : 1
    downs = ups = 0; run_len = 0; pend = 0
    for x in d.Rg
        s = x < 0.9thr ? -1 : (x > 1.1thr ? 1 : 0)
        (s == 0 || s == state) && (run_len = 0; pend = 0; continue)
        (s == pend) ? (run_len += 1) : (pend = s; run_len = 1)
        if run_len >= need
            s == -1 ? (downs += 1) : (ups += 1)
            state = s; run_len = 0; pend = 0
        end
    end
    downs > 0 && (global folded += 1)
    if downs > 0 && ups > 0
        global ever_up += 1
        (downs + ups) >= 3 && (global multi += 1)
    end
end
@printf("\n  runs examined                       : %d\n", tot)
@printf("  runs that folded at least once      : %d\n", folded)
@printf("  of those, that later UNFOLDED       : %d  (%.2f%%)\n",
        ever_up, folded == 0 ? 0.0 : 100ever_up/folded)
@printf("  runs with >=3 sustained transitions  : %d\n", multi)
println(ever_up == 0 ?
  "\n  => The folded state is ABSORBING in this dataset. No run escaped it.\n" *
  "     The seed split is stochastic BASIN SELECTION, not two-way bistability." :
  "\n  => Some runs DO escape the folded state — two-way transitions exist.")
