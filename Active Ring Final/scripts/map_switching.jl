# map_switching.jl
# Per-cell two-way switching statistics over a whole campaign, written to CSV so
# the multistability region can be mapped rather than asserted.
#
# For each run: hysteretic tracking of Rg against the fold threshold (down at
# 0.9*thr, up at 1.1*thr, each state must hold MIN_DWELL_S) gives the number of
# sustained fold and unfold events.  Per cell we then report:
#   p_fold      fraction of seeds ending folded
#   f_switch    fraction of seeds showing >=1 fold AND >=1 unfold
#   mean_trans  mean sustained transitions per run
#   f_multi     fraction with >=3 transitions (repeated switching, not one flip)
#
# Usage:  julia --project scripts/map_switching.jl [folder]

using Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_phi_long"
const RG_CURLED_FRAC = 0.70
const MIN_DWELL_S    = 0.5

rd(p) = [split(l, ",") for l in readlines(p)[2:end] if !isempty(strip(l))]
cells = Dict{Tuple{Float64,Float64}, Vector{String}}()
for r in rd(joinpath(ROOT, "results", folder, "phase_runlabels_route.csv"))
    push!(get!(cells, (parse(Float64, r[4]), parse(Float64, r[5])), String[]), String(r[2]))
end

function transitions(rp)
    obs = joinpath(rp, "observables.jld2"); par = joinpath(rp, "params.toml")
    (isfile(obs) && isfile(par)) || return nothing
    N = Int(TOML.parsefile(par)["params"]["N"]); thr = RG_CURLED_FRAC*(N/(2π))
    d = jldopen(obs, "r") do f
        haskey(f, "Rg") || return nothing
        (t = haskey(f,"times") ? f["times"] : collect(1:length(f["Rg"])), Rg = f["Rg"])
    end
    d === nothing && return nothing
    dt = length(d.t) > 1 ? (d.t[end]-d.t[1])/(length(d.t)-1) : 1.0
    need = max(2, round(Int, MIN_DWELL_S/dt))
    state = d.Rg[1] < 0.9thr ? -1 : 1
    downs = ups = 0; rl = 0; pend = 0
    for x in d.Rg
        s = x < 0.9thr ? -1 : (x > 1.1thr ? 1 : 0)
        (s == 0 || s == state) && (rl = 0; pend = 0; continue)
        (s == pend) ? (rl += 1) : (pend = s; rl = 1)
        if rl >= need
            s == -1 ? (downs += 1) : (ups += 1); state = s; rl = 0; pend = 0
        end
    end
    (downs = downs, ups = ups, ended_folded = d.Rg[end] < thr)
end

out = joinpath(ROOT, "results", folder, "switching_map.csv")
n_done = 0
open(out, "w") do io
    println(io, "kappa,f0,phi,n,p_fold,f_switch,mean_trans,f_multi")
    for (k, runs) in sort(collect(cells), by = x -> (x[1][1], x[1][2]))
        res = filter(!isnothing, [transitions(joinpath(ROOT,"results",folder,r)) for r in runs])
        isempty(res) && continue
        n = length(res)
        @printf(io, "%.8g,%.8g,%.4f,%d,%.3f,%.3f,%.3f,%.3f\n",
                k[1], k[2], k[2]/k[1], n,
                count(r -> r.ended_folded, res)/n,
                count(r -> r.downs > 0 && r.ups > 0, res)/n,
                mean(r.downs + r.ups for r in res),
                count(r -> (r.downs + r.ups) >= 3, res)/n)
        global n_done += 1
        n_done % 200 == 0 && @printf("  %d cells…\n", n_done)
    end
end
@printf("✓ %d cells → %s\n", n_done, relpath(out, ROOT))
