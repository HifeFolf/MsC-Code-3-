# plot_bistability.jl
# The evidence figure for two-state switching, in three panels.
#
#   A  raw Rg(t) traces that visibly hop between two levels — the most direct
#      evidence there is, and the one a reader believes without any analysis;
#   B  effective potential −ln P(Rg) computed PER Phi, showing the double well
#      TILT as the control parameter sweeps through the transition (the pooled
#      version in dwell_analysis.png mixes control values and hides this);
#   C  where switching actually occurs in (kappa, f0), from switching_map.csv.
#
# Usage:  julia --project scripts/plot_bistability.jl [long_folder] [map_folder]

using CairoMakie, Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
longf = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_bistable_long"
mapf  = length(ARGS) >= 2 ? ARGS[2] : "Results_N17_phi_long"
const RG_FRAC = 0.70
const MIN_DWELL_S = 0.5
const BURN_S = 5.0

dir = joinpath(ROOT, "results", longf)
runs = sort(filter(d -> occursin(r"^run_\d+$", d), readdir(dir)))

# ── scan: transitions per run, and pooled Rg per Phi ─────────────────────────
info = NamedTuple[]
rgby = Dict{Float64, Vector{Float64}}()
for rn in runs
    rp = joinpath(dir, rn)
    obs = joinpath(rp,"observables.jld2"); par = joinpath(rp,"params.toml")
    (isfile(obs) && isfile(par)) || continue
    ep = TOML.parsefile(par)["params"]
    N = Int(ep["N"]); κ = Float64(ep["kappa"]); f0 = Float64(ep["f0"]); Φ = f0/κ
    thr = RG_FRAC*(N/(2π))
    d = jldopen(obs,"r") do f
        haskey(f,"Rg") || return nothing
        (t = haskey(f,"times") ? f["times"] : collect(1:length(f["Rg"])), Rg = f["Rg"])
    end
    d === nothing && continue
    i0 = something(findfirst(≥(BURN_S), d.t), 1)
    t = d.t[i0:end]; Rg = d.Rg[i0:end]
    append!(get!(rgby, round(Φ, digits=2), Float64[]), Rg)
    dt = (t[end]-t[1])/(length(t)-1); need = max(2, round(Int, MIN_DWELL_S/dt))
    st = Rg[1] < thr ? -1 : 1; nt = 0; rl = 0; pend = 0
    for x in Rg
        s = x < 0.9thr ? -1 : (x > 1.1thr ? 1 : 0)
        (s == 0 || s == st) && (rl = 0; pend = 0; continue)
        (s == pend) ? (rl += 1) : (pend = s; rl = 1)
        rl >= need && (nt += 1; st = s; rl = 0; pend = 0)
    end
    push!(info, (run = rn, κ = κ, f0 = f0, Φ = Φ, thr = thr, nt = nt, t = t, Rg = Rg))
end
sort!(info, by = x -> -x.nt)
@printf("best traces: %s\n", join([@sprintf("%s(%d transitions, Phi=%.2f)", x.run, x.nt, x.Φ)
                                   for x in first(info,3)], ", "))

fig = Figure(size = (1500, 900))

# ── A: raw traces ────────────────────────────────────────────────────────────
ga = fig[1, 1:2] = GridLayout()
Label(ga[1, 1:1, Top()], "A   Raw trajectories — the ring hops between two states",
      fontsize = 15, font = :bold, padding = (0,0,8,0), halign = :left)
for (i, ex) in enumerate(first(info, 3))
    ax = Axis(ga[i, 1], ylabel = "Rg",
              xlabel = i == 3 ? "time  (s)" : "",
              xticklabelsvisible = i == 3)
    band!(ax, [0, ex.t[end]], fill(minimum(ex.Rg)-0.05, 2), fill(ex.thr, 2);
          color = (:darkorange, 0.13))
    band!(ax, [0, ex.t[end]], fill(ex.thr, 2), fill(maximum(ex.Rg)+0.05, 2);
          color = (:steelblue, 0.13))
    lines!(ax, ex.t, ex.Rg; color = :black, linewidth = 0.9)
    hlines!(ax, [ex.thr]; color = :red, linestyle = :dash, linewidth = 1.2)
    # Own column, not inside the axes — the trace spans the full width.
    Label(ga[i, 2], @sprintf("Φ = %.2f\nκ = %.2f\n%d transitions", ex.Φ, ex.κ, ex.nt);
          fontsize = 10, halign = :left, justification = :left, tellheight = false)
    xlims!(ax, 0, ex.t[end])
end

# ── B: effective potential per Phi ───────────────────────────────────────────
axB = Axis(fig[2, 1], xlabel = "Radius of gyration  Rg", ylabel = "−ln P(Rg)  [arb.]",
           title = "B   Double well tilts with activity")
Φsel = sort(collect(keys(rgby)))
Φsel = Φsel[round.(Int, range(1, length(Φsel), length = min(5, length(Φsel))))]
cmap = cgrad(:viridis, max(2,length(Φsel)); categorical = true)
for (i, Φ) in enumerate(Φsel)
    v = rgby[Φ]; length(v) < 5000 && continue
    lo, hi = 1.2, 2.9
    edges = range(lo, hi, length = 70); ctr = (edges[1:end-1] .+ edges[2:end])./2
    w = step(edges); h = zeros(Int, length(ctr))
    for x in v
        (lo <= x <= hi) || continue
        h[clamp(Int(fld(x-lo,w))+1, 1, length(h))] += 1
    end
    p = h ./ sum(h); k = p .> 0
    lines!(axB, ctr[k], -log.(p[k]) .- minimum(-log.(p[k]));
           color = cmap[i], linewidth = 2.2, label = @sprintf("Φ = %.1f", Φ))
end
axislegend(axB; position = :ct, framevisible = false, labelsize = 10, nbanks = 2)

# ── C: switching map ─────────────────────────────────────────────────────────
axC = Axis(fig[2, 2], xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
           title = "C   Where two-way switching occurs")
smap = joinpath(ROOT, "results", mapf, "switching_map.csv")
if isfile(smap)
    rows = [split(l,",") for l in readlines(smap)[2:end] if !isempty(strip(l))]
    κv = [parse(Float64,r[1]) for r in rows]; fv = [parse(Float64,r[2]) for r in rows]
    sv = [parse(Float64,r[6]) for r in rows]
    o = sortperm(sv)                      # draw switching cells on top
    sc = scatter!(axC, log10.(fv[o]), log10.(κv[o]); color = sv[o],
                  colormap = [:gray88, :gold, :crimson], colorrange = (0, 0.6),
                  markersize = 6)
    Colorbar(fig[2, 3], sc, label = "fraction of seeds switching both ways", width = 13)
    tk(v) = (range(minimum(v), maximum(v), length=5),
             [@sprintf("%.3g", 10^x) for x in range(minimum(v), maximum(v), length=5)])
    axC.xticks = tk(log10.(fv)); axC.yticks = tk(log10.(κv))
else
    text!(axC, 0.5, 0.5; text = "switching_map.csv not found", space = :relative,
          align = (:center,:center))
end

rowsize!(fig.layout, 1, Relative(0.46))
colsize!(ga, 2, Fixed(110))
out = joinpath(dir, "bistability_evidence.png")
save(out, fig; px_per_unit = 3)
CairoMakie.activate!(type = "pdf"); save(replace(out, ".png" => ".pdf"), fig); CairoMakie.activate!(type = "png")
println("✓ figure → $(relpath(out, ROOT))")
