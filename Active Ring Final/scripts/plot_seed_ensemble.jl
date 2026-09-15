# plot_seed_ensemble.jl
# Ten seeds at ONE parameter point, stacked, with the short-window cutoff marked.
#
# The argument this figure makes:
#   Two runs at identical parameters ending in different states is the intuitive
#   demonstration of bistability — and it is NOT conclusive on its own, because a
#   single stable state plus an absorbing one produces exactly the same picture.
#   Marking t = 30 s on 300 s traces shows why: at the cutoff the seeds split
#   5 folded / 5 open and look like basin selection, but every trace visits BOTH
#   states, and several are in the OPPOSITE state at 300 s.  The ensemble spread
#   is a snapshot of ergodic hopping, not a record of which basin noise picked.
#
# This is also the view that matches experiment: many nominally identical samples
# observed once, rather than one sample watched for a long time.
#
# Usage:  julia --project scripts/plot_seed_ensemble.jl [folder] [kappa] [f0]

using CairoMakie, Printf, Statistics, TOML, JLD2

const ROOT = joinpath(@__DIR__, "..")
folder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_bistable_long"
κt     = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
f0t    = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 2.2143464
const RG_FRAC = 0.70
const T_MARK  = 30.0     # the short-campaign window, for comparison

dir = joinpath(ROOT, "results", folder)
tr = NamedTuple[]
for rn in sort(filter(d -> occursin(r"^run_\d+$", d), readdir(dir)))
    par = joinpath(dir,rn,"params.toml"); obs = joinpath(dir,rn,"observables.jld2")
    (isfile(par) && isfile(obs)) || continue
    ep = TOML.parsefile(par)["params"]
    (isapprox(Float64(ep["kappa"]), κt; rtol=1e-4) &&
     isapprox(Float64(ep["f0"]), f0t; rtol=1e-4)) || continue
    N = Int(ep["N"]); thr = RG_FRAC*(N/(2π))
    d = jldopen(obs,"r") do f
        (t = f["times"], Rg = f["Rg"])
    end
    seed = 0
    try
        seed = parse(Int, match(r"\"seed\":\s*(\d+)", read(joinpath(dir,rn,"metadata.json"), String))[1])
    catch; end
    i5 = something(findfirst(≥(5.0), d.t), 1)
    im = something(findfirst(≥(T_MARK), d.t), length(d.t))
    push!(tr, (seed = seed, t = d.t, Rg = d.Rg, thr = thr,
               s30 = d.Rg[im] < thr, s300 = d.Rg[end] < thr,
               occ = count(<(thr), d.Rg[i5:end])/length(d.Rg[i5:end])))
end
isempty(tr) && error("no runs at kappa=$κt f0=$f0t in $folder")
sort!(tr, by = x -> x.seed)
@printf("%d seeds at kappa=%.3f f0=%.4f (Phi=%.2f)\n", length(tr), κt, f0t, f0t/κt)

n = length(tr)
fig = Figure(size = (1400, 130 + 78n))
Label(fig[1, 1], @sprintf("Ten seeds, identical parameters  (κ = %.2f, Φ = %.2f)", κt, f0t/κt);
      fontsize = 16, font = :bold, halign = :left, tellwidth = false)
Label(fig[2, 1], "At t = 30 s the ensemble splits — but every trace visits BOTH states, " *
      "and several end in the opposite state at 300 s.";
      fontsize = 12, halign = :left, tellwidth = false)
g = fig[3, 1] = GridLayout()
for (i, x) in enumerate(tr)
    ax = Axis(g[i, 1], ylabel = @sprintf("s%d", x.seed),
              xlabel = i == n ? "time  (s)" : "", xticklabelsvisible = i == n,
              ylabelrotation = 0, yticksvisible = false, yticklabelsvisible = false)
    band!(ax, [0, x.t[end]], fill(1.2, 2), fill(x.thr, 2); color = (:darkorange, 0.13))
    band!(ax, [0, x.t[end]], fill(x.thr, 2), fill(2.85, 2); color = (:steelblue, 0.13))
    lines!(ax, x.t, x.Rg; color = :black, linewidth = 0.7)
    vlines!(ax, [T_MARK]; color = :red, linewidth = 1.6)
    # Annotation goes in its OWN grid column, not inside the axes: the traces run
    # the full width, so text placed at the axis edge sits on top of the data.
    Label(g[i, 2], @sprintf("@30s %s\n@300s %s\nfolded %.0f%%",
                            x.s30 ? "FOLDED" : "open", x.s300 ? "FOLDED" : "open", 100x.occ);
          fontsize = 9, halign = :left, justification = :left, tellheight = false)
    xlims!(ax, 0, x.t[end]); ylims!(ax, 1.2, 2.85)
    i == 1 && text!(ax, T_MARK + 3, 2.7; text = "← 30 s window", color = :red, fontsize = 10)
end
rowgap!(g, 2)
colsize!(g, 2, Fixed(120))
out = joinpath(dir, "seed_ensemble.png")
save(out, fig); println("✓ figure → $(relpath(out, ROOT))")
