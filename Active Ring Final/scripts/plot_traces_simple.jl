# plot_traces_simple.jl
# The simplest evidence figure: one passive run and one active run at the same
# stiffness, Rg against time, nothing else on the axes.
using CairoMakie, Printf, Statistics, TOML, JLD2
const ROOT = joinpath(@__DIR__, "..")
const KREF = 1.0
pick(folder, f0want) = begin
    dir = joinpath(ROOT,"results",folder)
    for rn in sort(filter(d->occursin(r"^run_\d+$",d), readdir(dir)))
        ep = TOML.parsefile(joinpath(dir,rn,"params.toml"))["params"]
        (isapprox(ep["kappa"],KREF;rtol=1e-3) && isapprox(ep["f0"],f0want;rtol=5e-2)) || continue
        return jldopen(f->(t=f["times"], Rg=f["Rg"]), joinpath(dir,rn,"observables.jld2"), "r")
    end
    nothing
end
pas = pick("Results_N17_passive", 0.0)
act = pick("Results_N17_bistable_long", 2.2143464)
(pas === nothing || act === nothing) && error("missing a run")
thr = 0.70*17/(2π)
fig = Figure(size=(1600,660))
ax = Axis(fig[1,1], xlabel="time  [seconds]", ylabel="ring size  Rg",
          xlabelsize=30, ylabelsize=30, xticklabelsize=26, yticklabelsize=26)
hlines!(ax, [thr]; color=(:black,0.45), linestyle=:dash, linewidth=2)
text!(ax, 4, thr+0.035; text="folded below this line", fontsize=26, color=(:black,0.6))
lines!(ax, pas.t, pas.Rg; color=:steelblue, linewidth=1.6, label="motors OFF")
lines!(ax, act.t, act.Rg; color=:crimson,  linewidth=1.6, label="motors ON")
axislegend(ax; position=:rb, framevisible=false, labelsize=28)
xlims!(ax, 0, 300)
save(joinpath(ROOT,"results","campaign","traces_simple.png"), fig; px_per_unit=3)
println("✓ results/campaign/traces_simple.png")
