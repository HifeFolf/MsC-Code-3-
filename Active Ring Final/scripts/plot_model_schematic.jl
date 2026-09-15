# plot_model_schematic.jl
# The model, and the two states it chooses between — the "what is this system?"
# slide.  Nothing in the project drew this before.
#
#   A  the N=17 anti-polar ring: half the beads drive counter-clockwise, half
#      clockwise, meeting at ONE passive junction bead.  Net torque is zero by
#      construction, so the drive cannot be relieved by rotation — it has to go
#      into shape.  The junction is where that frustration localises, which is
#      why Kconn (the turning angle there) is the order parameter.
#   B  the two states, taken from a REAL switching trajectory rather than drawn:
#      open ring (Rg ~ N/2pi) and folded crescent (Rg ~ N/4pi).
#
# Usage:  julia --project scripts/plot_model_schematic.jl

using CairoMakie, Printf, Statistics, JLD2, TOML

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "src", "EngineRing.jl")); using .EngineRing
include(joinpath(ROOT, "src", "InitialConditions.jl")); using .InitialConditions

pp = TOML.parsefile(joinpath(ROOT, "configs", "default.toml"))["params"]
p = EngineRing.Params(N=Int(pp["N"]), l=pp["l"], ks=pp["ks"], κ=pp["kappa"],
    theta_max=pp["theta_max"], theta_buf=pp["theta_buf"], k_stop=pp["k_stop"],
    sigma=pp["sigma"], epsilon=pp["epsilon"], topology=Symbol(pp["topology"]),
    f0=pp["f0"], mu=pp["mu"], kBT=pp["kBT"], dt=pp["dt"])
bt = InitialConditions.antipolar_bead_types(p.N)
r  = InitialConditions.perfect_circle_ic(p, bt).r
N  = p.N

COL = Dict(:active => RGBAf(0.85,0.33,0.10,1), :active_rev => RGBAf(0.20,0.45,0.70,1),
           :passive => RGBAf(0.15,0.15,0.15,1))

fig = Figure(size = (1500, 700))

# ── A: the model ─────────────────────────────────────────────────────────────
axA = Axis(fig[1,1], aspect = DataAspect(), title = "A   The anti-polar ring  (N = 17)",
           xlabel = "", ylabel = "")
hidedecorations!(axA); hidespines!(axA)
cx = vcat(r[1,:], r[1,1]); cy = vcat(r[2,:], r[2,1])
lines!(axA, cx, cy; color = (:gray50, 0.8), linewidth = 2)
# tangential drive arrows: t̂ = mean of the two adjacent bond directions
for i in 1:N
    bt[i] == :passive && continue
    ip = i == N ? 1 : i+1; im = i == 1 ? N : i-1
    d1 = r[:,ip] .- r[:,i];  d2 = r[:,i] .- r[:,im]
    t  = (d1./sqrt(sum(abs2,d1)) .+ d2./sqrt(sum(abs2,d2)))
    t ./= sqrt(sum(abs2,t))
    s = bt[i] == :active ? 1.0 : -1.0
    arrows!(axA, [r[1,i]], [r[2,i]], [0.62s*t[1]], [0.62s*t[2]];
            color = COL[bt[i]], linewidth = 2.4, arrowsize = 11)
end
for i in 1:N
    scatter!(axA, [r[1,i]], [r[2,i]]; color = COL[bt[i]], markersize = bt[i]==:passive ? 26 : 19,
             strokecolor = :white, strokewidth = 1.5)
end
ip = 1; im = N   # passive bead is index (N-1)÷2+1
j = (N-1)÷2 + 1
text!(axA, r[1,j]*1.42, r[2,j]*1.42; text = "passive junction\n— Kconn measured here",
      align = (:center,:center), fontsize = 12, font = :bold)
lines!(axA, [r[1,j]*0.92, r[1,j]*0.58], [r[2,j]*0.92, r[2,j]*0.58]; color = :black, linewidth = 1.2)
Legend(fig[2,1],
       [MarkerElement(color=COL[:active], marker=:circle, markersize=15),
        MarkerElement(color=COL[:active_rev], marker=:circle, markersize=15),
        MarkerElement(color=COL[:passive], marker=:circle, markersize=15)],
       ["active  (drives CCW)", "active_rev  (drives CW)", "passive junction"],
       orientation = :horizontal, framevisible = false, labelsize = 12)

# ── B: the two states, from a real trajectory ────────────────────────────────
tp = joinpath(ROOT,"results","Results_N17_bistable_long","run_0004","traj.jld2")
op = joinpath(ROOT,"results","Results_N17_bistable_long","run_0004","observables.jld2")
gb = fig[1,2] = GridLayout()
if isfile(tp) && isfile(op)
    pos = jldopen(f->f["positions"], tp, "r")
    Rg  = jldopen(f->f["Rg"], op, "r")
    thr = 0.70*(N/(2π))
    iopen   = argmax(Rg); ifold = argmin(Rg)
    for (col,(idx,lab)) in enumerate([(iopen,"OPEN"), (ifold,"FOLDED")])
        q = pos[:,:,idx]; q = q .- mean(q, dims=2)
        ax = Axis(gb[1,col], aspect = DataAspect(),
                  title = @sprintf("%s   Rg = %.2f", lab, Rg[idx]))
        hidedecorations!(ax); hidespines!(ax)
        lines!(ax, vcat(q[1,:],q[1,1]), vcat(q[2,:],q[2,1]);
               color = lab=="OPEN" ? RGBAf(0.27,0.47,0.72,1) : RGBAf(0.95,0.55,0.10,1),
               linewidth = 3)
        for i in 1:N
            scatter!(ax, [q[1,i]],[q[2,i]]; color=COL[bt[i]], markersize=13,
                     strokecolor=:white, strokewidth=1)
        end
        limits!(ax, -3.2, 3.2, -3.2, 3.2)
    end
    Label(gb[2,1:2], @sprintf("the same run, %.0f s apart — open Rg ≈ N/2π = %.2f, folded ≈ N/4π = %.2f",
          abs(iopen-ifold)*0.05, N/(2π), N/(4π)); fontsize = 12, tellwidth = false)
else
    Label(gb[1,1], "trajectory not found"; tellwidth = false)
end
Label(fig[0,2], "B   The two states it chooses between"; fontsize = 15, font = :bold,
      halign = :left, tellwidth = false)

base = joinpath(ROOT,"results","campaign","model_schematic")
save(base*".png", fig; px_per_unit=3)
CairoMakie.activate!(type="pdf"); save(base*".pdf", fig); CairoMakie.activate!(type="png")
println("✓ ", relpath(base, ROOT), ".png / .pdf")
