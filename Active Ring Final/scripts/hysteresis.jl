# hysteresis.jl
# Hysteresis protocol for the anti-polar ring.
#
# Ramp f₀ up from f0_min to f0_max (log-spaced), then back down.
# The time-averaged shape parameter b is recorded at each f₀ step.
# A hysteresis loop in b vs f₀ proves bistability.
#
# Key: the state is NOT reinitialised between f₀ steps — it continues
# from wherever the previous step left it.  The up-ramp starts from
# a perfect circle IC; the down-ramp starts from the end of the up-ramp.
#
# Recommended κ: 1–2  (so the stability boundary f₀_c≈75κ falls within
# the sweep window).  At κ=1: f₀_c≈75, sweep 1→200 crosses it cleanly.
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/hysteresis.jl
#   julia --project scripts/hysteresis.jl configs/default.toml
#   julia --project scripts/hysteresis.jl configs/default.toml  kappa  f0_max  f0_min  n_steps
#
# Example — κ=1, sweep f₀ over 1→200 in 30 steps:
#   julia --project scripts/hysteresis.jl configs/default.toml  1.0  200.0  1.0  30
#
# Output:
#   results/hysteresis/hysteresis_N{N}_kappa{k}.jld2   (raw data)
#   results/hysteresis/hysteresis_N{N}_kappa{k}.png    (hysteresis curve)

using TOML, Random, Printf, Statistics, JLD2, CairoMakie

const PROJECT_ROOT_HY = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_HY, "src", "EngineRing.jl"));        using .EngineRing
include(joinpath(PROJECT_ROOT_HY, "src", "InitialConditions.jl")); using .InitialConditions
include(joinpath(PROJECT_ROOT_HY, "src", "Observables.jl"));       using .Observables

# ── Parse arguments ───────────────────────────────────────────────────────────
config_path = length(ARGS) >= 1 ? ARGS[1] :
              joinpath(PROJECT_ROOT_HY, "configs", "default.toml")
cfg = TOML.parsefile(config_path)
ep  = cfg["params"]
rc  = cfg["run"]

kappa_val = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
f0_max    = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 200.0
f0_min    = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
n_steps   = length(ARGS) >= 5 ? parse(Int,     ARGS[5]) : 30
kBT_val   = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : ep["kBT"]

# Equilibration and recording frames per f₀ step.
# Increase frames_equil if the ring hasn't relaxed between steps.
frames_equil  = 400
frames_record = 200
substeps      = rc["substeps"]
seed          = rc["seed"]

N = Int(ep["N"])
dt_frame = substeps * ep["dt"]   # simulation time per frame

println("=" ^ 65)
@printf("Hysteresis  |  N=%d  κ=%.2f  kBT=%.3f\n", N, kappa_val, kBT_val)
@printf("f₀ sweep: %.2f → %.2f → %.2f  (%d steps, log-spaced)\n",
        f0_min, f0_max, f0_min, n_steps)
@printf("Per step: %d equil + %d record frames  (Δt=%.4f each)\n",
        frames_equil, frames_record, dt_frame)
@printf("Time per step: %.2f + %.2f = %.2f sim units\n",
        frames_equil*dt_frame, frames_record*dt_frame,
        (frames_equil+frames_record)*dt_frame)
println("=" ^ 65)

# ── Params factory ────────────────────────────────────────────────────────────
function make_params(f0_val)
    EngineRing.Params(
        N         = N,
        l         = ep["l"],
        ks        = ep["ks"],
        κ         = kappa_val,
        theta_max = ep["theta_max"],
        theta_buf = ep["theta_buf"],
        k_stop    = ep["k_stop"],
        sigma     = ep["sigma"],
        epsilon   = ep["epsilon"],
        topology  = Symbol(ep["topology"]),
        f0        = f0_val,
        mu        = ep["mu"],
        kBT       = kBT_val,
        dt        = ep["dt"],
    )
end

# ── Run a segment at fixed f₀, return time-averaged b ────────────────────────
function run_segment!(state, f0_val, n_equil, n_record)
    p = make_params(f0_val)
    for _ in 1:n_equil
        for _ in 1:substeps
            EngineRing.step!(state, p)
        end
    end
    b_acc = 0.0
    for _ in 1:n_record
        for _ in 1:substeps
            EngineRing.step!(state, p)
        end
        b_acc += Observables.compute_frame_obs(state.r).b
    end
    return b_acc / n_record
end

# ── f₀ ramp values (log-spaced) ───────────────────────────────────────────────
f0_up   = exp.(range(log(f0_min), log(f0_max), length = n_steps))
f0_down = reverse(f0_up)

Random.seed!(seed)
bead_types = InitialConditions.antipolar_bead_types(N)
p0 = make_params(f0_min)

# ── Up ramp ───────────────────────────────────────────────────────────────────
println("\nUp ramp  ↑  (start from circle IC, ramp f₀ upward)")
state_up = InitialConditions.create_ic("antipolar", p0, bead_types)
b_up = Vector{Float64}(undef, n_steps)
t_wall = time()
for (i, f0) in enumerate(f0_up)
    b_up[i] = run_segment!(state_up, f0, frames_equil, frames_record)
    @printf("  ↑  %2d/%2d  f₀=%8.2f  b=%.4f\n", i, n_steps, f0, b_up[i])
end
@printf("Up ramp done in %.1f s\n", time() - t_wall)

# ── Down ramp (continue from final state of up ramp) ─────────────────────────
println("\nDown ramp  ↓  (continue from up-ramp final state, ramp f₀ downward)")
state_dn = deepcopy(state_up)
b_dn = Vector{Float64}(undef, n_steps)
t_wall = time()
for (i, f0) in enumerate(f0_down)
    b_dn[i] = run_segment!(state_dn, f0, frames_equil, frames_record)
    @printf("  ↓  %2d/%2d  f₀=%8.2f  b=%.4f\n", i, n_steps, f0, b_dn[i])
end
@printf("Down ramp done in %.1f s\n", time() - t_wall)

# Reverse b_dn so index i corresponds to f0_up[i] (both ordered low→high)
b_dn_ordered = reverse(b_dn)

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n── Results ───────────────────────────────────────────────────────────")
println("  f₀          b_up     b_down    Δb")
println("  " * "-"^42)
for i in 1:n_steps
    db = b_dn_ordered[i] - b_up[i]
    @printf("  %8.2f   %.4f   %.4f   %+.4f\n", f0_up[i], b_up[i], b_dn_ordered[i], db)
end

BISTABLE_THR = 0.05
diff = abs.(b_up .- b_dn_ordered)
bistable_mask = diff .> BISTABLE_THR
if any(bistable_mask)
    f0_bi = f0_up[bistable_mask]
    @printf("\nBistable region (|Δb| > %.2f): f₀ ∈ [%.2f,  %.2f]\n",
            BISTABLE_THR, minimum(f0_bi), maximum(f0_bi))
else
    println("\nNo hysteresis detected (|Δb| < $BISTABLE_THR everywhere).")
    println("Try longer equilibration, lower kBT, or adjust κ so that")
    println("the stability boundary f₀_c≈75κ falls within [f0_min, f0_max].")
end

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = joinpath(PROJECT_ROOT_HY, "results", "hysteresis")
mkpath(out_dir)
tag = @sprintf("N%d_kappa%.2f_kBT%.3f", N, kappa_val, kBT_val)

jld_path = joinpath(out_dir, "hysteresis_$(tag).jld2")
jldopen(jld_path, "w") do f
    f["f0_vals"]      = f0_up
    f["b_up"]         = b_up
    f["b_dn_ordered"] = b_dn_ordered
    f["kappa"]        = kappa_val
    f["N"]            = N
    f["kBT"]          = kBT_val
end
println("\nSaved data: $jld_path")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(size = (750, 460))
ax  = Axis(fig[1, 1];
           xlabel  = "Active force  f₀",
           ylabel  = "Shape parameter  ⟨b⟩",
           title   = "Hysteresis  |  N=$N  κ=$kappa_val  kBT=$kBT_val",
           xscale  = log10,
           xticks  = LogTicks(WilkinsonTicks(6)),
           limits  = (nothing, (-0.02, 1.02)))

# Shaded bistable band
if any(bistable_mask)
    f0_bi = f0_up[bistable_mask]
    vspan!(ax, minimum(f0_bi), maximum(f0_bi);
           color = (:gold, 0.25), label = "hysteresis band (|Δb|>$(BISTABLE_THR))")
end

# Stability boundary
f0_c = 75.0 * kappa_val
if f0_min <= f0_c <= f0_max
    vlines!(ax, [f0_c]; color = :black, linestyle = :dash, linewidth = 1.5,
            label = @sprintf("linear stability  f₀_c≈%d", round(Int, f0_c)))
end

lines!(ax,   f0_up, b_up;          color = :royalblue, linewidth = 2.5, label = "↑ up ramp (circle IC)")
scatter!(ax, f0_up, b_up;          color = :royalblue, markersize = 7)
lines!(ax,   f0_up, b_dn_ordered;  color = :crimson,   linewidth = 2.5, label = "↓ down ramp")
scatter!(ax, f0_up, b_dn_ordered;  color = :crimson,   markersize = 7)

axislegend(ax; position = :lt)

png_path = joinpath(out_dir, "hysteresis_$(tag).png")
save(png_path, fig, px_per_unit = 2)
println("Saved plot:  $png_path")

println("\nDone.")
