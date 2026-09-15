# phase_diagram_motion.jl
# κ × f₀ phase diagram of HOW THE RING MOVES — rotating in place vs translating.
#
# One number per run:   translation fraction  =  v_com / v_mean
#
#   v_com  = speed of the centre of mass      (bulk motion)
#   v_mean = mean speed of the individual beads (all motion)
#
#   ≈ 1  every bead moves together        → rigid TRANSLATION (a gliding ellipse)
#   < 1  much of the motion is internal    → ROTATION / churn (a spinning crumple)
#
# Unlike the panel inside phase_diagram_kconn_states.jl, this covers EVERY cell,
# not just the ellipse family — the rotating/translating question is meaningful
# for folded and slithering rings too.
#
# The median (not mean) is taken over the run: the periodic wrap shifts the whole
# ring by L in a single frame, which would otherwise inject a huge spurious
# centre-of-mass speed spike.
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_motion.jl
#   julia --project scripts/phase_diagram_motion.jl Results_N17_circle_indep
#   julia --project scripts/phase_diagram_motion.jl Results_N17_circle_indep 1 500

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT = joinpath(@__DIR__, "..")
const SKIP_SECONDS = 0.5    # drop the brief IC-relaxation transient (sim-time)
const V_STILL      = 1.0    # v_mean below this ⇒ barely moving, ratio meaningless

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle_indep"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id_start ≤ parse(Int, m[1]) ≤ id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder…")

κ_raw = Float64[]; f0_raw = Float64[]; rat_raw = Float64[]; vm_raw = Float64[]
for rd in run_dirs
    obs = joinpath(results_dir, rd, "observables.jld2")
    par = joinpath(results_dir, rd, "params.toml")
    (isfile(obs) && isfile(par)) || continue
    ep = TOML.parsefile(par)["params"]
    d = jldopen(obs, "r") do f
        (haskey(f, "v_com") && haskey(f, "v_mean")) || return nothing
        m = length(f["v_com"])
        t = haskey(f, "times") ? f["times"] : collect(1:m)
        i0 = something(findfirst(≥(SKIP_SECONDS), t), 1)
        (vc = median(@view f["v_com"][i0:end]),
         vm = median(@view f["v_mean"][i0:end]))
    end
    d === nothing && continue
    push!(κ_raw,  Float64(ep["kappa"]))
    push!(f0_raw, Float64(ep["f0"]))
    push!(vm_raw, d.vm)
    # A ring that is barely moving has no meaningful direction of motion — its
    # ratio is thermal noise over thermal noise.  Mark it rather than plot junk.
    push!(rat_raw, d.vm > V_STILL ? d.vc / d.vm : NaN)
end

n = length(rat_raw)
println("Loaded $n runs with velocity data.")
n == 0 && error("No v_com/v_mean found — this sweep predates those observables.")
finite = filter(isfinite, rat_raw)
@printf("translation fraction: min=%.3f  median=%.3f  max=%.3f   (%d cells too still to score)\n",
        minimum(finite), median(finite), maximum(finite), n - length(finite))

# ── κ × f₀ grid ───────────────────────────────────────────────────────────────
kappa_vals = sort(unique(κ_raw)); f0_vals = sort(unique(f0_raw))
kidx = Dict(v => i for (i, v) in enumerate(kappa_vals))
fidx = Dict(v => i for (i, v) in enumerate(f0_vals))
mat = fill(NaN, length(kappa_vals), length(f0_vals))
for i in 1:n
    mat[kidx[κ_raw[i]], fidx[f0_raw[i]]] = rat_raw[i]
end

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (980, 780))
ax = Axis(fig[1, 1];
    xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
    title  = "Motion: rotating ↔ translating — $subfolder  (N runs = $n)",
    xscale = log10, yscale = log10,
    xticks = LogTicks(WilkinsonTicks(6)), yticks = LogTicks(WilkinsonTicks(6)))
hm = heatmap!(ax, f0_vals, kappa_vals, mat';
              colormap = :viridis, colorrange = (0.3, 1.0),
              lowclip = :black, nan_color = :lightgrey)
Colorbar(fig[1, 2], hm;
         label = "v_com / v_mean    dark = rotates in place,  bright = glides",
         width = 15)
Label(fig[2, 1:2], "grey = too still to have a direction of motion (v_mean < $V_STILL)";
      fontsize = 11, tellwidth = false)

out = joinpath(results_dir, "phase_diagram_motion.png")
save(out, fig, px_per_unit = 2)
println("Saved: $out")
