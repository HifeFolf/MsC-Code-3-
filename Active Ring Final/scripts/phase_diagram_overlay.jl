# phase_diagram_overlay.jl
# Side-by-side comparison of circle IC vs kinked IC phase diagrams for N=17.
#
# Three panels (all log-log):
#   Left   — circle IC   ⟨b⟩ heatmap
#   Middle — kinked IC   ⟨b⟩ heatmap
#   Right  — Δb = b_kinked − b_circle  (diverging colormap)
#
# The Δb panel reveals where the two ICs reach different steady states —
# i.e. the bistable region. A contour line marks |Δb| = BISTABLE_THR.
# The linear stability boundary is overlaid on all three panels.
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_overlay.jl
#   julia --project scripts/phase_diagram_overlay.jl Results_N17_circle Results_N17_kinked
#
# Arguments:
#   arg 1 — circle IC results folder  (default: Results_N17_circle)
#   arg 2 — kinked IC results folder  (default: Results_N17_kinked)
#
# Output: results/phase_diagram_overlay.png

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_OV = joinpath(@__DIR__, "..")
const LAST_SECONDS    = 20.0
const FRAMERATE       = 60
const LAST_FRAMES     = round(Int, LAST_SECONDS * FRAMERATE)
const BISTABLE_THR    = 0.1   # Δb threshold for bistable-region contour

folder_circle = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle"
folder_kinked = length(ARGS) >= 2 ? ARGS[2] : "Results_N17_kinked"

dir_circle = joinpath(PROJECT_ROOT_OV, "results", folder_circle)
dir_kinked = joinpath(PROJECT_ROOT_OV, "results", folder_kinked)
isdir(dir_circle) || error("Circle IC folder not found: $dir_circle")
isdir(dir_kinked) || error("Kinked IC folder not found: $dir_kinked")

# ── Load b_ss for one results folder ─────────────────────────────────────────
function load_b_dict(dir)
    d = Dict{Tuple{Float64,Float64}, Float64}()
    run_dirs = sort(filter(s -> !isnothing(match(r"^run_\d+$", s)),
                           readdir(dir, join=false)))
    for rd in run_dirs
        obs = joinpath(dir, rd, "observables.jld2")
        par = joinpath(dir, rd, "params.toml")
        (isfile(obs) && isfile(par)) || continue
        ep    = TOML.parsefile(par)["params"]
        b_vec = jldopen(obs, "r") do f
            haskey(f, "b") ? f["b"] : nothing
        end
        b_vec === nothing && continue
        n   = length(b_vec)
        idx = max(1, n - LAST_FRAMES + 1):n
        d[(Float64(ep["kappa"]), Float64(ep["f0"]))] = mean(b_vec[idx])
    end
    return d
end

println("Loading circle IC  ($folder_circle) …")
dict_c = load_b_dict(dir_circle)
println("  $(length(dict_c)) runs loaded.")

println("Loading kinked IC  ($folder_kinked) …")
dict_k = load_b_dict(dir_kinked)
println("  $(length(dict_k)) runs loaded.")

# ── Build common grid ─────────────────────────────────────────────────────────
common_keys = intersect(keys(dict_c), keys(dict_k))
@printf("Common (κ,f₀) pairs: %d\n", length(common_keys))

κ_vals  = sort(unique(k[1] for k in common_keys))
f0_vals = sort(unique(k[2] for k in common_keys))
nκ = length(κ_vals);  nf = length(f0_vals)
@printf("Grid: %d κ × %d f₀ = %d cells\n", nκ, nf, nκ * nf)

# b_matrix[ic, ir] = b at (κ_vals[ic], f0_vals[ir])
b_circle = fill(NaN, nκ, nf)
b_kinked = fill(NaN, nκ, nf)
for (ic, κ) in enumerate(κ_vals), (ir, f₀) in enumerate(f0_vals)
    key = (κ, f₀)
    haskey(dict_c, key) && (b_circle[ic, ir] = dict_c[key])
    haskey(dict_k, key) && (b_kinked[ic, ir] = dict_k[key])
end

Δb = b_kinked .- b_circle   # NaN wherever either run is missing

Δb_finite = filter(!isnan, Δb)
if !isempty(Δb_finite)
    n_pos = count(x -> x >  BISTABLE_THR, Δb_finite)
    n_neg = count(x -> x < -BISTABLE_THR, Δb_finite)
    @printf("Δb range : [%.4f,  %.4f]\n", minimum(Δb_finite), maximum(Δb_finite))
    @printf("Cells with Δb >  %.2f : %d / %d\n", BISTABLE_THR, n_pos, nκ*nf)
    @printf("Cells with Δb < -%.2f : %d / %d\n", BISTABLE_THR, n_neg, nκ*nf)
end

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (1500, 560))

Label(fig[0, 1:3],
      "N=17 phase diagram: circle IC vs kinked IC  (kBT=0.1,  last $(round(Int,LAST_SECONDS))s)";
      fontsize = 15, tellwidth = false)

axis_kw = (xlabel = "Active force  f₀",
           xscale = log10,  yscale = log10,
           xticks = LogTicks(WilkinsonTicks(5)),
           yticks = LogTicks(WilkinsonTicks(5)))

ax1 = Axis(fig[1, 1]; title = "Circle IC  ⟨b⟩",          ylabel = "Bending stiffness  κ", axis_kw...)
ax2 = Axis(fig[1, 2]; title = "Kinked IC  ⟨b⟩",           ylabel = "",                     axis_kw...)
ax3 = Axis(fig[1, 3]; title = "Δb = kinked − circle",     ylabel = "",                     axis_kw...)

linkyaxes!(ax1, ax2, ax3)
linkxaxes!(ax1, ax2, ax3)

b_finite = filter(!isnan, vcat(vec(b_circle), vec(b_kinked)))
cmax     = isempty(b_finite) ? 1.0 : min(1.0, maximum(b_finite))

hm1 = heatmap!(ax1, f0_vals, κ_vals, b_circle';
               colormap = :viridis, colorrange = (0.0, cmax), nan_color = :white)
hm2 = heatmap!(ax2, f0_vals, κ_vals, b_kinked';
               colormap = :viridis, colorrange = (0.0, cmax), nan_color = :white)

Δmax = isempty(Δb_finite) ? 0.5 : max(maximum(abs, Δb_finite), 0.05)
hm3  = heatmap!(ax3, f0_vals, κ_vals, Δb';
                colormap = :RdBu, colorrange = (-Δmax, Δmax), nan_color = :lightgray)

Colorbar(fig[1, 4], hm1; label = "⟨b⟩  (0 = circular,  1 = rod-like)", width = 16)
Colorbar(fig[1, 5], hm3; label = "Δb  (kinked − circle)",               width = 16)

# ── Stability boundary overlay ────────────────────────────────────────────────
added_boundary = false
for N_try in [17, 16]
    csv_path = joinpath(PROJECT_ROOT_OV, "results", "linear_stability",
                        "phase_boundary_N$(N_try).csv")
    isfile(csv_path) || continue
    fields  = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
    κ_stab  = [parse(Float64, f[1]) for f in fields]
    f0_stab = [parse(Float64, f[2]) for f in fields]
    in_range = (f0_stab .>= minimum(f0_vals)) .& (f0_stab .<= maximum(f0_vals)) .&
               (κ_stab  .>= minimum(κ_vals))  .& (κ_stab  .<= maximum(κ_vals))
    any(in_range) || continue
    for ax in (ax1, ax2)
        lines!(ax, f0_stab[in_range], κ_stab[in_range];
               color = :white, linewidth = 2.0, linestyle = :dash,
               label = "Linear stability (N=$N_try)")
    end
    lines!(ax3, f0_stab[in_range], κ_stab[in_range];
           color = :black, linewidth = 2.0, linestyle = :dash,
           label = "Linear stability (N=$N_try)")
    println("  Overlaid stability boundary (N=$N_try)")
    global added_boundary = true
end
if added_boundary
    axislegend(ax1; position = :lt, backgroundcolor = (:black, 0.4), labelcolor = :white)
end

# ── Bistable-region contour on Δb panel ───────────────────────────────────────
if !isempty(Δb_finite) && maximum(Δb_finite) > BISTABLE_THR
    contour!(ax3, f0_vals, κ_vals, Δb';
             levels = [BISTABLE_THR], color = :black, linewidth = 1.5)
    @printf("  Bistable contour drawn at Δb = %.2f\n", BISTABLE_THR)
end

out_path = joinpath(PROJECT_ROOT_OV, "results", "phase_diagram_overlay.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
