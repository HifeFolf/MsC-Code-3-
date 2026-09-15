# phase_diagram_seeds.jl
# Aggregated phase diagram combining three seed runs of the N=17 circle sweep.
#
# For each (κ, f₀) cell the script computes:
#   • ⟨b⟩  — mean asphericity across seeds  (signal)
#   • σ(b)  — std dev across seeds  (highlights IC-sensitive / bistable region)
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_seeds.jl
#
# Output:
#   results/phase_diagram_seeds.png

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_PS = joinpath(@__DIR__, "..")

const LAST_SECONDS = 20.0   # average over the last N video-seconds of each run
const FRAMERATE    = 60     # frames per video second

# ── Folders to aggregate ──────────────────────────────────────────────────────
seed_folders = [
    ("seed 41", "Results_N17_circle_seed41"),
    ("seed 42", "Results_N17_circle"),
    ("seed 43", "Results_N17_circle_seed43"),
]

# ── Load one folder → Dict{(κ,f₀) => b_ss} ───────────────────────────────────
function load_folder(subfolder)
    results_dir = joinpath(PROJECT_ROOT_PS, "results", subfolder)
    isdir(results_dir) || error("Folder not found: $results_dir")

    lookup = Dict{Tuple{Float64,Float64}, Float64}()
    run_dirs = filter(d -> occursin(r"^run_\d+$", d), readdir(results_dir, join=false))
    for rd in run_dirs
        run_path    = joinpath(results_dir, rd)
        obs_file    = joinpath(run_path, "observables.jld2")
        params_file = joinpath(run_path, "params.toml")
        (isfile(obs_file) && isfile(params_file)) || continue

        ep = TOML.parsefile(params_file)["params"]
        κ  = Float64(ep["kappa"])
        f₀ = Float64(ep["f0"])

        b_vec = jldopen(obs_file, "r") do f
            haskey(f, "b") ? f["b"] : nothing
        end
        b_vec === nothing && continue

        last_frames = round(Int, LAST_SECONDS * FRAMERATE)
        idx  = max(1, length(b_vec) - last_frames + 1):length(b_vec)
        lookup[(κ, f₀)] = mean(b_vec[idx])
    end
    @printf("  %-30s  %d runs\n", subfolder, length(lookup))
    return lookup
end

println("Loading seed runs…")
lookups = [load_folder(sf) for (_, sf) in seed_folders]

# ── Build shared grid ─────────────────────────────────────────────────────────
all_kappas = sort(unique(k for lk in lookups for (k, _) in keys(lk)))
all_f0s    = sort(unique(f for lk in lookups for (_, f) in keys(lk)))
nκ = length(all_kappas)
nf = length(all_f0s)
ns = length(lookups)
@printf("\nGrid: %d κ  ×  %d f₀  ×  %d seeds\n", nκ, nf, ns)

# 3-D array: [κ_idx, f0_idx, seed_idx]
b_cube = fill(NaN, nκ, nf, ns)
for (si, lk) in enumerate(lookups)
    for (ik, κ) in enumerate(all_kappas), (jf, f₀) in enumerate(all_f0s)
        v = get(lk, (κ, f₀), NaN)
        b_cube[ik, jf, si] = v
    end
end

# Mean and std dev — only over non-NaN seeds at each cell
b_mean = fill(NaN, nκ, nf)
b_std  = fill(NaN, nκ, nf)
for ik in 1:nκ, jf in 1:nf
    vals = filter(!isnan, b_cube[ik, jf, :])
    length(vals) < 2 && continue
    b_mean[ik, jf] = mean(vals)
    b_std[ik, jf]  = std(vals)
end

n_full  = count(!isnan, b_mean)
n_total = nκ * nf
@printf("Complete cells (≥2 seeds): %d / %d\n", n_full, n_total)

# ── Individual per-seed matrices ─────────────────────────────────────────────
b_each = [b_cube[:, :, si] for si in 1:ns]

# ── Plot ──────────────────────────────────────────────────────────────────────
σ_max = maximum(filter(!isnan, b_std); init = 0.0)

fig = Figure(size = (1700, 1050))

Label(fig[0, 1:5],
      "N=17 circle IC  |  phase diagram across seeds  (kBT=0.1, last $(round(Int,LAST_SECONDS))s averaged)";
      fontsize = 15, tellwidth = false)

# ── Helper: one heatmap panel ────────────────────────────────────────────────
function add_panel!(fig, row, col, mat, title_str; cmap=:viridis, crange=(0.0,1.0))
    ax = Axis(fig[row, col];
              xlabel  = "Active force  f₀",
              ylabel  = "Bending stiffness  κ",
              title   = title_str,
              xscale  = log10,
              yscale  = log10)
    hm = heatmap!(ax, all_f0s, all_kappas, mat';
                  colormap   = cmap,
                  colorrange = crange,
                  nan_color  = :lightgrey)
    return ax, hm
end

# Row 1: individual seeds
for (si, (label, _)) in enumerate(seed_folders)
    cmax = maximum(filter(!isnan, b_each[si]); init=1.0)
    ax, hm = add_panel!(fig, 1, si, b_each[si], label;
                        cmap=:viridis, crange=(0.0, min(1.0, cmax)))
    if si == ns
        Colorbar(fig[1, ns+1], hm; label="b", width=14)
    end
end

# Row 2 left: mean across seeds
ax_m, hm_m = add_panel!(fig, 2, 1, b_mean, "Mean  ⟨b⟩  (all seeds)";
                         cmap=:viridis, crange=(0.0, 1.0))
Colorbar(fig[2, 2], hm_m; label="⟨b⟩", width=14)

# Row 2 right: std dev across seeds
ax_s, hm_s = add_panel!(fig, 2, 3, b_std,
                         "Std dev  σ(b)  across seeds  — IC sensitivity";
                         cmap=:heat, crange=(0.0, max(σ_max, 0.01)))
Colorbar(fig[2, 4], hm_s; label="σ(b)", width=14)

# Linear stability boundary overlay on all axes (if CSV exists)
for N_try in [17, 16]
    csv_path = joinpath(PROJECT_ROOT_PS, "results", "linear_stability",
                        "phase_boundary_N$(N_try).csv")
    isfile(csv_path) || continue
    fields  = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
    κ_stab  = [parse(Float64, f[1]) for f in fields]
    f0_stab = [parse(Float64, f[2]) for f in fields]
    for ax in [ax_m, ax_s, (add_panel! for _ in 1:0)...]
        lines!(ax, f0_stab, κ_stab; color=:white, linewidth=2, linestyle=:dash)
    end
    for si in 1:ns
        ax_i = content(fig[1, si])
        lines!(ax_i, f0_stab, κ_stab; color=:white, linewidth=2, linestyle=:dash)
    end
    println("  Overlaid linear stability boundary from $csv_path")
    break
end

out_path = joinpath(PROJECT_ROOT_PS, "results", "phase_diagram_seeds.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
