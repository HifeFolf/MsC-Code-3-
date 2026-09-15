# phase_diagram_seeds_aggregate.jl
# Seed-AGGREGATED phase diagram for the N=17 circle sweep.
#
# Unlike phase_diagram_seeds.jl (which shows one panel per seed side-by-side),
# this script pools ALL circle seed folders into a SINGLE aggregated diagram and
# plots the three main order parameters as κ × f₀ heatmaps:
#
#   1. ⟨b⟩            — mean asphericity                      (viridis, 0…1)
#   2. ⟨Kconn⟩        — mean passive-junction turning angle   (RdBu, centred 2π/N)
#   3. ⟨std(Kconn)⟩   — junction fluctuation amplitude        (viridis)
#
# Aggregation, per (κ, f₀) cell:
#   • Each seed's run is reduced to (b_ss, mean_Kconn, std_Kconn) over its
#     post-burn-in steady-state window.
#   • The three panels are the MEAN of those per-run values across the seeds.
#     ⟨std(Kconn)⟩ is therefore the seed-averaged TEMPORAL std within each run
#     (the junction-fluctuation order parameter of phase_diagram_kconn.jl), NOT
#     the std across seeds.
#
# Steady-state window: drop every frame with t < BURN_IN_T sim-units, selected
# from the `times` vector — never a frame count. The stored `framerate` is a
# movie-render setting (60 fps) and does NOT convert sim-time to frames. Runs are
# 7.5 sim-units / 1500 frames, so BURN_IN_T = 2.5 keeps the last 5 sim-units.
# This matches reduce_kconn.jl exactly.
#
# Seeds: auto-discovered — every folder matching Results_N17_circle[_seedNN]
# (the trailing-digit "Results_N17_circle1" is deliberately excluded).
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_seeds_aggregate.jl
#
# Output:
#   results/phase_diagram_seeds_aggregate.png

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_AGG = joinpath(@__DIR__, "..")
const RESULTS_DIR      = joinpath(PROJECT_ROOT_AGG, "results")

const BURN_IN_T      = 2.5                                   # sim-units (see header)
const SEED_FOLDER_RE = r"^Results_N17_circle(_seed\d+)?$"    # excludes "…circle1"
const FL_C           = 75.0                                  # Fl_c guide: f0 = FL_C·κ
const STD_CLIP_Q     = 0.98                                  # std panel upper clip (heavy tail)

seed_of(folder) = (m = match(r"_seed(\d+)$", folder); m === nothing ? "42" : m.captures[1])

# ── Reduce one seed folder → Dict{(κ,f₀) => (b_ss, mean_Kconn, std_Kconn)} ─────
function load_folder(sub)
    dir = joinpath(RESULTS_DIR, sub)
    lookup = Dict{Tuple{Float64,Float64}, NTuple{3,Float64}}()
    N = 0
    run_dirs = filter(d -> occursin(r"^run_\d+$", d), readdir(dir))
    for rd in run_dirs
        rp       = joinpath(dir, rd)
        obs_file = joinpath(rp, "observables.jld2")
        pfile    = joinpath(rp, "params.toml")
        (isfile(obs_file) && isfile(pfile)) || continue

        ep = TOML.parsefile(pfile)["params"]
        κ  = Float64(ep["kappa"]); f₀ = Float64(ep["f0"]); N = Int(ep["N"])

        data = jldopen(obs_file, "r") do f
            (haskey(f, "times") && haskey(f, "b") && haskey(f, "Kconn")) || return nothing
            (f["times"], f["b"], f["Kconn"])
        end
        data === nothing && continue
        times, b, K = data

        keep = findall(>=(BURN_IN_T), times)      # steady-state window by sim-time
        length(keep) < 2 && continue
        @views lookup[(κ, f₀)] = (mean(b[keep]), mean(K[keep]), std(K[keep]))
    end
    @printf("  %-32s seed %-4s  %4d cells\n", sub, seed_of(sub), length(lookup))
    return lookup, N
end

# ── Discover + load all circle seed folders ───────────────────────────────────
seed_folders = sort(filter(d -> occursin(SEED_FOLDER_RE, d) && isdir(joinpath(RESULTS_DIR, d)),
                           readdir(RESULTS_DIR)))
isempty(seed_folders) && error("No Results_N17_circle[_seedNN] folders found in $RESULTS_DIR")

println("Aggregating $(length(seed_folders)) seed folder(s):")
loaded  = [load_folder(sf) for sf in seed_folders]
lookups = first.(loaded)
Ns      = unique(filter(>(0), last.(loaded)))
length(Ns) <= 1 || @warn "Seed folders disagree on N" Ns
N     = isempty(Ns) ? 17 : Ns[1]
seeds = seed_of.(seed_folders)

# ── Shared grid (union of sampled (κ, f₀) across seeds) ────────────────────────
all_κ = sort(unique(k for lk in lookups for (k, _) in keys(lk)))
all_f = sort(unique(f for lk in lookups for (_, f) in keys(lk)))
nκ, nf, ns = length(all_κ), length(all_f), length(lookups)
@printf("\nGrid: %d κ × %d f₀   |   %d seeds   |   N=%d\n", nκ, nf, ns, N)

b_mean = fill(NaN, nκ, nf)      # ⟨b⟩ across seeds
k_mean = fill(NaN, nκ, nf)      # ⟨Kconn⟩ across seeds (rad)
k_std  = fill(NaN, nκ, nf)      # ⟨temporal std(Kconn)⟩ across seeds (rad)
n_cell = zeros(Int, nκ, nf)     # seeds contributing to each cell
for (ik, κ) in enumerate(all_κ), (jf, f) in enumerate(all_f)
    bs = Float64[]; kms = Float64[]; kss = Float64[]
    for lk in lookups
        haskey(lk, (κ, f)) || continue
        (b, km, ks) = lk[(κ, f)]
        push!(bs, b); push!(kms, km); push!(kss, ks)
    end
    isempty(bs) && continue
    n_cell[ik, jf] = length(bs)
    b_mean[ik, jf] = mean(bs)
    k_mean[ik, jf] = mean(kms)
    k_std[ik, jf]  = mean(kss)
end

n_full = count(>(0), n_cell)
@printf("Filled cells: %d / %d   (seeds per cell: %d…%d)\n",
        n_full, nκ * nf,
        (f = filter(>(0), vec(n_cell)); isempty(f) ? 0 : minimum(f)),
        maximum(n_cell))

k_mean_deg = rad2deg.(k_mean)   # NaN stays NaN
k_std_deg  = rad2deg.(k_std)

# ── Colour normalisations ─────────────────────────────────────────────────────
K_eq  = 360.0 / N                                           # equilibrium angle (°)
allm  = filter(!isnan, vec(k_mean_deg))
half  = (isempty(allm) ? 20.0 : maximum(abs.(allm .- K_eq))) * 1.02
clo, chi = K_eq - half, K_eq + half                         # diverging, centred on 2π/N
alls    = filter(!isnan, vec(k_std_deg))
std_max = isempty(alls) ? 1.0 : quantile(alls, STD_CLIP_Q)
@printf("⟨Kconn⟩ range: %.1f°…%.1f°  (white = 2π/N = %.2f°)\n", clo, chi, K_eq)
@printf("std range: 0°…%.2f°  (clipped at p%d)\n", std_max, round(Int, 100 * STD_CLIP_Q))

# ── Axis scale / ticks: log only when the sampled axis is log-spaced ───────────
function is_log_sampled(v)
    length(v) < 3 && return false
    any(<=(0), v) && return false
    cv(x) = (m = mean(x); m == 0 ? Inf : std(x) / abs(m))
    return cv(diff(log.(v))) < cv(diff(v))
end
scale_of(islog) = islog ? log10 : identity
ticks_of(islog) = islog ? LogTicks(WilkinsonTicks(5)) : Makie.automatic
flog, klog = is_log_sampled(all_f), is_log_sampled(all_κ)

# ── Stability boundary overlay: measured curve if cached, else Fl_c guide ──────
function add_boundary!(ax)
    csv = joinpath(RESULTS_DIR, "linear_stability", "phase_boundary_N$(N).csv")
    if isfile(csv)
        rows = [split(r, ",") for r in readlines(csv)[2:end] if !isempty(r)]
        κb = [parse(Float64, r[1]) for r in rows]      # col 1 = κ
        fb = [parse(Float64, r[2]) for r in rows]      # col 2 = f0
        lines!(ax, fb, κb; color = :black, linewidth = 2, linestyle = :dash)
    else
        κlo = max(minimum(all_κ), minimum(all_f) / FL_C)
        κhi = min(maximum(all_κ), maximum(all_f) / FL_C)
        κlo < κhi || return
        κl = range(κlo, κhi; length = 64)
        lines!(ax, FL_C .* κl, collect(κl); color = :black, linewidth = 2, linestyle = :dash)
    end
end

# ── Figure: three panels in one row, each with its own colorbar ───────────────
fig = Figure(size = (1650, 560))
Label(fig[0, 1:6],
      "N=$N circle IC — seed-aggregated phase diagram   " *
      "($(ns) seeds: $(join(seeds, ", "));  steady state t ≥ $(BURN_IN_T), last 5 sim-units)";
      fontsize = 15, font = :bold, tellwidth = false)

function panel!(col, mat, title_str; cmap, crange, cbar_label)
    ax = Axis(fig[1, 2col - 1];
              title  = title_str,
              xlabel = "f₀  (active force)", ylabel = "κ  (bending stiffness)",
              xscale = scale_of(flog), yscale = scale_of(klog),
              xticks = ticks_of(flog), yticks = ticks_of(klog))
    hm = heatmap!(ax, all_f, all_κ, mat';
                  colormap = cmap, colorrange = crange, nan_color = :white)
    Colorbar(fig[1, 2col], hm; label = cbar_label, width = 14)
    add_boundary!(ax)
    return ax
end

panel!(1, b_mean, "Asphericity  ⟨b⟩";
       cmap = :viridis, crange = (0.0, 1.0), cbar_label = "⟨b⟩")
panel!(2, k_mean_deg, "Junction angle  ⟨Kconn⟩   (white = 2π/N = $(round(K_eq, digits=1))°)";
       cmap = :RdBu, crange = (clo, chi), cbar_label = "⟨Kconn⟩  (°)")
panel!(3, k_std_deg, "Junction fluctuation  ⟨std(Kconn)⟩";
       cmap = :viridis, crange = (0.0, max(std_max, 1e-3)), cbar_label = "std(Kconn)  (°)")

colgap!(fig.layout, 10)

out_path = joinpath(RESULTS_DIR, "phase_diagram_seeds_aggregate.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
