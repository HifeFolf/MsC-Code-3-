# pca_structures.jl
# PCA on the time-averaged turning-angle sequence dψ across all runs.
#
# Each run contributes one N-dimensional point: the mean dψ vector (signed
# turning angle at each bead) averaged over the last 20 s of simulation.
# PCA on this ensemble finds the dominant axes of shape variation.
#
# PCA formulation:
#   X  (n_runs × N)  — rows = mean dψ per run
#   X_c = X - mean(X, dims=1)   (mean-centred; no per-bead scaling so that
#                                 high-variance beads are not suppressed)
#   SVD: X_c = U S Vᵀ
#   Scores   = U S   (n_runs × N)  — run coordinates in PC space
#   Loadings = V     (N × N)        — bead contributions to each PC
#
# Output figure  results/<subfolder>/pca_structures.png:
#   Row 1 — PC1 vs PC2 scatter coloured by asphericity b | f₀ | κ
#   Row 2 — Scree plot | PC1 loading profile | PC2 loading profile
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/pca_structures.jl
#   julia --project scripts/pca_structures.jl Results_N17_kinked
#   julia --project scripts/pca_structures.jl Results_N17_circle 1 750

using CairoMakie, Statistics, LinearAlgebra, Printf, TOML, JLD2

const PROJECT_ROOT_PCA = joinpath(@__DIR__, "..")
const LAST_FRAMES = round(Int, 20.0 * 60)   # last 20 s at 60 fps = 1200

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_quintkinked"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_PCA, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Mean dψ vector for one run ────────────────────────────────────────────────
# Computes the time-averaged signed turning angle at each bead.
# dψ[j] = wrap(ψ[j+1] − ψ[j]), where ψ[j] = direction of bond j → j+1.
function mean_dpsi_vec(traj_path, N)
    jldopen(traj_path, "r") do f
        pos   = f["positions"]              # 2 × N × n_frames
        n_tot = size(pos, 3)
        i_lo  = max(1, n_tot - LAST_FRAMES + 1)

        ψ   = Vector{Float64}(undef, N)
        acc = zeros(N)
        cnt = 0

        for fr in i_lo:n_tot
            r = view(pos, :, :, fr)
            for j in 1:N
                jp = (j == N) ? 1 : j + 1
                ψ[j] = atan(r[2, jp] - r[2, j], r[1, jp] - r[1, j])
            end
            for j in 1:N
                jp = (j == N) ? 1 : j + 1
                d  = ψ[jp] - ψ[j]
                d -= 2π * round(d / (2π))
                acc[j] += d
            end
            cnt += 1
        end
        cnt > 0 ? acc ./ cnt : fill(NaN, N)
    end
end

# ── Scan runs ─────────────────────────────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

n_total = length(run_dirs)
println("Scanning $n_total runs from $subfolder…")
println("Computing mean dψ from traj.jld2 — may take a few minutes.\n")

κ_all    = Float64[]
f0_all   = Float64[]
b_all    = Float64[]
dpsi_all = Vector{Vector{Float64}}()
N_ring   = Ref(17)

for (i, rd) in enumerate(run_dirs)
    run_path   = joinpath(results_dir, rd)
    traj_file  = joinpath(run_path, "traj.jld2")
    param_file = joinpath(run_path, "params.toml")
    obs_file   = joinpath(run_path, "observables.jld2")
    (isfile(traj_file) && isfile(param_file)) || continue

    ep = TOML.parsefile(param_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])
    N  = Int(ep["N"])
    N_ring[] = N

    dpsi = try
        mean_dpsi_vec(traj_file, N)
    catch e
        @warn "Skipping $rd: $e"
        fill(NaN, N)
    end
    any(isnan, dpsi) && continue

    # asphericity b from observables (NaN if unavailable)
    b_mean = NaN
    if isfile(obs_file)
        b_mean = jldopen(obs_file, "r") do f
            haskey(f, "b") || return NaN
            bv = f["b"]; n = length(bv)
            mean(bv[max(1, n - LAST_FRAMES + 1):n])
        end
    end

    push!(κ_all, κ); push!(f0_all, f₀); push!(b_all, b_mean)
    push!(dpsi_all, dpsi)

    i % 200 == 0 && @printf("  %d / %d runs processed\n", i, n_total)
end

n_runs = length(dpsi_all)
println("Loaded $n_runs runs.")
n_runs < 3 && error("Need at least 3 runs for PCA.")

N = N_ring[]
passive_idx = isodd(N) ? (N - 1) ÷ 2 + 1 : N ÷ 2 + 1

# ── PCA via SVD ───────────────────────────────────────────────────────────────
X   = reduce(hcat, dpsi_all)'       # n_runs × N
μ   = mean(X, dims = 1)             # 1 × N  (global mean shape)
X_c = X .- μ                        # mean-centred

println("\nRunning SVD  ($n_runs runs × $N beads)…")
F = svd(X_c)

# Scores: coordinates of each run in PC space
scores   = F.U .* F.S'              # n_runs × rank
loadings = F.V                       # N × rank  (bead loadings per PC)

var_exp  = (F.S .^ 2) ./ sum(F.S .^ 2)
cum_var  = cumsum(var_exp)

n_show = min(10, N, n_runs - 1)
println("\nExplained variance (first $n_show PCs):")
for k in 1:n_show
    @printf("  PC%2d: %5.1f%%   cumulative: %5.1f%%\n",
            k, 100 * var_exp[k], 100 * cum_var[k])
end

pc1 = scores[:, 1]
pc2 = scores[:, 2]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (1500, 820))

Label(fig[0, 1:6],
      "PCA on mean turning-angle sequence  dψ — $subfolder   (N=$N, kBT=0.1)   " *
      "| PC1=$(round(100*var_exp[1]; digits=1))%   " *
      "PC2=$(round(100*var_exp[2]; digits=1))%   " *
      "PC3=$(round(100*var_exp[3]; digits=1))%   " *
      "Σ(1:3)=$(round(100*cum_var[3]; digits=1))%";
      fontsize = 13, tellwidth = false)

function scatter_ax(col, title_str)
    Axis(fig[1, col];
         title    = title_str, titlesize = 11,
         xlabel   = "PC1  ($(round(100*var_exp[1]; digits=1))%)",
         ylabel   = "PC2  ($(round(100*var_exp[2]; digits=1))%)")
end

# Panel A — coloured by asphericity b
b_finite = filter(!isnan, b_all)
ax_b = scatter_ax(1, "Coloured by asphericity  b")
if !isempty(b_finite)
    sc_b = scatter!(ax_b, pc1, pc2;
                    color      = b_all,
                    colormap   = Reverse(:RdYlBu),
                    colorrange = (0.0, min(1.0, maximum(b_finite))),
                    nan_color  = :lightgrey,
                    markersize = 6, strokewidth = 0)
    Colorbar(fig[1, 2], sc_b; label = "mean b  (0=circular, 1=rod)", width = 14)
else
    scatter!(ax_b, pc1, pc2; color = :steelblue, markersize = 6, strokewidth = 0)
    Label(fig[1, 2], "b\nnot\navail."; tellheight = false)
end

# Panel B — coloured by log10(f₀)
ax_f = scatter_ax(3, "Coloured by active force  f₀")
sc_f = scatter!(ax_f, pc1, pc2;
                color      = log10.(f0_all),
                colormap   = :plasma,
                markersize = 6, strokewidth = 0)
Colorbar(fig[1, 4], sc_f; label = "log₁₀(f₀)", width = 14)

# Panel C — coloured by log10(κ)
ax_k = scatter_ax(5, "Coloured by bending stiffness  κ")
sc_k = scatter!(ax_k, pc1, pc2;
                color      = log10.(κ_all),
                colormap   = :viridis,
                markersize = 6, strokewidth = 0)
Colorbar(fig[1, 6], sc_k; label = "log₁₀(κ)", width = 14)

# ── Bottom row ────────────────────────────────────────────────────────────────
bead_clr = [i == passive_idx ? RGBf(0.8, 0.1, 0.1) : RGBf(0.27, 0.51, 0.71)
            for i in 1:N]

# Panel D — Scree plot
ax_sc = Axis(fig[2, 1:2];
             title  = "Scree — explained variance",
             xlabel = "Principal component",
             ylabel = "% variance",
             xticks = 1:n_show)
barplot!(ax_sc, 1:n_show, 100 .* var_exp[1:n_show];
         color = :steelblue, strokewidth = 0)
lines!(ax_sc, 1:n_show, 100 .* cum_var[1:n_show];
       color = :crimson, linewidth = 2.5, label = "Cumulative")
scatter!(ax_sc, 1:n_show, 100 .* cum_var[1:n_show];
         color = :crimson, markersize = 8)
axislegend(ax_sc; position = :rc, framevisible = false)

# Panel E — PC1 loading profile
ax_l1 = Axis(fig[2, 3:4];
             title  = "PC1 loadings  ($(round(100*var_exp[1]; digits=1))% variance)",
             xlabel = "Bead index",
             ylabel = "Loading",
             xticks = 1:N)
barplot!(ax_l1, 1:N, loadings[:, 1]; color = bead_clr, strokewidth = 0)
hlines!(ax_l1, [0.0]; color = :black, linewidth = 0.8)
vlines!(ax_l1, [passive_idx]; color = :crimson, linewidth = 1.5,
        linestyle = :dash, label = "Passive ($(passive_idx))")
axislegend(ax_l1; position = :rt, framevisible = false)

# Panel F — PC2 loading profile
ax_l2 = Axis(fig[2, 5:6];
             title  = "PC2 loadings  ($(round(100*var_exp[2]; digits=1))% variance)",
             xlabel = "Bead index",
             ylabel = "Loading",
             xticks = 1:N)
barplot!(ax_l2, 1:N, loadings[:, 2]; color = bead_clr, strokewidth = 0)
hlines!(ax_l2, [0.0]; color = :black, linewidth = 0.8)
vlines!(ax_l2, [passive_idx]; color = :crimson, linewidth = 1.5,
        linestyle = :dash, label = "Passive ($(passive_idx))")
axislegend(ax_l2; position = :rt, framevisible = false)

# Make colorbar columns narrow
for col in [2, 4, 6]
    colsize!(fig.layout, col, Fixed(40))
end

out_path = joinpath(results_dir, "pca_structures.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
