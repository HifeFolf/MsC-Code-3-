# pca_dynamics.jl
# Modal PCA on curvature fluctuations, adapting the method of
# Isele-Holder et al. (Soft Matter 2016, 12, 8495) to the active ring.
#
# Method (per run):
#   dψ_i(t)  = turning angle at bead i at time t
#   dφ_i(t)  = dψ_i(t) - 2π/N          (deviation from equilibrium circle)
#   C_ij     = <dφ_i dφ_j>_t            (uncentered covariance of equilibrium deviations)
#   C ψ_k   = e_k ψ_k                   (eigenmodes, eigenvalues)
#   λ_k(t)   = ψ_k · dφ(t)              (modal amplitude time series)
#
# In the circular phase: dφ ≈ thermal noise → λ_k are uncorrelated Gaussians
#   → normalised radius r₁₂ = sqrt(λ₁²/e₁ + λ₂²/e₂) follows Rayleigh(1)
# In the kinked (buckled) phase: large mean deviation at passive bead
#   → λ₁ has non-zero mean → p(λ₁, λ₂) is an off-centre blob → non-Rayleigh
#
# χ² = Σ_bins (p_empirical(r₁₂) - Rayleigh(r₁₂))² Δr
# is the buckling criterion (0 = circular, large = buckled) — Fig. 4 in the paper.
#
# Outputs:
#   pca_dynamics_phase.png  — χ² and e₁/Σeₖ phase diagrams
#   pca_dynamics_modes.png  — p(λ₁, λ₂) distributions for 9 representative runs
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/pca_dynamics.jl
#   julia --project scripts/pca_dynamics.jl Results_N17_kinked

using CairoMakie, Statistics, LinearAlgebra, Printf, TOML, JLD2

const PROJECT_ROOT_DYN = joinpath(@__DIR__, "..")
const LAST_FRAMES      = round(Int, 20.0 * 60)   # last 20 s at 60 fps = 1200
const BIN_WIDTH        = 0.1                      # histogram bin width for χ²

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_DYN, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Core: load trajectory and run modal PCA ───────────────────────────────────
#
# Returns: (e, V, amp, χ², ef1)
#   e   = eigenvalues descending (= <λ_k²>)
#   V   = eigenvectors as columns (bead loadings per mode)
#   amp = modal amplitudes λ_k(t),  shape (N, n_frames)
#   χ²  = deviation of r₁₂ = sqrt(λ₁²/e₁ + λ₂²/e₂) from Rayleigh(1)
#   ef1 = e₁ / Σeₖ  (fraction of variance in dominant mode)
function modal_pca(traj_path, N)
    jldopen(traj_path, "r") do f
        pos   = f["positions"]          # 2 × N × n_frames
        n_tot = size(pos, 3)
        i_lo  = max(1, n_tot - LAST_FRAMES + 1)
        n_fr  = n_tot - i_lo + 1

        ψ  = Vector{Float64}(undef, N)
        dφ = Matrix{Float64}(undef, N, n_fr)   # equilibrium-subtracted curvature
        eq = 2π / N                              # equilibrium turning angle per bead

        for (fi, fr) in enumerate(i_lo:n_tot)
            r = view(pos, :, :, fr)
            for j in 1:N
                jp = (j == N) ? 1 : j + 1
                ψ[j] = atan(r[2, jp] - r[2, j], r[1, jp] - r[1, j])
            end
            for j in 1:N
                jp = (j == N) ? 1 : j + 1
                d  = ψ[jp] - ψ[j]
                d -= 2π * round(d / (2π))    # wrap to (-π, π]
                dφ[j, fi] = d - eq            # subtract equilibrium
            end
        end

        # Uncentered covariance of equilibrium deviations
        C = (dφ * dφ') / n_fr           # N × N

        # Eigendecompose in descending order
        F = eigen(Symmetric(C))
        e = max.(reverse(F.values),  0.0)          # eigenvalues descending
        V = reverse(F.vectors, dims = 2)            # eigenvectors as columns

        # Modal amplitudes: λ_k(t) = V[:, k]' * dφ(t)
        amp = V' * dφ                   # N × n_fr

        # ── χ² buckling test ────────────────────────────────────────────────
        # Normalised radius in (λ₁, λ₂) space
        σ₁  = sqrt(max(e[1], 1e-30));  σ₂ = sqrt(max(e[2], 1e-30))
        r12 = sqrt.((amp[1, :] ./ σ₁).^2 .+ (amp[2, :] ./ σ₂).^2)

        r_max  = max(maximum(r12), 4.0)
        n_bins = ceil(Int, r_max / BIN_WIDTH) + 2
        cnt    = zeros(n_bins)
        for r in r12
            bi = floor(Int, r / BIN_WIDTH) + 1
            1 <= bi <= n_bins && (cnt[bi] += 1)
        end
        p_emp  = cnt / (sum(cnt) * BIN_WIDTH)
        ctrs   = [(i - 0.5) * BIN_WIDTH for i in 1:n_bins]
        p_ray  = ctrs .* exp.(-ctrs.^2 ./ 2)       # Rayleigh(σ=1) PDF
        χ²val  = sum((p_emp .- p_ray).^2) * BIN_WIDTH

        return (e = e, V = V, amp = amp,
                χ² = χ²val,
                ef1 = e[1] / max(sum(e), 1e-30))
    end
end

# ── 2D histogram helper ────────────────────────────────────────────────────────
function hist2d(x, y; nbins = 60)
    hw  = 1.3 * max(maximum(abs.(x)), maximum(abs.(y)), 1e-6)
    edg = range(-hw, hw, length = nbins + 1)
    dx  = step(edg)
    H   = zeros(Int, nbins, nbins)
    for (xi, yi) in zip(x, y)
        ix = floor(Int, (xi + hw) / dx) + 1
        iy = floor(Int, (yi + hw) / dx) + 1
        1 <= ix <= nbins && 1 <= iy <= nbins && (H[ix, iy] += 1)
    end
    ctrs = collect(edg[1:nbins] .+ dx / 2)
    return ctrs, H, hw
end

# ── Scan all runs ──────────────────────────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

n_total = length(run_dirs)
println("Scanning $n_total runs from $subfolder…")
println("Computing modal PCA from traj.jld2 — may take a few minutes.\n")

κ_all   = Float64[];  f0_all  = Float64[]
χ²_all  = Float64[];  ef1_all = Float64[]
rd_all  = String[];   N_ring  = Ref(17)

for (i, rd) in enumerate(run_dirs)
    run_path   = joinpath(results_dir, rd)
    traj_file  = joinpath(run_path, "traj.jld2")
    param_file = joinpath(run_path, "params.toml")
    (isfile(traj_file) && isfile(param_file)) || continue

    ep = TOML.parsefile(param_file)["params"]
    κ  = Float64(ep["kappa"]);  f₀ = Float64(ep["f0"]);  N = Int(ep["N"])
    N_ring[] = N

    res = try
        modal_pca(traj_file, N)
    catch e
        @warn "Skipping $rd: $e"
        nothing
    end
    res === nothing && continue

    push!(κ_all, κ);  push!(f0_all, f₀)
    push!(χ²_all, res.χ²);  push!(ef1_all, res.ef1)
    push!(rd_all, rd)

    i % 200 == 0 && @printf("  %d / %d runs processed\n", i, n_total)
end

n_runs = length(κ_all)
println("Loaded $n_runs runs.")
isempty(κ_all) && error("No data — check that traj.jld2 files exist.")

N = N_ring[]
passive_idx = isodd(N) ? (N - 1) ÷ 2 + 1 : N ÷ 2 + 1
K_eq = 2π / N

# ── Build log-grid matrices ────────────────────────────────────────────────────
kappa_vals = sort(unique(κ_all));  f0_vals = sort(unique(f0_all))
nκ = length(kappa_vals);  nf = length(f0_vals)
@printf("Grid: %d κ × %d f₀\n", nκ, nf)

function to_matrix(vals)
    lk = Dict(zip(zip(κ_all, f0_all), vals))
    m  = fill(NaN, nκ, nf)
    for (ic, κ) in enumerate(kappa_vals), (ir, f) in enumerate(f0_vals)
        haskey(lk, (κ, f)) && (m[ic, ir] = lk[(κ, f)])
    end
    m
end

χ²_mat  = to_matrix(χ²_all)
ef1_mat = to_matrix(ef1_all)
χ²_max  = quantile(filter(isfinite, χ²_all), 0.98)
@printf("χ² range: 0 … %.4f (98th pctile)\n", χ²_max)

# ── Figure 1: Phase diagrams ───────────────────────────────────────────────────
println("\nBuilding phase diagram figure…")
fig1 = Figure(size = (1250, 600))

Label(fig1[0, 1:4],
      "Modal PCA (Isele-Holder 2016 method) — $subfolder   (N=$N, kBT=0.1)";
      fontsize = 13, tellwidth = false)

function log_ax(parent, pos, title_str)
    Axis(parent[pos...];
         title  = title_str, titlesize = 11,
         xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
         xscale = log10, yscale = log10,
         xticks = LogTicks(WilkinsonTicks(5)),
         yticks = LogTicks(WilkinsonTicks(5)))
end

ax_χ = log_ax(fig1, [1, 1],
    "Buckling criterion  χ²\n(0 = Gaussian/circular,  large = buckled)")
hm_χ = heatmap!(ax_χ, f0_vals, kappa_vals, χ²_mat';
                colormap = :hot, colorrange = (0.0, χ²_max), nan_color = :lightgrey)
Colorbar(fig1[1, 2], hm_χ; label = "χ²  (deviation from Rayleigh)", width = 14)

ax_e = log_ax(fig1, [1, 3],
    "Dominant mode fraction  e₁/Σeₖ\n(1 = single mode dominates = strongly buckled)")
hm_e = heatmap!(ax_e, f0_vals, kappa_vals, ef1_mat';
                colormap = :viridis, colorrange = (0.0, 1.0), nan_color = :lightgrey)
Colorbar(fig1[1, 4], hm_e; label = "e₁ / Σeₖ", width = 14)

# Linear stability boundary (optional)
for N_try in [N, 16, 17]
    csv_path = joinpath(PROJECT_ROOT_DYN, "results", "linear_stability",
                        "phase_boundary_N$(N_try).csv")
    isfile(csv_path) || continue
    rows    = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
    κ_stab  = [parse(Float64, r[1]) for r in rows]
    f0_stab = [parse(Float64, r[2]) for r in rows]
    for ax in [ax_χ, ax_e]
        mask = (f0_stab .>= f0_vals[1]) .& (f0_stab .<= f0_vals[end]) .&
               (κ_stab  .>= kappa_vals[1]) .& (κ_stab  .<= kappa_vals[end])
        any(mask) || continue
        lines!(ax, f0_stab[mask], κ_stab[mask];
               color = :white, linewidth = 2.5, linestyle = :dash)
    end
    println("  Stability boundary overlaid (N=$N_try).")
    break
end

save(joinpath(results_dir, "pca_dynamics_phase.png"), fig1, px_per_unit = 2)
println("Saved: pca_dynamics_phase.png")

# ── Figure 2: p(λ₁, λ₂) for 9 representative runs ────────────────────────────
# Pick 3 runs from each tertile of χ² (low / medium / high)
println("\nSelecting 9 representative runs for p(λ₁, λ₂) panels…")
sorted_idx = sortperm(χ²_all)
n_v   = length(sorted_idx)
picks = Int[]
for (t, qs) in enumerate([(0.1, 0.5, 0.9), (0.1, 0.5, 0.9), (0.1, 0.5, 0.9)])
    offset = (t - 1) * (n_v ÷ 3)
    width  = n_v ÷ 3
    for q in qs
        push!(picks, sorted_idx[clamp(offset + round(Int, q * width), 1, n_v)])
    end
end
picks = unique(picks)[1:min(9, end)]

fig2 = Figure(size = (1200, 1320))
Label(fig2[0, 1:3],
    "p(λ₁, λ₂) joint distributions — $subfolder\n" *
    "rows: low χ² (circular) → medium → high χ² (buckled)   " *
    "| cyan ellipse = Rayleigh reference (σ₁, σ₂)";
    fontsize = 11, tellwidth = false)

bead_clr = [i == passive_idx ? RGBf(0.8, 0.1, 0.1) : RGBf(0.27, 0.51, 0.71) for i in 1:N]

for (pi, idx) in enumerate(picks)
    row = (pi - 1) ÷ 3 + 1
    col = (pi - 1) % 3 + 1

    rd   = rd_all[idx]
    κ_r  = κ_all[idx];  f₀_r = f0_all[idx]
    χ²_r = χ²_all[idx]; ef1_r = ef1_all[idx]

    traj_file = joinpath(results_dir, rd, "traj.jld2")
    res = try modal_pca(traj_file, N)
    catch e; @warn "Skipping $rd: $e"; continue; end

    amp1 = res.amp[1, :];  amp2 = res.amp[2, :]
    σ₁   = sqrt(max(res.e[1], 1e-30));  σ₂ = sqrt(max(res.e[2], 1e-30))

    # 2D histogram of (λ₁, λ₂)
    ctrs, H, hw = hist2d(amp1, amp2)

    ax_p = Axis(fig2[row, col];
                title     = @sprintf("κ=%.1f  f₀=%.2f\nχ²=%.4f   e₁=%.1f%%",
                                     κ_r, f₀_r, χ²_r, 100 * ef1_r),
                titlesize = 9,
                xlabel    = "λ₁  (rad)",  ylabel = "λ₂  (rad)",
                aspect    = DataAspect())
    heatmap!(ax_p, ctrs, ctrs, H; colormap = :hot, lowclip = :black)

    # Rayleigh reference ellipse: semi-axes = 1σ₁, 1σ₂ in physical units
    θ = range(0, 2π, length = 200)
    lines!(ax_p, σ₁ .* cos.(θ), σ₂ .* sin.(θ);
           color = :cyan, linewidth = 1.5, linestyle = :dash)

    # Mode 1 loading profile as a small inset axis
    ax_m = Axis(fig2[row + 3, col];
                title     = @sprintf("ψ₁ loadings  κ=%.1f  f₀=%.2f", κ_r, f₀_r),
                titlesize = 9,
                xlabel    = "Bead  (passive=$(passive_idx))",
                ylabel    = "ψ₁[j]",
                xticks    = 1:N)
    barplot!(ax_m, 1:N, res.V[:, 1]; color = bead_clr, strokewidth = 0)
    hlines!(ax_m, [0.0]; color = :black, linewidth = 0.8)
    vlines!(ax_m, [passive_idx]; color = :crimson, linewidth = 1.5, linestyle = :dash)
end

save(joinpath(results_dir, "pca_dynamics_modes.png"), fig2, px_per_unit = 2)
println("Saved: pca_dynamics_modes.png")
println("\nDone.")
