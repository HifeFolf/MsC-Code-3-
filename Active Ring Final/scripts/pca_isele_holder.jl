# pca_isele_holder.jl
# Ring modal PCA presented in the style of Isele-Holder et al.
# (Soft Matter 2016, 12, 8495) Figures 3 and 4.
#
# Phase classification (analogous to paper's elongated / beat / rotation):
#   :circular  χ² < CHI2_LO  — Gaussian blob at origin, ring stays circular
#   :kinked    CHI2_LO ≤ χ² < CHI2_HI — off-centre blob, single stable kink
#   :dynamic   χ² ≥ CHI2_HI  — non-Gaussian / bimodal, switching or oscillating
#
# For each phase, one representative run is chosen and shown as:
#   • p(λ₁, λ₂) joint distribution  (cf. paper Fig. 3a–d)
#   • ψ₁ dominant mode shape         (cf. paper Fig. 3e)
#   • λ₁(t), λ₂(t) time series       (cf. paper Fig. 3f)
# Phase diagram with symbols follows paper Fig. 4.
#
# Output:
#   pca_paper_modes.png   — characterisation panels for 3 representative runs
#   pca_paper_phases.png  — phase diagram with □ / ● / ▲ symbols
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/pca_isele_holder.jl
#   julia --project scripts/pca_isele_holder.jl Results_N17_kinked

using CairoMakie, Statistics, LinearAlgebra, Printf, TOML, JLD2

const PROJECT_ROOT_IH = joinpath(@__DIR__, "..")
const LAST_FRAMES     = round(Int, 20.0 * 60)   # last 20 s at 60 fps = 1200
const BIN_WIDTH       = 0.1
const CHI2_LO         = 0.5    # circular → kinked boundary
const CHI2_HI         = 2.0    # kinked  → dynamic boundary

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_kinked"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_IH, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Modal PCA on equilibrium-subtracted curvature ─────────────────────────────
function modal_pca(traj_path, N)
    jldopen(traj_path, "r") do f
        pos   = f["positions"]
        n_tot = size(pos, 3)
        i_lo  = max(1, n_tot - LAST_FRAMES + 1)
        n_fr  = n_tot - i_lo + 1

        ψ  = Vector{Float64}(undef, N)
        dφ = Matrix{Float64}(undef, N, n_fr)   # dψ - 2π/N  (deviation from circle)
        eq = 2π / N

        for (fi, fr) in enumerate(i_lo:n_tot)
            r = view(pos, :, :, fr)
            for j in 1:N
                jp = (j == N) ? 1 : j + 1
                ψ[j] = atan(r[2, jp] - r[2, j], r[1, jp] - r[1, j])
            end
            for j in 1:N
                jp = (j == N) ? 1 : j + 1
                d  = ψ[jp] - ψ[j];  d -= 2π * round(d / (2π))
                dφ[j, fi] = d - eq
            end
        end

        C = (dφ * dφ') / n_fr
        F = eigen(Symmetric(C))
        e = max.(reverse(F.values), 0.0)
        V = reverse(F.vectors, dims = 2)
        amp = V' * dφ   # N × n_fr  (modal amplitudes over time)

        # χ²: deviation of normalised radius r₁₂ from Rayleigh(1)
        σ₁  = sqrt(max(e[1], 1e-30));  σ₂ = sqrt(max(e[2], 1e-30))
        r12 = sqrt.((amp[1, :] ./ σ₁).^2 .+ (amp[2, :] ./ σ₂).^2)
        r_max  = max(maximum(r12), 4.0)
        n_bins = ceil(Int, r_max / BIN_WIDTH) + 2
        cnt    = zeros(n_bins)
        for r in r12; bi = floor(Int, r / BIN_WIDTH) + 1
            1 <= bi <= n_bins && (cnt[bi] += 1)
        end
        p_emp = cnt / (sum(cnt) * BIN_WIDTH)
        ctrs  = [(i - 0.5) * BIN_WIDTH for i in 1:n_bins]
        p_ray = ctrs .* exp.(-ctrs.^2 ./ 2)
        χ²val = sum((p_emp .- p_ray).^2) * BIN_WIDTH

        return (e=e, V=V, amp=amp, χ²=χ²val, ef1=e[1]/max(sum(e), 1e-30))
    end
end

# ── 2D histogram ────────────────────────────────────────────────────────────────
function hist2d(x, y; nbins=60)
    hw  = 1.3 * max(maximum(abs.(x)), maximum(abs.(y)), 1e-6)
    edg = range(-hw, hw, length = nbins + 1);  dx = step(edg)
    H   = zeros(Int, nbins, nbins)
    for (xi, yi) in zip(x, y)
        ix = floor(Int, (xi + hw) / dx) + 1
        iy = floor(Int, (yi + hw) / dx) + 1
        1 <= ix <= nbins && 1 <= iy <= nbins && (H[ix, iy] += 1)
    end
    return collect(edg[1:nbins] .+ dx / 2), H
end

# ── Phase classification ────────────────────────────────────────────────────────
classify(χ²) = χ² < CHI2_LO ? :circular : χ² >= CHI2_HI ? :dynamic : :kinked

const PHASE_NAME   = Dict(:circular => "Circular",
                           :kinked   => "Stable kink",
                           :dynamic  => "Dynamic / switching")
const PHASE_COLOR  = Dict(:circular => RGBf(0.27, 0.51, 0.71),
                           :kinked   => RGBf(0.85, 0.40, 0.10),
                           :dynamic  => RGBf(0.75, 0.10, 0.15))
const PHASE_MARKER = Dict(:circular => :rect,
                           :kinked   => :circle,
                           :dynamic  => :utriangle)

# ── First pass: scalars for all runs ───────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1]);  return id_start <= id <= id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder…")
println("Computing modal PCA from traj.jld2 — may take a few minutes.\n")

κ_all   = Float64[];  f0_all  = Float64[]
χ²_all  = Float64[];  ef1_all = Float64[]
rd_all  = String[];   N_ring  = Ref(17)

for (i, rd) in enumerate(run_dirs)
    rp = joinpath(results_dir, rd)
    tf = joinpath(rp, "traj.jld2");  pf = joinpath(rp, "params.toml")
    (isfile(tf) && isfile(pf)) || continue
    ep = TOML.parsefile(pf)["params"]
    κ = Float64(ep["kappa"]);  f₀ = Float64(ep["f0"]);  N = Int(ep["N"])
    N_ring[] = N
    res = try modal_pca(tf, N) catch e; @warn "Skipping $rd: $e"; nothing end
    res === nothing && continue
    push!(κ_all, κ);  push!(f0_all, f₀)
    push!(χ²_all, res.χ²);  push!(ef1_all, res.ef1);  push!(rd_all, rd)
    i % 200 == 0 && @printf("  %d / %d runs\n", i, length(run_dirs))
end

n_runs = length(κ_all)
println("Loaded $n_runs runs.")
isempty(κ_all) && error("No data.")

N           = N_ring[]
passive_idx = isodd(N) ? (N - 1) ÷ 2 + 1 : N ÷ 2 + 1
phases      = classify.(χ²_all)

for ph in [:circular, :kinked, :dynamic]
    @printf("  %-20s %d runs\n", PHASE_NAME[ph], count(==(ph), phases))
end

# ── Select 3 representative runs (one per phase) ───────────────────────────────
function pick_run(ph_target, χ²_all, ef1_all, phases; prefer_extreme=:none)
    cands = findall(==(ph_target), phases)
    isempty(cands) && return 1
    if prefer_extreme == :min_chi2
        return cands[argmin(χ²_all[cands])]
    elseif prefer_extreme == :max_chi2
        return cands[argmax(χ²_all[cands])]
    else  # :kinked — pick most unambiguous (highest ef1)
        return cands[argmax(ef1_all[cands])]
    end
end

circ_idx = pick_run(:circular, χ²_all, ef1_all, phases; prefer_extreme=:min_chi2)
kink_idx = pick_run(:kinked,   χ²_all, ef1_all, phases)
dyn_idx  = pick_run(:dynamic,  χ²_all, ef1_all, phases; prefer_extreme=:max_chi2)

rep_order = [(circ_idx, :circular), (dyn_idx, :dynamic), (kink_idx, :kinked)]

println("\nRepresentative runs selected:")
for (idx, ph) in rep_order
    @printf("  %-22s %s  κ=%.2f  f₀=%.2f  χ²=%.4f  e₁=%.1f%%\n",
            PHASE_NAME[ph], rd_all[idx],
            κ_all[idx], f0_all[idx], χ²_all[idx], 100 * ef1_all[idx])
end

# ── Second pass: full PCA for the 3 representative runs ────────────────────────
println("\nLoading full modal time series for representative runs…")
rep_data = []
for (idx, ph) in rep_order
    tf  = joinpath(results_dir, rd_all[idx], "traj.jld2")
    res = modal_pca(tf, N)
    push!(rep_data, (phase=ph, κ=κ_all[idx], f₀=f0_all[idx],
                     χ²=χ²_all[idx], ef1=ef1_all[idx],
                     e=res.e, V=res.V, amp=res.amp))
end

# ── Figure 1: characterisation panels (cf. Isele-Holder Fig. 3) ───────────────
println("\nBuilding mode characterisation figure…")

fig1 = Figure(size = (1420, 1000))

Label(fig1[0, 1:3],
      "Active ring modal PCA — $subfolder   (N=$N, kBT=0.1)" *
      " ";
      fontsize = 12, tellwidth = false)

bead_clr = [i == passive_idx ? RGBf(0.8, 0.1, 0.1) : RGBf(0.27, 0.51, 0.71) for i in 1:N]

for (col, dat) in enumerate(rep_data)
    ph     = dat.phase
    ph_clr = PHASE_COLOR[ph]
    amp1   = dat.amp[1, :]
    amp2   = dat.amp[2, :]
    σ₁     = sqrt(max(dat.e[1], 1e-30))
    σ₂     = sqrt(max(dat.e[2], 1e-30))
    n_fr   = size(dat.amp, 2)
    t_ax   = range(0.0, 20.0, length = n_fr)

    # ── Row 1: p(λ₁, λ₂) joint distribution ────────────────────────────────
    ctrs, H = hist2d(amp1, amp2)
    ax_p = Axis(fig1[1, col];
                title     = @sprintf("%s\nκ = %.1f   f₀ = %.2f   χ² = %.3f   e₁ = %.1f%%",
                                     PHASE_NAME[ph], dat.κ, dat.f₀, dat.χ², 100 * dat.ef1),
                titlesize = 10,
                xlabel    = "λ₁  (rad)",
                ylabel    = "λ₂  (rad)",
                aspect    = DataAspect())
    heatmap!(ax_p, ctrs, ctrs, H; colormap = :hot, lowclip = :black)
    # Rayleigh reference ellipse: semi-axes σ₁, σ₂
    θ = range(0, 2π, length = 200)
    lines!(ax_p, σ₁ .* cos.(θ), σ₂ .* sin.(θ);
           color = :cyan, linewidth = 1.5, linestyle = :dash, label = "Rayleigh σ")
    axislegend(ax_p; position = :rt, backgroundcolor = (:black, 0.5),
               labelcolor = :cyan, labelsize = 8, framevisible = false)

    # ── Row 2: ψ₁ dominant mode shape (cf. Fig. 3e) ─────────────────────────
    ax_m = Axis(fig1[2, col];
                title     = "Dominant mode  ψ₁  ($(round(100*dat.ef1, digits=1))% of variance)",
                titlesize = 10,
                xlabel    = "Bead index",
                ylabel    = "ψ₁ loading",
                xticks    = 1:N)
    barplot!(ax_m, 1:N, dat.V[:, 1]; color = bead_clr, strokewidth = 0)
    hlines!(ax_m, [0.0]; color = :black, linewidth = 0.8)
    vlines!(ax_m, [passive_idx]; color = :crimson, linewidth = 1.5,
            linestyle = :dash, label = "Passive ($passive_idx)")
    axislegend(ax_m; position = :rt, framevisible = false, labelsize = 8)

    # ── Row 3: λ₁(t), λ₂(t) time series (cf. Fig. 3f) ───────────────────────
    ax_t = Axis(fig1[3, col];
                title     = "Modal amplitude time series",
                titlesize = 10,
                xlabel    = "Time  (s)",
                ylabel    = "λ  (rad)")
    lines!(ax_t, t_ax, amp1; color = ph_clr,   linewidth = 1.0, label = "λ₁")
    lines!(ax_t, t_ax, amp2; color = :grey60,  linewidth = 0.7, label = "λ₂")
    hlines!(ax_t, [0.0]; color = :black, linewidth = 0.6)
    axislegend(ax_t; position = :rt, framevisible = false, labelsize = 8)
end

rowsize!(fig1.layout, 1, Relative(0.4))   # more height for density panels

save(joinpath(results_dir, "pca_paper_modes.png"), fig1, px_per_unit = 2)
println("Saved: pca_paper_modes.png")

# ── Figure 2: phase diagram with symbols (cf. Isele-Holder Fig. 4) ────────────
println("Building phase diagram figure…")

fig2 = Figure(size = (820, 720))
Label(fig2[0, 1],
      "Phase diagram — $subfolder   (N=$N, kBT=0.1)" *
      "   cf. Isele-Holder et al. Soft Matter 2016, Fig. 4";
      fontsize = 12, tellwidth = false)

ax2 = Axis(fig2[1, 1];
           xlabel = "Active force  f₀",
           ylabel = "Bending stiffness  κ",
           xscale = log10, yscale = log10,
           xticks = LogTicks(WilkinsonTicks(6)),
           yticks = LogTicks(WilkinsonTicks(6)))

for ph in [:circular, :kinked, :dynamic]
    idxs = findall(==(ph), phases)
    isempty(idxs) && continue
    scatter!(ax2, f0_all[idxs], κ_all[idxs];
             marker      = PHASE_MARKER[ph],
             color       = (PHASE_COLOR[ph], 0.80),
             markersize  = 9,
             strokewidth = 0.6,
             strokecolor = :black,
             label       = PHASE_NAME[ph] * " (χ²∈" *
                           (ph == :circular ? "[0, $(CHI2_LO))" :
                            ph == :kinked   ? "[$(CHI2_LO), $(CHI2_HI))" :
                                              "[$(CHI2_HI), ∞))"))
end

# Mark the 3 representative runs with larger outlined symbols
for (idx, ph) in rep_order
    scatter!(ax2, [f0_all[idx]], [κ_all[idx]];
             marker      = PHASE_MARKER[ph],
             color       = PHASE_COLOR[ph],
             markersize  = 16,
             strokewidth = 2.0,
             strokecolor = :white)
end

# Threshold annotation
text!(ax2, minimum(f0_all) * 1.2, CHI2_LO;
      text = "χ²=$(CHI2_LO)", fontsize = 8, color = :grey40)

axislegend(ax2; position = :lb, backgroundcolor = (:white, 0.85),
           framevisible = true, labelsize = 10)

# Linear stability boundary
for N_try in [N, 16, 17]
    csv_path = joinpath(PROJECT_ROOT_IH, "results", "linear_stability",
                        "phase_boundary_N$(N_try).csv")
    isfile(csv_path) || continue
    rows   = [split(r, ",") for r in readlines(csv_path)[2:end] if !isempty(r)]
    κ_stab = [parse(Float64, r[1]) for r in rows]
    f_stab = [parse(Float64, r[2]) for r in rows]
    kv     = sort(unique(κ_all));  fv = sort(unique(f0_all))
    mask   = (f_stab .>= fv[1]) .& (f_stab .<= fv[end]) .&
             (κ_stab .>= kv[1]) .& (κ_stab .<= kv[end])
    any(mask) && lines!(ax2, f_stab[mask], κ_stab[mask];
                        color = :black, linewidth = 2.0, linestyle = :dash,
                        label = "Linear stability")
    break
end

save(joinpath(results_dir, "pca_paper_phases.png"), fig2, px_per_unit = 2)
println("Saved: pca_paper_phases.png")
println("\nDone.")
