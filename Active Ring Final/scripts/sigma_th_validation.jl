# sigma_th_validation.jl
# Does the analytic thermal jitter σ_th = sqrt(kBT·l/κ) actually describe the
# measured turning-angle fluctuations in the CIRCULAR region?  If yes, the
# κ-dependent ±3·σ_th band used by reduce_kconn.jl (P_out / P_circ / P_in and the
# crossing rule) is validated with no tuned constant.  If the measured std sits
# systematically ABOVE σ_th (extra correlations through ring closure), the offset
# factor tells us how much to widen the band.
#
# Data: the circle-IC CSV (reduce_kconn.jl output).  Cells are restricted to the
# known-circular corner — low f₀ (lowest-f₀ decade) AND P_circ ≈ 1 — so the
# junction is genuinely a thermal circular joint, not a buckled/dynamic one.
#
# σ_th = sqrt(kBT·l/κ); with l = 1 this is the sqrt(kBT/κ) reference curve.  We
# draw it straight from the CSV's sigma_th column (which is exactly sqrt(kBT·l/κ)
# for each run's params), so nothing here is hardcoded.
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/sigma_th_validation.jl

using CairoMakie, Statistics, Printf

const ROOT      = joinpath(@__DIR__, "..")
const OUT_DIR   = joinpath(ROOT, "results", "kconn_phase")
const CSV_PATH  = joinpath(OUT_DIR, "circle.csv")
const PCIRC_MIN = 0.95        # "P_circ ≈ 1": genuinely circular cells only

# ── Minimal CSV → column dict (numeric columns as Float64; NaN if unparseable) ──
function load_csv_cols(path)
    lines  = filter(!isempty, readlines(path))
    header = string.(split(lines[1], ","))
    data   = [split(l, ",") for l in lines[2:end]]
    cols   = Dict{String,Vector{Float64}}()
    for (j, h) in enumerate(header)
        cols[h] = Float64[something(tryparse(Float64, d[j]), NaN) for d in data]
    end
    return cols
end

c      = load_csv_cols(CSV_PATH)
κ      = c["kappa"]
f0     = c["f0"]
stdK   = c["std_Kconn"]
Pcirc  = c["P_circ"]
σth    = c["sigma_th"]

# ── Restrict to the circular low-f0 region ────────────────────────────────────
f0_decade = minimum(f0) * 10.0                       # lowest-f0 decade
sel       = (f0 .<= f0_decade) .& (Pcirc .>= PCIRC_MIN)
n_sel     = count(sel)
n_sel == 0 && error("No circular low-f0 cells selected (f0 ≤ $f0_decade, P_circ ≥ $PCIRC_MIN).")

ratio_all = stdK ./ σth                              # measured / analytic

# Per-κ aggregation (mean measured std and mean ratio over the qualifying f0 cells)
κs_sel   = sort(unique(κ[sel]))
std_byκ  = [mean(stdK[sel .& (κ .== k)])      for k in κs_sel]
ratio_byκ= [mean(ratio_all[sel .& (κ .== k)]) for k in κs_sel]

# Analytic σ_th reference curve over the FULL κ range (σ_th is κ-only)
κs_all = sort(unique(κ))
σ_curve = [σth[findfirst(==(k), κ)] for k in κs_all]

med_ratio = median(ratio_all[sel])
# The systematic offset lives in the mid/high-κ circular band; report both the
# global median and the max per-κ mean so the widen-the-band decision is explicit.
max_ratio = maximum(ratio_byκ)
κ_at_max  = κs_sel[argmax(ratio_byκ)]

println("σ_th validation (circle IC, f0 ≤ $(round(f0_decade;digits=2)), P_circ ≥ $PCIRC_MIN):")
@printf("  %d cells selected across %d κ values\n", n_sel, length(κs_sel))
@printf("  median measured/σ_th ratio: %.3f\n", med_ratio)
@printf("  max per-κ mean ratio: %.3f (at κ=%.2f)\n", max_ratio, κ_at_max)
println("  per-κ:  κ  |  mean std  |  σ_th  |  ratio")
for i in eachindex(κs_sel)
    σ_i = σth[findfirst(==(κs_sel[i]), κ)]
    @printf("    %7.3f   %.4f   %.4f   %.2f\n", κs_sel[i], std_byκ[i], σ_i, ratio_byκ[i])
end

# ── Figure: (left) std vs κ with the σ_th curve, (right) ratio vs κ ───────────
fig = Figure(size = (1050, 460))
Label(fig[0, 1:2],
      "σ_th validation — circle IC, circular low-f₀ cells (f₀ ≤ $(round(f0_decade;digits=1)), P_circ ≥ $PCIRC_MIN)";
      fontsize = 14, tellwidth = false)

ax1 = Axis(fig[1, 1];
           xlabel = "Bending stiffness  κ",
           ylabel = "std(Kconn)  [rad]",
           title  = "measured std vs analytic σ_th = √(k_BT·l/κ)",
           xscale = log10, yscale = log10,
           xticks = LogTicks(WilkinsonTicks(6)),
           yticks = LogTicks(WilkinsonTicks(6)))
lines!(ax1, κs_all, σ_curve; color = :black, linewidth = 2,
       label = "σ_th = √(k_BT·l/κ)")
scatter!(ax1, κ[sel], stdK[sel]; color = (:steelblue, 0.45), markersize = 6,
         label = "measured (per cell)")
scatter!(ax1, κs_sel, std_byκ; color = :crimson, markersize = 9,
         label = "measured (per-κ mean)")
axislegend(ax1; position = :lb, framevisible = true, labelsize = 10)

ax2 = Axis(fig[1, 2];
           xlabel = "Bending stiffness  κ",
           ylabel = "ratio  std(Kconn) / σ_th",
           title  = "offset factor (>1 ⇒ widen band)",
           xscale = log10,
           xticks = LogTicks(WilkinsonTicks(6)))
scatter!(ax2, κ[sel], ratio_all[sel]; color = (:steelblue, 0.35), markersize = 5)
lines!(ax2, κs_sel, ratio_byκ; color = :crimson, linewidth = 2,
       label = "per-κ mean")
hlines!(ax2, [1.0]; color = :black, linestyle = :dash, label = "σ_th exact (ratio = 1)")
hlines!(ax2, [med_ratio]; color = :seagreen, linestyle = :dot, linewidth = 2,
        label = @sprintf("median = %.2f", med_ratio))
axislegend(ax2; position = :rt, framevisible = true, labelsize = 10)

out_path = joinpath(OUT_DIR, "sigma_th_validation.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
