# transition_plot.jl
# Plot mean asphericity b and mean COM speed v as continuous functions of f₀/κ.
#
# Each dot is one run.  Curled runs (compact ring, strong mode-2) are shown in a
# distinct colour so the reader can judge whether they stand apart from the trend.
# Binned means (± 1 SD) are overlaid to guide the eye.
#
# Two-panel layout:
#   Left  — zoomed linear axis, f₀/κ ∈ [0, 30]
#   Right — full range on a log axis
# Each panel shows b (solid) and v (dashed, right y-axis via twinning) so both
# quantities appear together without needing four separate panels.
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/transition_plot.jl
#
# Output:  results/transition_curves.png

using CairoMakie, Statistics, Printf, JSON3

const PROJECT_ROOT_TP = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_TP, "src", "EngineRing.jl"))
using .EngineRing

include(joinpath(PROJECT_ROOT_TP, "src", "IOUtils.jl"))
using .IOUtils

# ── Parameters ────────────────────────────────────────────────────────────────
const N_STEADY_TP  = 150    # frames from the end for steady-state average
const CURLED_RG    = 2.0    # Rg threshold for curled detection
const CURLED_A2    = 0.25   # A2 threshold for curled detection
const N_BINS_ZOOM  = 20     # bins for zoomed panel
const N_BINS_FULL  = 30     # bins for full-range panel (log-spaced)

# ── Load all runs ─────────────────────────────────────────────────────────────
results_dir = joinpath(PROJECT_ROOT_TP, "results")
run_dirs    = filter(d -> (m = match(r"^run_(\d+)$", d); m !== nothing && 500 <= parse(Int, m[1]) <= 1100),
                     readdir(results_dir, join = false))
sort!(run_dirs)

ratios    = Float64[]   # f₀/κ for each run
b_means   = Float64[]   # steady-state mean b
v_means   = Float64[]   # mean COM speed (NaN if unavailable)
curled_fl = Bool[]      # is this run in the curled state?

println("Loading $(length(run_dirs)) runs…\n")

for rd in run_dirs
    rp = joinpath(results_dir, rd)
    isfile(joinpath(rp, "observables.jld2")) || continue
    isfile(joinpath(rp, "params.toml"))      || continue

    p, _   = IOUtils.load_params(rp)
    _, obs = IOUtils.load_observables(rp)

    haskey(obs, "b")  || continue
    haskey(obs, "Rg") || continue
    haskey(obs, "A2") || continue

    # Skip runs with κ = 0 (ratio undefined) or tiny κ
    p.κ < 0.1 && continue

    n      = length(obs["b"])
    ss_idx = max(1, n - N_STEADY_TP + 1):n

    b_ss  = mean(obs["b"][ss_idx])
    Rg_ss = mean(obs["Rg"][ss_idx])
    A2_ss = mean(obs["A2"][ss_idx])

    curled = (Rg_ss < CURLED_RG) && (A2_ss > CURLED_A2)

    v_mean = NaN
    msj = joinpath(rp, "motion_summary.json")
    if isfile(msj)
        d = JSON3.read(read(msj, String))
        v_mean = Float64(get(d, "com_speed_mean", NaN))
    end

    push!(ratios,    p.f0 / p.κ)
    push!(b_means,   b_ss)
    push!(v_means,   v_mean)
    push!(curled_fl, curled)

    @printf("  f0=%5.1f  κ=%4.1f  f0/κ=%6.2f  b=%.3f  v=%s  %s\n",
            p.f0, p.κ, p.f0 / p.κ, b_ss,
            isnan(v_mean) ? "N/A" : @sprintf("%.2f", v_mean),
            curled ? "[CURLED]" : "")
end

println("\nLoaded $(length(ratios)) runs  ($(sum(curled_fl)) curled)")

has_v = .!isnan.(v_means)

# ── Binning helper ────────────────────────────────────────────────────────────
# Returns (bin_centers, b_mean_binned, b_sd_binned, v_mean_binned, v_sd_binned, counts)
function bin_continuous(ratios, b_vals, v_vals; edges)
    nb      = length(edges) - 1
    centers = [(edges[i] + edges[i+1]) / 2 for i in 1:nb]
    b_sum   = zeros(nb); b_sum2 = zeros(nb)
    v_sum   = zeros(nb); v_sum2 = zeros(nb)
    b_cnt   = zeros(Int, nb); v_cnt = zeros(Int, nb)
    ev      = collect(edges)

    for j in eachindex(ratios)
        bi = clamp(searchsortedlast(ev, ratios[j]), 1, nb)
        b_sum[bi]  += b_vals[j]; b_sum2[bi] += b_vals[j]^2; b_cnt[bi] += 1
        if !isnan(v_vals[j])
            v_sum[bi] += v_vals[j]; v_sum2[bi] += v_vals[j]^2; v_cnt[bi] += 1
        end
    end

    b_m = [b_cnt[i] > 0 ? b_sum[i] / b_cnt[i] : NaN for i in 1:nb]
    b_s = [b_cnt[i] > 1 ? sqrt(max(0, b_sum2[i]/b_cnt[i] - b_m[i]^2)) : 0.0 for i in 1:nb]
    v_m = [v_cnt[i] > 0 ? v_sum[i] / v_cnt[i] : NaN for i in 1:nb]
    v_s = [v_cnt[i] > 1 ? sqrt(max(0, v_sum2[i]/v_cnt[i] - v_m[i]^2)) : 0.0 for i in 1:nb]

    m_b = b_cnt .> 0
    m_v = v_cnt .> 0
    return centers, b_m, b_s, v_m, v_s, m_b, m_v
end

# ── Build figure ──────────────────────────────────────────────────────────────
# Two columns; each column has two y-axes (b left, v right).

fig = Figure(size = (1400, 500))

# colours
c_nc_b   = :royalblue
c_cu_b   = :tomato
c_nc_v   = :darkorange
c_cu_v   = :firebrick

function add_panel!(fig, col, xlabel_str, title_str, edges, ratio_mask)
    rm   = ratio_mask
    r_s  = ratios[rm];    b_s = b_means[rm];  v_s = v_means[rm]
    cu_s = curled_fl[rm]

    # Primary axis: b
    ax_b = Axis(fig[1, col];
        xlabel   = xlabel_str,
        ylabel   = "mean asphericity  b",
        title    = title_str,
        yticklabelcolor = c_nc_b,
        ylabelcolor     = c_nc_b,
    )

    # Scatter b — non-curled
    nc = .!cu_s
    scatter!(ax_b, r_s[nc], b_s[nc];
             color = (c_nc_b, 0.4), markersize = 8)
    # Scatter b — curled
    if any(cu_s)
        scatter!(ax_b, r_s[cu_s], b_s[cu_s];
                 color = (c_cu_b, 0.6), marker = :star5, markersize = 14)
    end

    # Binned b curve
    centers, b_m, b_sd, v_m, v_sd, m_b, m_v = bin_continuous(r_s, b_s, v_s; edges)
    cx = centers[m_b]; by = b_m[m_b]; bsd = b_sd[m_b]
    if length(cx) > 1
        band!(ax_b, cx, by .- bsd, by .+ bsd; color = (c_nc_b, 0.15))
        lines!(ax_b, cx, by; color = c_nc_b, linewidth = 2.5)
    end

    # Secondary axis: v (right side)
    ax_v = Axis(fig[1, col];
        yaxisposition   = :right,
        ylabel          = "mean COM speed  v",
        yticklabelcolor = c_nc_v,
        ylabelcolor     = c_nc_v,
        xticksvisible   = false,
        xticklabelsvisible = false,
        xlimits         = ax_b.xlimits,
    )
    linkxaxes!(ax_b, ax_v)
    hidespines!(ax_v, :l, :t, :b)

    hv = .!isnan.(v_s)
    if any(hv .& nc)
        scatter!(ax_v, r_s[hv .& nc], v_s[hv .& nc];
                 color = (c_nc_v, 0.4), marker = :diamond, markersize = 8)
    end
    if any(hv .& cu_s)
        scatter!(ax_v, r_s[hv .& cu_s], v_s[hv .& cu_s];
                 color = (c_cu_v, 0.6), marker = :star5, markersize = 14)
    end

    cx_v = centers[m_v]; vy = v_m[m_v]; vsd = v_sd[m_v]
    if length(cx_v) > 1
        band!(ax_v, cx_v, vy .- vsd, vy .+ vsd; color = (c_nc_v, 0.15))
        lines!(ax_v, cx_v, vy; color = c_nc_v, linewidth = 2.5, linestyle = :dash)
    end

    return ax_b, ax_v
end

# --- Panel A: zoomed linear, f₀/κ ∈ [0, 30] ---
zoom_lo, zoom_hi = 0.0, 30.0
mask_zoom = (ratios .>= zoom_lo) .& (ratios .<= zoom_hi)
edges_zoom = range(zoom_lo, zoom_hi, length = N_BINS_ZOOM + 1)

ax_b1, ax_v1 = add_panel!(fig, 1,
    "f₀ / κ",
    "A)  Transition region  (f₀/κ = 0 – 30)",
    edges_zoom, mask_zoom)
xlims!(ax_b1, zoom_lo, zoom_hi)

# --- Panel B: full range, log scale ---
rmin = max(minimum(ratios), 0.01)
rmax = maximum(ratios)
log_edges = 10 .^ range(log10(rmin), log10(rmax), length = N_BINS_FULL + 1)

ax_b2, ax_v2 = add_panel!(fig, 3,
    "f₀ / κ  (log scale)",
    "B)  Full range",
    log_edges, fill(true, length(ratios)))
ax_b2.xscale = log10
ax_v2.xscale = log10
xlims!(ax_b2, rmin * 0.8, rmax * 1.2)

# ── Legend ────────────────────────────────────────────────────────────────────
elems = [
    MarkerElement(color = c_nc_b,  marker = :circle,  markersize = 10),
    MarkerElement(color = c_cu_b,  marker = :star5,   markersize = 12),
    LineElement(   color = c_nc_b, linewidth = 2.5),
    LineElement(   color = c_nc_v, linewidth = 2.5, linestyle = :dash),
]
labels = ["Non-curled (scatter)", "Curled (scatter)", "Binned mean b  ±SD", "Binned mean v  ±SD"]
Legend(fig[1, 2], elems, labels; framevisible = false, labelsize = 12)

out_path = joinpath(results_dir, "transition_curves.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
