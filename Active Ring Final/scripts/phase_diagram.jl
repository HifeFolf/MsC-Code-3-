# phase_diagram.jl
# Phase diagram for the flexure-number / kBT sweep (runs 1500–9000).
#
# Axes:
#   x — flexure number  Fl = f₀ / κ   (dimensionless activity)
#   y — thermal energy  kBT
#
# Two panels:
#   Left  — colored by steady-state mean asphericity b  (viridis)
#   Right — colored by mean COM speed v                  (plasma, requires motion_summary.json)
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/phase_diagram.jl
#
# Output:  results/phase_diagram.png

using CairoMakie, Statistics, Printf, JSON3

const PROJECT_ROOT_PD = joinpath(@__DIR__, "..")

if !isdefined(Main, :EngineRing)
    include(joinpath(PROJECT_ROOT_PD, "src", "EngineRing.jl"))
    using .EngineRing
end
if !isdefined(Main, :IOUtils)
    include(joinpath(PROJECT_ROOT_PD, "src", "IOUtils.jl"))
    using .IOUtils
end

# ── Parameters ────────────────────────────────────────────────────────────────
const N_STEADY_PD = 150   # frames from the end used for steady-state averages

# ── Scan runs 1500–9000 ───────────────────────────────────────────────────────
results_dir = joinpath(PROJECT_ROOT_PD, "results")
run_dirs    = filter(d -> (m = match(r"^run_(\d+)$", d);
                           m !== nothing && 1500 <= parse(Int, m[1]) <= 9000),
                     readdir(results_dir, join = false))
sort!(run_dirs)

flex_vals = Float64[]   # f₀/κ  (flexure number)
kbt_vals  = Float64[]   # kBT
b_vals    = Float64[]   # steady-state mean asphericity
v_vals    = Float64[]   # mean COM speed (NaN if unavailable)

println("Scanning $(length(run_dirs)) runs…\n")

for rd in run_dirs
    run_path = joinpath(results_dir, rd)

    isfile(joinpath(run_path, "observables.jld2")) || continue
    isfile(joinpath(run_path, "params.toml"))      || continue

    params, _ = IOUtils.load_params(run_path)
    _, obs    = IOUtils.load_observables(run_path)

    haskey(obs, "b")  || continue
    haskey(obs, "Rg") || continue

    n      = length(obs["b"])
    ss_idx = max(1, n - N_STEADY_PD + 1):n

    b_ss = mean(obs["b"][ss_idx])
    flex = params.f0 / params.κ

    v_mean = NaN
    msj = joinpath(run_path, "motion_summary.json")
    if isfile(msj)
        d = JSON3.read(read(msj, String))
        v_mean = Float64(get(d, "com_speed_mean", NaN))
    end

    push!(flex_vals, flex)
    push!(kbt_vals,  params.kBT)
    push!(b_vals,    b_ss)
    push!(v_vals,    v_mean)

    @printf("  f0=%5.1f  κ=%4.1f  Fl=%5.2f  kBT=%.1f  b=%.3f  v=%s\n",
            params.f0, params.κ, flex, params.kBT, b_ss,
            isnan(v_mean) ? "N/A" : @sprintf("%.2f", v_mean))
end

println("\nTotal runs loaded: $(length(flex_vals))")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (1100, 520))

fl_lo, fl_hi  = isempty(flex_vals) ? (0.0, 1.0) : extrema(flex_vals)
kbt_lo, kbt_hi = isempty(kbt_vals) ? (0.0, 1.0) : extrema(kbt_vals)
pad_fl  = 0.08 * max(fl_hi  - fl_lo,  1.0)
pad_kbt = 0.08 * max(kbt_hi - kbt_lo, 0.1)

function setup_axis_pd(pos, title_str)
    ax = Axis(pos;
        xlabel = "Flexure number  Fl = f₀/κ",
        ylabel = "Thermal energy  kBT",
        title  = title_str,
    )
    xlims!(ax, fl_lo  - pad_fl,  fl_hi  + pad_fl)
    ylims!(ax, kbt_lo - pad_kbt, kbt_hi + pad_kbt)
    ax
end

# Panel A: colored by b
ax1    = setup_axis_pd(fig[1, 1], "Phase diagram  —  color = asphericity  b")
if !isempty(b_vals)
    crange = (0.0, max(0.5, maximum(b_vals)))
    sc1    = scatter!(ax1, flex_vals, kbt_vals;
                      color       = b_vals,
                      colormap    = :viridis,
                      colorrange  = crange,
                      markersize  = 14,
                      strokewidth = 0.3,
                      strokecolor = :black)
    Colorbar(fig[1, 2], sc1; label = "steady-state mean  b", width = 16)
end

# Panel B: colored by v
ax2   = setup_axis_pd(fig[1, 3], "Phase diagram  —  color = COM speed  v")
has_v = .!isnan.(v_vals)
if any(has_v)
    v_sub  = v_vals[has_v]
    crange = (0.0, max(1.0, maximum(v_sub)))
    sc2    = scatter!(ax2, flex_vals[has_v], kbt_vals[has_v];
                      color       = v_sub,
                      colormap    = :plasma,
                      colorrange  = crange,
                      markersize  = 14,
                      strokewidth = 0.3,
                      strokecolor = :black)
    Colorbar(fig[1, 4], sc2; label = "mean COM speed  v", width = 16)
end

out_path = joinpath(results_dir, "phase_diagram.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
