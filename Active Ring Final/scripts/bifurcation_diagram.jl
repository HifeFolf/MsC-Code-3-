# bifurcation_diagram.jl
# Construct a 1-D bifurcation diagram from a completed κ × f₀ sweep.
#
# The control parameter is Φ = f₀ / κ (activity-to-bending ratio).
# The order parameter is ⟨Kconn⟩_ss — the steady-state mean signed turning
# angle at the passive junction bead.  Positive = outward buckling,
# negative = inward buckling, near-zero = circle / droplet.
#
# A supplementary panel shows the mean Fourier mode amplitudes A2–A6 along
# the same path, so the dominant shape mode can be read off directly.
#
# Path selection:
#   --all          use every run in the sweep (projected onto Φ axis) [default]
#   --kappa VAL    fixed-κ slice  (pick the κ value nearest to VAL)
#   --f0    VAL    fixed-f₀ slice (pick the f₀ value nearest to VAL)
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/bifurcation_diagram.jl
#   julia --project scripts/bifurcation_diagram.jl Results_log
#   julia --project scripts/bifurcation_diagram.jl Results_log --kappa 2.0
#   julia --project scripts/bifurcation_diagram.jl Results_log --f0 10.0
#
# Arguments:
#   arg 1              — results subfolder (default: Results_log)
#   --kappa / --f0 VAL — path selector (default: all runs)
#   --transient FRAC   — fraction of frames to discard as transient (default: 0.3)
#
# Output:  results/<subfolder>/bifurcation_diagram[_kappa<V>|_f0<V>].png

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_BD = joinpath(@__DIR__, "..")

# ── Parse arguments ───────────────────────────────────────────────────────────

function _parse_args(args)
    subfolder      = "Results_N17_kinked2"
    path_mode      = :all
    path_val       = NaN
    transient_frac = 0.3
    i = 1
    while i <= length(args)
        a = args[i]
        if !startswith(a, "-")
            subfolder = a
        elseif a == "--kappa" && i < length(args)
            path_mode = :kappa; path_val = parse(Float64, args[i+1]); i += 1
        elseif a == "--f0" && i < length(args)
            path_mode = :f0;    path_val = parse(Float64, args[i+1]); i += 1
        elseif a == "--all"
            path_mode = :all
        elseif a == "--transient" && i < length(args)
            transient_frac = parse(Float64, args[i+1]); i += 1
        end
        i += 1
    end
    return subfolder, path_mode, path_val, transient_frac
end

subfolder, path_mode, path_val, transient_frac = _parse_args(ARGS)

results_dir = joinpath(PROJECT_ROOT_BD, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Collect available κ and f₀ values from params files ──────────────────────

run_dirs = sort(filter(d -> occursin(r"^run_\d+$", d),
                       readdir(results_dir, join = false)))

println("Scanning $(length(run_dirs)) runs in $subfolder …")

# First pass: collect all distinct (κ, f₀) to pick the nearest slice value
all_kappa = Float64[]
all_f0    = Float64[]
for rd in run_dirs
    pf = joinpath(results_dir, rd, "params.toml")
    isfile(pf) || continue
    ep = TOML.parsefile(pf)["params"]
    push!(all_kappa, Float64(ep["kappa"]))
    push!(all_f0,    Float64(ep["f0"]))
end

# Snap path_val to the nearest available grid value
if path_mode == :kappa
    unique_kappa = sort(unique(all_kappa))
    path_val = unique_kappa[argmin(abs.(unique_kappa .- path_val))]
    @printf("Fixed-κ slice: κ = %.4g  (nearest grid value)\n", path_val)
elseif path_mode == :f0
    unique_f0 = sort(unique(all_f0))
    path_val = unique_f0[argmin(abs.(unique_f0 .- path_val))]
    @printf("Fixed-f₀ slice: f₀ = %.4g  (nearest grid value)\n", path_val)
else
    println("Using all runs projected onto Φ = f₀/κ axis.")
end

# ── Second pass: load steady-state averages ───────────────────────────────────

Φ_vec        = Float64[]
Kconn_mean   = Float64[]
Kconn_abs    = Float64[]   # ⟨|Kconn|⟩ — unsigned order parameter
Kconn_std    = Float64[]
A2_mean      = Float64[]
A3_mean      = Float64[]
A4_mean      = Float64[]
A5_mean      = Float64[]
A6_mean      = Float64[]
vcom_mean    = Float64[]
b_mean       = Float64[]
κ_vec        = Float64[]
f0_vec       = Float64[]

n_skipped = 0

for rd in run_dirs
    run_path = joinpath(results_dir, rd)
    obs_file = joinpath(run_path, "observables.jld2")
    par_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(par_file)) || (n_skipped += 1; continue)

    ep = TOML.parsefile(par_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    # Filter to requested path
    if path_mode == :kappa && κ  != path_val; continue; end
    if path_mode == :f0    && f₀ != path_val; continue; end

    # Load only the keys we need
    obs = jldopen(obs_file, "r") do f
        haskey(f, "Kconn") || return nothing
        (Kconn = f["Kconn"],
         A2    = f["A2"],
         A3    = f["A3"],
         A4    = f["A4"],
         A5    = haskey(f, "A5") ? f["A5"] : nothing,
         A6    = haskey(f, "A6") ? f["A6"] : nothing,
         v_com = haskey(f, "v_com") ? f["v_com"] : nothing,
         b     = f["b"])
    end
    obs === nothing && (n_skipped += 1; continue)

    n   = length(obs.Kconn)
    i0  = max(1, round(Int, transient_frac * n) + 1)   # first steady-state frame
    ss  = i0:n

    push!(Φ_vec,      f₀ / κ)
    push!(κ_vec,      κ)
    push!(f0_vec,     f₀)
    push!(Kconn_mean, mean(obs.Kconn[ss]))
    push!(Kconn_abs,  mean(abs.(obs.Kconn[ss])))
    push!(Kconn_std,  std(obs.Kconn[ss]))
    push!(A2_mean,    mean(obs.A2[ss]))
    push!(A3_mean,    mean(obs.A3[ss]))
    push!(A4_mean,    mean(obs.A4[ss]))
    push!(A5_mean,    obs.A5 !== nothing ? mean(obs.A5[ss]) : NaN)
    push!(A6_mean,    obs.A6 !== nothing ? mean(obs.A6[ss]) : NaN)
    push!(vcom_mean,  obs.v_com !== nothing ? mean(obs.v_com[ss]) : NaN)
    push!(b_mean,     mean(obs.b[ss]))
end

n_loaded = length(Φ_vec)
@printf("Loaded %d runs  (%d skipped / missing Kconn)\n", n_loaded, n_skipped)
n_loaded == 0 && error("No runs with Kconn found. Re-run the sweep with the updated Observables.jl.")

# Sort by Φ (or by f₀ for fixed-κ slices, since Φ ∝ f₀ there anyway)
ord = sortperm(Φ_vec)
Φ_vec      = Φ_vec[ord]
Kconn_mean = Kconn_mean[ord]
Kconn_abs  = Kconn_abs[ord]
Kconn_std  = Kconn_std[ord]
A2_mean    = A2_mean[ord]
A3_mean    = A3_mean[ord]
A4_mean    = A4_mean[ord]
A5_mean    = A5_mean[ord]
A6_mean    = A6_mean[ord]
vcom_mean  = vcom_mean[ord]
b_mean     = b_mean[ord]
κ_vec      = κ_vec[ord]
f0_vec     = f0_vec[ord]

# ── Figure ────────────────────────────────────────────────────────────────────

path_label = if path_mode == :kappa
    @sprintf("κ = %.3g", path_val)
elseif path_mode == :f0
    @sprintf("f₀ = %.3g", path_val)
else
    "all runs"
end

# Build clean tick labels for the Φ axis (shows actual values, not log exponents)
function _phi_ticks(Φ_vec)
    lo = floor(log10(minimum(Φ_vec)))
    hi = ceil(log10(maximum(Φ_vec)))
    vals = Float64[]
    for e in lo:hi
        for m in (1.0, 2.0, 5.0)
            v = m * 10.0^e
            minimum(Φ_vec) <= v <= maximum(Φ_vec) && push!(vals, v)
        end
    end
    unique!(sort!(vals))
    labels = String[]; for v in vals; push!(labels, @sprintf("%.4g", v)); end
    return vals, labels
end

phi_ticks = _phi_ticks(Φ_vec)

fig = Figure(size = (900, 780))

Label(fig[0, 1:2],
      "Bifurcation diagram — $subfolder   ($path_label,  transient = $(round(Int,100*transient_frac))%)";
      fontsize = 13, tellwidth = false)

# ── Panel 1: Kconn bifurcation ────────────────────────────────────────────────

ax1 = Axis(fig[1, 1];
    xlabel  = "Activity-to-bending ratio  Φ = f₀ / κ",
    ylabel  = "Kconn  (signed curvature at junction)",
    title   = "Order parameter",
    xscale  = log10,
    xticks  = phi_ticks,
)

# ⟨|Kconn|⟩ — unsigned: rises from zero at bifurcation regardless of which branch
lines!(ax1, Φ_vec, Kconn_abs;  color = :firebrick,  linewidth = 2.5,
       label = "⟨|Kconn|⟩  (unsigned, continuous/discontinuous diagnostic)")
scatter!(ax1, Φ_vec, Kconn_abs; color = :firebrick, markersize = 5)

# ⟨Kconn⟩ ± std — signed: averages to ~0 when bistable, but band width shows fluctuations
band!(ax1, Φ_vec, Kconn_mean .- Kconn_std, Kconn_mean .+ Kconn_std;
      color = (:steelblue, 0.18))
lines!(ax1, Φ_vec, Kconn_mean; color = :steelblue, linewidth = 1.5,
       linestyle = :dash, label = "⟨Kconn⟩ ± std  (signed, bistable → ≈ 0)")

# Zero reference
hlines!(ax1, [0.0]; color = (:black, 0.35), linestyle = :dot, linewidth = 1)

axislegend(ax1; position = :lt, framevisible = false, labelsize = 10)

# Annotate sign convention
text!(ax1, minimum(Φ_vec), maximum(Kconn_abs) * 0.9;
      text = "continuous:  ⟨|Kconn|⟩ rises smoothly from 0\ndiscontinuous:  ⟨|Kconn|⟩ jumps",
      fontsize = 9, color = (:black, 0.55), align = (:left, :top))

# ── Panel 2: mode amplitudes ──────────────────────────────────────────────────

ax2 = Axis(fig[1, 2];
    xlabel  = "Φ = f₀ / κ",
    ylabel  = "Mean mode amplitude",
    title   = "Fourier modes (angle-based)",
    xscale  = log10,
    xticks  = phi_ticks,
)

lines!(ax2, Φ_vec, A2_mean; color = :dodgerblue,  linewidth = 2, label = "A₂ (ellipse)")
lines!(ax2, Φ_vec, A3_mean; color = :orange,       linewidth = 2, label = "A₃ (triangle)")
lines!(ax2, Φ_vec, A4_mean; color = :green,         linewidth = 2, label = "A₄ (square)")
if !all(isnan, A5_mean)
    lines!(ax2, Φ_vec, A5_mean; color = :purple,   linewidth = 2, label = "A₅")
end
if !all(isnan, A6_mean)
    lines!(ax2, Φ_vec, A6_mean; color = :red,       linewidth = 2, label = "A₆")
end
axislegend(ax2; position = :lt, framevisible = false, labelsize = 11)

# ── Panel 3: Kconn std (fluctuations — peaks near transitions) ────────────────

ax3 = Axis(fig[2, 1];
    xlabel  = "Φ = f₀ / κ",
    ylabel  = "std(Kconn)",
    title   = "Fluctuations of Kconn  (peaks signal transitions)",
    xscale  = log10,
    xticks  = phi_ticks,
)
lines!(ax3,   Φ_vec, Kconn_std; color = :darkorange, linewidth = 2)
scatter!(ax3, Φ_vec, Kconn_std; color = :darkorange, markersize = 5)

# ── Panel 4: COM speed and asphericity ───────────────────────────────────────

ax4 = Axis(fig[2, 2];
    xlabel  = "Φ = f₀ / κ",
    ylabel  = "Mean value",
    title   = "Asphericity b  and  COM speed v_com",
    xscale  = log10,
    xticks  = phi_ticks,
)
lines!(ax4, Φ_vec, b_mean; color = :teal,      linewidth = 2, label = "b (asphericity)")
if !all(isnan, vcom_mean)
    ax4r = Axis(fig[2, 2];
        ylabel       = "⟨v_com⟩",
        yaxisposition = :right,
        xscale       = log10,
        xticks       = LogTicks(WilkinsonTicks(6)),
        yticklabelcolor = :firebrick,
        rightspinecolor  = :firebrick,
    )
    hidexdecorations!(ax4r)
    lines!(ax4r, Φ_vec, vcom_mean; color = :firebrick, linewidth = 2,
           linestyle = :dash, label = "v_com")
end
axislegend(ax4; position = :lt, framevisible = false, labelsize = 11)

# ── Save ──────────────────────────────────────────────────────────────────────

suffix = if path_mode == :kappa
    @sprintf("_kappa%.4g", path_val)
elseif path_mode == :f0
    @sprintf("_f0_%.4g", path_val)
else
    "_all"
end

out_path = joinpath(results_dir, "bifurcation_diagram$(suffix).png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
