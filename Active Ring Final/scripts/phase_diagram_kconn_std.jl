# phase_diagram_kconn_std.jl
# A simple κ × f₀ phase diagram coloured by ONE quantity: the standard deviation
# of Kconn (the passive-junction turning angle, in degrees).
#
# std(Kconn) is taken over the WHOLE post-transient window — from SKIP_SECONDS to
# the end of the run.  Nothing is time-gated: if the ring locks into a fixed shape
# partway through, that frozen stretch still counts towards the std, exactly as it
# would in a naive measurement.
#
# Two panels are saved side by side:
#   LEFT  — a DECISIVE categorical map.  The structural state (Circular / Inward-
#           curled / Outward-curled) comes from ⟨Kconn⟩ vs 360/N, which has real
#           gaps, so those boundaries are crisp.  A "Dynamic (keeps moving)" flag
#           is added on top — that is the one soft axis (std is a continuum).
#   RIGHT — the raw continuous std(Kconn) field, for reference.
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_kconn_std.jl
#   julia --project scripts/phase_diagram_kconn_std.jl Results_N17_circle
#   julia --project scripts/phase_diagram_kconn_std.jl Results_N17_circle 1 500

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT  = joinpath(@__DIR__, "..")
const SKIP_SECONDS  = 0.5    # drop the brief IC-relaxation transient (sim-time)

# ── Decisive categorical map: structure (⟨Kconn⟩) + a dynamic flag ─────────────
# Structure separates cleanly (real gaps in ⟨Kconn⟩), so those cuts are crisp.
# "Dynamic" (keeps moving) is the one soft axis — std is a continuum.
const CIRC_BAND_DEG = 8.0    # |⟨Kconn⟩−360/N| < this ⇒ Circular (real gap here)
const DYN_STD_FLOOR = 10.0   # std(Kconn) above this ⇒ genuinely moving

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan runs ─────────────────────────────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id_start ≤ parse(Int, m[1]) ≤ id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder…")

# 1 = Dynamic   2 = Circular   3 = Inward-curled   4 = Outward-curled
function classify(kstd, dev)
    kstd > DYN_STD_FLOOR      && return 1   # keeps moving
    abs(dev) < CIRC_BAND_DEG  && return 2   # circular
    dev < 0                   && return 3   # inward
    return 4                                # outward
end

κ_raw = Float64[]; f0_raw = Float64[]; std_raw = Float64[]; code_raw = Int[]
for rd in run_dirs
    obs = joinpath(results_dir, rd, "observables.jld2")
    par = joinpath(results_dir, rd, "params.toml")
    (isfile(obs) && isfile(par)) || continue

    ep = TOML.parsefile(par)["params"]
    N  = Int(ep["N"])
    d = jldopen(obs, "r") do f
        haskey(f, "Kconn") || return nothing
        m = length(f["Kconn"])
        t = haskey(f, "times") ? f["times"] : collect(1:m)   # fallback: frame index
        (t = t, K = rad2deg.(f["Kconn"]))
    end
    d === nothing && continue
    length(d.K) < 10 && continue

    # Whole post-transient window — no time gating.
    i0   = something(findfirst(≥(SKIP_SECONDS), d.t), 1)
    seg  = @view d.K[i0:end]
    kstd = std(seg)
    dev  = mean(seg) - 360.0 / N               # structural axis

    push!(κ_raw,  Float64(ep["kappa"]))
    push!(f0_raw, Float64(ep["f0"]))
    push!(std_raw, kstd)
    push!(code_raw, classify(kstd, dev))
end

n = length(std_raw)
println("Loaded $n runs with Kconn data.")
n == 0 && error("No Kconn data found — was this sweep run with run_sweep.jl?")
@printf("std(Kconn) over the whole post-transient window:  min=%.2f  median=%.2f  max=%.2f  (deg)\n",
        minimum(std_raw), median(std_raw), maximum(std_raw))

phase_colors = [:crimson, :gainsboro, :darkorange, :steelblue]
phase_labels = ["Dynamic (keeps moving)", "Circular", "Inward-curled", "Outward-curled"]
for c in 1:4
    @printf("  %-24s %d cells\n", phase_labels[c], count(==(c), code_raw))
end

# ── κ × f₀ grid ───────────────────────────────────────────────────────────────
kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f0_raw))
nκ = length(kappa_vals); nf = length(f0_vals)
std_lookup  = Dict(zip(zip(κ_raw, f0_raw), std_raw))
code_lookup = Dict(zip(zip(κ_raw, f0_raw), code_raw))
std_mat  = fill(NaN, nκ, nf)
code_mat = fill(NaN, nκ, nf)
for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
    haskey(std_lookup,  (κ, f₀)) && (std_mat[ic, ir]  = std_lookup[(κ, f₀)])
    haskey(code_lookup, (κ, f₀)) && (code_mat[ic, ir] = code_lookup[(κ, f₀)])
end

# ── Figure: decisive categorical map (left) + continuous std (right) ──────────
fig = Figure(size = (1500, 760))
Label(fig[0, 1:4], "Kconn phase diagram — $subfolder  (N runs = $n)";
      fontsize = 15, tellwidth = false)

ax1 = Axis(fig[1, 1];
    xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
    title  = "Decisive states  (structure = ⟨Kconn⟩, + dynamic flag)",
    xscale = log10, yscale = log10,
    xticks = LogTicks(WilkinsonTicks(6)), yticks = LogTicks(WilkinsonTicks(6)))
heatmap!(ax1, f0_vals, kappa_vals, code_mat';
         colormap = cgrad(phase_colors; categorical = true),
         colorrange = (0.5, 4.5), nan_color = :white)
Legend(fig[1, 2],
       [PolyElement(color = c) for c in phase_colors], phase_labels, "Phase";
       framevisible = true)

hi_clip = quantile(std_raw, 0.98)               # robust upper bound (avoid outliers)
ax2 = Axis(fig[1, 3];
    xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
    title  = "std(Kconn) over the active window  (continuous)",
    xscale = log10, yscale = log10,
    xticks = LogTicks(WilkinsonTicks(6)), yticks = LogTicks(WilkinsonTicks(6)))
hm = heatmap!(ax2, f0_vals, kappa_vals, std_mat';
              colormap = :viridis, colorrange = (0.0, hi_clip),
              highclip = :yellow, nan_color = :lightgrey)
Colorbar(fig[1, 4], hm; label = "std(Kconn)  [deg]  (whole run)", width = 16)

out = joinpath(results_dir, "phase_diagram_kconn_std.png")
save(out, fig, px_per_unit = 2)
println("Saved: $out")
