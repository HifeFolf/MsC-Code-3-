# phase_diagram_kinked_comparison.jl
# Compare phase diagrams across all kinked IC variants.
#
# Produces two figures:
#   results/kinked_comparison_grid.png    — 2×3 heatmap panels, shared colorbar
#   results/kinked_comparison_summary.png — contour overlay | average | IC spread
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_kinked_comparison.jl

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_KC = joinpath(@__DIR__, "..")
const LAST_FRAMES     = round(Int, 20.0 * 60)   # last 20 s at 60 fps

DATASETS = [
    ("Results_N17_circle",       "Circle (k=0)"),
    ("Results_N17_kinked",       "Kinked (k=1)"),
    ("Results_N17_doublekinked", "Double (k=2)"),
    ("Results_N17_triplekinked", "Triple (k=3)"),
    ("Results_N17_quadkinked",   "Quad (k=4)"),
    ("Results_N17_quintkinked",  "Quint (k=5)"),
]

IC_COLORS = [:black, :royalblue, :darkorange, :forestgreen, :crimson, :purple]

# ── Load one dataset ──────────────────────────────────────────────────────────
function load_dataset(subfolder)
    dir = joinpath(PROJECT_ROOT_KC, "results", subfolder)
    isdir(dir) || return nothing

    κ_raw = Float64[]; f0_raw = Float64[]; b_raw = Float64[]
    for rd in sort(filter(d -> match(r"^run_\d+$", d) !== nothing,
                          readdir(dir, join = false)))
        op = joinpath(dir, rd, "observables.jld2")
        pp = joinpath(dir, rd, "params.toml")
        isfile(op) && isfile(pp) || continue
        ep = TOML.parsefile(pp)["params"]
        κ  = Float64(ep["kappa"]); f₀ = Float64(ep["f0"])
        bv = jldopen(op, "r") do f; haskey(f, "b") ? f["b"] : nothing; end
        bv === nothing && continue
        n  = length(bv)
        push!(κ_raw, κ); push!(f0_raw, f₀)
        push!(b_raw, mean(bv[max(1, n - LAST_FRAMES + 1):n]))
    end
    isempty(b_raw) && return nothing

    kv = sort(unique(κ_raw)); fv = sort(unique(f0_raw))
    lk = Dict(v => i for (i, v) in enumerate(kv))
    lf = Dict(v => i for (i, v) in enumerate(fv))
    bm = fill(NaN, length(kv), length(fv))
    for (κ, f₀, b) in zip(κ_raw, f0_raw, b_raw)
        bm[lk[κ], lf[f₀]] = b
    end
    @printf("  %-30s  %d κ × %d f₀  (%d runs)\n", subfolder, length(kv), length(fv), length(b_raw))
    return (kv, fv, bm)
end

println("Loading datasets…")
loaded = Tuple{String, NTuple{3, Any}}[]
for (sub, label) in DATASETS
    d = load_dataset(sub)
    d !== nothing && push!(loaded, (label, d))
end
println("Loaded $(length(loaded)) / $(length(DATASETS)) datasets.\n")
isempty(loaded) && error("No data loaded.")

# ── Align all matrices to the reference grid ──────────────────────────────────
kappa_ref, f0_ref, _ = loaded[1][2]

function align(kv, fv, bm, kref, fref)
    length(kv) == length(kref) && length(fv) == length(fref) && return bm
    lk = Dict(round(v, sigdigits = 10) => i for (i, v) in enumerate(kv))
    lf = Dict(round(v, sigdigits = 10) => i for (i, v) in enumerate(fv))
    m  = fill(NaN, length(kref), length(fref))
    for (ic, κ) in enumerate(kref), (ir, f₀) in enumerate(fref)
        ik = get(lk, round(κ,  sigdigits = 10), 0)
        if_ = get(lf, round(f₀, sigdigits = 10), 0)
        ik > 0 && if_ > 0 && (m[ic, ir] = bm[ik, if_])
    end
    m
end

matrices = [align(kv, fv, bm, kappa_ref, f0_ref) for (_, (kv, fv, bm)) in loaded]
labels   = [lbl for (lbl, _) in loaded]
colors   = IC_COLORS[1:length(loaded)]

all_b = filter(!isnan, vcat(matrices...))
cmax  = isempty(all_b) ? 1.0 : min(1.0, maximum(all_b))

# ── Average and spread (std dev) across ICs ───────────────────────────────────
stacked = cat(matrices...; dims = 3)
b_avg   = mapslices(stacked; dims = 3) do v
    v2 = filter(!isnan, v); isempty(v2) ? NaN : mean(v2)
end[:, :, 1]
b_std   = mapslices(stacked; dims = 3) do v
    v2 = filter(!isnan, v); length(v2) >= 2 ? std(v2) : NaN
end[:, :, 1]

# ── Linear stability boundary (optional) ─────────────────────────────────────
function load_stability()
    for N_try in [17, 16]
        p = joinpath(PROJECT_ROOT_KC, "results", "linear_stability",
                     "phase_boundary_N$(N_try).csv")
        isfile(p) || continue
        rows    = [split(r, ",") for r in readlines(p)[2:end] if !isempty(r)]
        f0_vals = [parse(Float64, r[2]) for r in rows]
        κ_vals  = [parse(Float64, r[1]) for r in rows]
        return f0_vals, κ_vals
    end
    return nothing, nothing
end
f0_stab, κ_stab = load_stability()
f0_stab !== nothing && println("  Stability boundary loaded.")

function add_stability!(ax)
    f0_stab === nothing && return
    mask = (f0_stab .>= f0_ref[1])   .& (f0_stab .<= f0_ref[end]) .&
           (κ_stab  .>= kappa_ref[1]) .& (κ_stab  .<= kappa_ref[end])
    any(mask) || return
    lines!(ax, f0_stab[mask], κ_stab[mask];
           color = :white, linewidth = 2, linestyle = :dash)
end

# ── Figure 1: 2×3 side-by-side heatmap grid ──────────────────────────────────
println("\nBuilding grid figure…")

nc = 3; nr = cld(length(loaded), nc)
fig1 = Figure(size = (nc * 400 + 100, nr * 380 + 80))
Label(fig1[0, 1:nc],
      "IC sensitivity — mean asphericity  b̄   (last 20 s, N=17, kBT=0.1)";
      fontsize = 14, tellwidth = false)

for (idx, (label, mat)) in enumerate(zip(labels, matrices))
    row = (idx - 1) ÷ nc + 1
    col = (idx - 1) % nc + 1
    ax  = Axis(fig1[row, col];
               title    = label, titlesize = 12,
               xlabel   = "f₀",  ylabel   = "κ",
               xscale   = log10, yscale   = log10,
               xticks   = LogTicks(WilkinsonTicks(5)),
               yticks   = LogTicks(WilkinsonTicks(5)))
    heatmap!(ax, f0_ref, kappa_ref, mat';
             colormap = :viridis, colorrange = (0.0, cmax), nan_color = :white)
    add_stability!(ax)
end

Colorbar(fig1[1:nr, nc + 1];
         colormap = :viridis, limits = (0.0, cmax),
         label = "mean b  (0 = circular,  1 = rod-like)", width = 18)

out1 = joinpath(PROJECT_ROOT_KC, "results", "kinked_comparison_grid.png")
save(out1, fig1, px_per_unit = 2)
println("Saved: $out1")

# ── Figure 2: contour overlay | average | IC spread ──────────────────────────
println("Building summary figure…")

fig2 = Figure(size = (1500, 560))
Label(fig2[0, 1:6],
      "IC sensitivity — overlay, average, and spread   ($(length(loaded)) variants)";
      fontsize = 14, tellwidth = false)

# Panel A: contour overlay — b = 0.25, 0.5, 0.75 for each IC
ax_ov = Axis(fig2[1, 1];
             title  = "Contour overlay  (b = 0.25 / 0.50 / 0.75)",
             xlabel = "f₀", ylabel = "κ",
             xscale = log10, yscale = log10,
             xticks = LogTicks(WilkinsonTicks(5)),
             yticks = LogTicks(WilkinsonTicks(5)))
legend_elems = LegendElement[]
for (label, mat, col) in zip(labels, matrices, colors)
    contour!(ax_ov, f0_ref, kappa_ref, mat';
             levels = [0.25, 0.50, 0.75], color = col, linewidth = 1.5)
    push!(legend_elems, LineElement(color = col, linewidth = 2.5))
end
add_stability!(ax_ov)
Legend(fig2[1, 2], legend_elems, labels;
       framevisible = false, labelsize = 11, rowgap = 4)

# Panel B: element-wise average
ax_avg = Axis(fig2[1, 3];
              title  = "Average  ⟨b⟩  across ICs",
              xlabel = "f₀", ylabel = "κ",
              xscale = log10, yscale = log10,
              xticks = LogTicks(WilkinsonTicks(5)),
              yticks = LogTicks(WilkinsonTicks(5)))
hm_avg = heatmap!(ax_avg, f0_ref, kappa_ref, b_avg';
                  colormap = :viridis, colorrange = (0.0, cmax), nan_color = :white)
add_stability!(ax_avg)
Colorbar(fig2[1, 4], hm_avg; label = "⟨b⟩", width = 18)

# Panel C: std dev (IC sensitivity map)
std_finite = filter(!isnan, b_std)
std_max    = isempty(std_finite) ? 0.1 : maximum(std_finite)
ax_std = Axis(fig2[1, 5];
              title  = "IC spread  σ(b)  — sensitivity to initial condition",
              xlabel = "f₀", ylabel = "κ",
              xscale = log10, yscale = log10,
              xticks = LogTicks(WilkinsonTicks(5)),
              yticks = LogTicks(WilkinsonTicks(5)))
hm_std = heatmap!(ax_std, f0_ref, kappa_ref, b_std';
                  colormap = :heat, colorrange = (0.0, std_max), nan_color = :white)
add_stability!(ax_std)
Colorbar(fig2[1, 6], hm_std; label = "σ(b)  — high = IC-sensitive", width = 18)

out2 = joinpath(PROJECT_ROOT_KC, "results", "kinked_comparison_summary.png")
save(out2, fig2, px_per_unit = 2)
println("Saved: $out2")

println("\nDone.")
