# phase_diagram_seeds_aggregate_basin.jl
# Seed-AGGREGATED phase diagram for the N=17 circle sweep — basin-balance variant.
#
# Same idea as phase_diagram_seeds_aggregate.jl (pool all circle seed folders into
# a SINGLE aggregated diagram), but the MIDDLE panel is replaced:
#
#   1. ⟨b⟩                    — mean asphericity                 (viridis, 0…1)
#   2. Basin balance ⟨P_out − P_in⟩ — bimodal-safe junction sign (RdBu, −1…+1)
#   3. ⟨std(Kconn)⟩           — junction fluctuation amplitude   (viridis)
#
# WHY basin balance instead of the seed-mean ⟨Kconn⟩ angle:
#   Kconn is a BIMODAL variable in the bistable zone — each run sits near the
#   +outward well or the −inward well, and never in between.  Averaging the ANGLE
#   across seeds (and across time) therefore lands on pale intermediate values the
#   ring never actually occupies, which is exactly the misleading wash the old
#   mean panel showed.  Occupancy FRACTIONS average sensibly: reduce_kconn.jl
#   already reports, per run, the fraction of post-burn-in frames in each ±3σ_th
#   band around 2π/N (P_out / P_circ / P_in).  Their difference P_out − P_in ∈
#   [−1, +1] is a well-behaved basin-balance order parameter:
#       +1 → always outward,  0 → balanced / circular,  −1 → always inward.
#   We plot its seed-average per cell.
#
# Data sources (no new statistics, no reduction changes):
#   • ⟨b⟩ and ⟨std(Kconn)⟩  — computed from each run's observables.jld2 over the
#     steady-state window t ≥ BURN_IN_T (sliced from the `times` vector, matching
#     reduce_kconn.jl), then seed-averaged.  b is NOT in the Kconn CSV, so this
#     pass stays.  NOTE: the std panel is the seed-mean of each run's OWN temporal
#     std(Kconn) — std(K) per run, then mean across seeds — NOT the std of pooled
#     values.  (Confirmed unchanged from phase_diagram_seeds_aggregate.jl.)
#   • Basin balance + seed disagreement — read straight from the cached per-run
#     occupancy fractions in results/kconn_phase/circle.csv (all five seeds).
#
# Seed disagreement:  a cell is flagged when the five seeds do NOT all share one
# dominant basin (dominant = argmax of P_out / P_circ / P_in per run, same rule as
# phase_diagram_kconn_categorical.jl).  Flagged cells are stippled on the basin
# panel and written to results/kconn_phase/basin_disagreement_circle.csv (the
# shopping list if we later decide to buy extra seeds there — not acted on here).
#
# Steady-state window: drop every frame with t < BURN_IN_T sim-units, selected
# from the `times` vector — never a frame count (the stored framerate is a movie
# setting).  Runs are 7.5 sim-units / 1500 frames, so BURN_IN_T = 2.5 keeps the
# last 5 sim-units.  Matches reduce_kconn.jl exactly.
#
# Seeds: auto-discovered — every folder matching Results_N17_circle[_seedNN]
# (the trailing-digit "Results_N17_circle1" is deliberately excluded).
#
# Usage (from the Active Ring Final/ project root, AFTER reduce_kconn.jl has
# populated all five seeds into results/kconn_phase/circle.csv):
#   julia --project scripts/phase_diagram_seeds_aggregate_basin.jl
#
# Output:
#   results/phase_diagram_seeds_aggregate_basin.png
#   results/kconn_phase/basin_disagreement_circle.csv

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_AGG = joinpath(@__DIR__, "..")
const RESULTS_DIR      = joinpath(PROJECT_ROOT_AGG, "results")
const KCONN_DIR        = joinpath(RESULTS_DIR, "kconn_phase")

const BURN_IN_T      = 2.5                                   # sim-units (see header)
const SEED_FOLDER_RE = r"^Results_N17_circle(_seed\d+)?$"    # excludes "…circle1"
const STD_CLIP_Q     = 0.98                                  # std panel upper clip (heavy tail)

rk(x) = round(x; sigdigits = 10)                            # key guard vs float drift
seed_of(folder) = (m = match(r"_seed(\d+)$", folder); m === nothing ? "42" : m.captures[1])

# Dominant occupancy basin of one run (static rule of the categorical script:
# check circular first, then inward, else outward).
function dominant_basin(pout, pcirc, pin)
    m = max(pout, pcirc, pin)
    pcirc == m && return :circ
    pin   == m && return :in
    return :out
end

# ── CSV pass: basin balance + dominant-basin disagreement (from circle.csv) ────
function load_basin_records()
    path = joinpath(KCONN_DIR, "circle.csv")
    isfile(path) || error("Missing $path — run reduce_kconn.jl on the seed folders first.")
    lines = filter(!isempty, readlines(path))
    hdr   = split(lines[1], ",")
    ci(name) = findfirst(==(name), hdr)
    ic_i, κ_i, f_i, s_i, N_i = ci("ic"), ci("kappa"), ci("f0"), ci("seed"), ci("N")
    po_i, pc_i, pin_i        = ci("P_out"), ci("P_circ"), ci("P_in")
    recs = NamedTuple[]
    for ln in lines[2:end]
        v = split(ln, ",")
        (ic_i === nothing || v[ic_i] == "circle") || continue
        push!(recs, (κ = rk(parse(Float64, v[κ_i])), f0 = rk(parse(Float64, v[f_i])),
                     seed = parse(Int, v[s_i]), N = parse(Int, v[N_i]),
                     Pout = parse(Float64, v[po_i]), Pcirc = parse(Float64, v[pc_i]),
                     Pin  = parse(Float64, v[pin_i])))
    end
    isempty(recs) && error("No circle rows in $path")
    return recs
end

println("Loading basin occupancy from circle.csv…")
recs   = load_basin_records()
N      = recs[1].N
κs     = sort(unique(getproperty.(recs, :κ)))
f0s    = sort(unique(getproperty.(recs, :f0)))
nκ, nf = length(κs), length(f0s)
ik     = Dict(v => i for (i, v) in enumerate(κs))
jf     = Dict(v => i for (i, v) in enumerate(f0s))
csv_seeds = sort(unique(getproperty.(recs, :seed)))
@printf("  %d rows  |  %d κ × %d f₀  |  seeds: %s\n",
        length(recs), nκ, nf, join(csv_seeds, ", "))

bal_sum = zeros(nκ, nf); bal_cnt = zeros(Int, nκ, nf)
n_out   = zeros(Int, nκ, nf); n_circ = zeros(Int, nκ, nf); n_in = zeros(Int, nκ, nf)
for r in recs
    i, j = ik[r.κ], jf[r.f0]
    bal_sum[i, j] += (r.Pout - r.Pin); bal_cnt[i, j] += 1
    d = dominant_basin(r.Pout, r.Pcirc, r.Pin)
    d === :out ? (n_out[i, j] += 1) : d === :circ ? (n_circ[i, j] += 1) : (n_in[i, j] += 1)
end
balance   = [bal_cnt[i, j] > 0 ? bal_sum[i, j] / bal_cnt[i, j] : NaN for i in 1:nκ, j in 1:nf]
n_seed_c  = n_out .+ n_circ .+ n_in
# Disagreement: ≥2 seeds present AND more than one distinct dominant basin.
disagree  = [n_seed_c[i, j] >= 2 &&
             ((n_out[i, j] > 0) + (n_circ[i, j] > 0) + (n_in[i, j] > 0)) > 1
             for i in 1:nκ, j in 1:nf]

# ── Observables pass: ⟨b⟩ and ⟨std(Kconn)⟩ (unchanged from the base aggregate) ─
function load_folder_obs(sub)
    dir    = joinpath(RESULTS_DIR, sub)
    lookup = Dict{Tuple{Float64,Float64}, NTuple{2,Float64}}()   # (b_ss, std_Kconn)
    for rd in filter(d -> occursin(r"^run_\d+$", d), readdir(dir))
        rp       = joinpath(dir, rd)
        obs_file = joinpath(rp, "observables.jld2")
        pfile    = joinpath(rp, "params.toml")
        (isfile(obs_file) && isfile(pfile)) || continue
        ep = TOML.parsefile(pfile)["params"]
        κ  = rk(Float64(ep["kappa"])); f₀ = rk(Float64(ep["f0"]))
        data = jldopen(obs_file, "r") do f
            (haskey(f, "times") && haskey(f, "b") && haskey(f, "Kconn")) || return nothing
            (f["times"], f["b"], f["Kconn"])
        end
        data === nothing && continue
        times, b, K = data
        keep = findall(>=(BURN_IN_T), times)      # steady-state window by sim-time
        length(keep) < 2 && continue
        @views lookup[(κ, f₀)] = (mean(b[keep]), std(K[keep]))
    end
    @printf("  %-32s seed %-4s  %4d cells\n", sub, seed_of(sub), length(lookup))
    return lookup
end

seed_folders = sort(filter(d -> occursin(SEED_FOLDER_RE, d) && isdir(joinpath(RESULTS_DIR, d)),
                           readdir(RESULTS_DIR)))
println("\nAggregating asphericity + std from $(length(seed_folders)) seed folder(s):")
obs_lookups = [load_folder_obs(sf) for sf in seed_folders]

b_sum = zeros(nκ, nf); b_cnt = zeros(Int, nκ, nf)
s_sum = zeros(nκ, nf); s_cnt = zeros(Int, nκ, nf)
for lk in obs_lookups, ((κ, f0), (bv, sv)) in lk
    (haskey(ik, κ) && haskey(jf, f0)) || continue
    i, j = ik[κ], jf[f0]
    b_sum[i, j] += bv; b_cnt[i, j] += 1
    s_sum[i, j] += sv; s_cnt[i, j] += 1
end
b_mean    = [b_cnt[i, j] > 0 ? b_sum[i, j] / b_cnt[i, j] : NaN for i in 1:nκ, j in 1:nf]
k_std_deg = [s_cnt[i, j] > 0 ? rad2deg(s_sum[i, j] / s_cnt[i, j]) : NaN for i in 1:nκ, j in 1:nf]

@printf("\nGrid: %d κ × %d f₀   |   N=%d   |   obs seeds/cell: %d…%d   |   csv seeds/cell: %d…%d\n",
        nκ, nf, N,
        minimum(b_cnt), maximum(b_cnt), minimum(n_seed_c), maximum(n_seed_c))

# ── Disagreement stipple coords + shopping-list CSV ───────────────────────────
dis_f = Float64[]; dis_κ = Float64[]; dis_rows = NamedTuple[]
for i in 1:nκ, j in 1:nf
    disagree[i, j] || continue
    push!(dis_f, f0s[j]); push!(dis_κ, κs[i])
    push!(dis_rows, (kappa = κs[i], f0 = f0s[j],
                     n_out = n_out[i, j], n_circ = n_circ[i, j], n_in = n_in[i, j],
                     n_seeds = n_seed_c[i, j]))
end
sort!(dis_rows; by = r -> (r.kappa, r.f0))
dis_path = joinpath(KCONN_DIR, "basin_disagreement_circle.csv")
open(dis_path, "w") do io
    println(io, "kappa,f0,n_out,n_circ,n_in,n_seeds")
    for r in dis_rows
        println(io, join((r.kappa, r.f0, r.n_out, r.n_circ, r.n_in, r.n_seeds), ","))
    end
end
@printf("Basin-disagreement cells (seeds split on dominant basin): %d / %d\n",
        length(dis_rows), nκ * nf)
println("  wrote $dis_path")

# ── Colour normalisations ─────────────────────────────────────────────────────
alls    = filter(!isnan, vec(k_std_deg))
std_max = isempty(alls) ? 1.0 : quantile(alls, STD_CLIP_Q)
@printf("std range: 0°…%.2f°  (clipped at p%d)\n", std_max, round(Int, 100 * STD_CLIP_Q))

# ── Linear-stability overlay (clipped to the sampled range, as in the other
#    phase scripts) — draw only where the boundary CSV exists ──────────────────
function load_stability_boundary()
    for N_try in [N, 16, 17]
        csv = joinpath(RESULTS_DIR, "linear_stability", "phase_boundary_N$(N_try).csv")
        isfile(csv) || continue
        rows    = [split(r, ",") for r in readlines(csv)[2:end] if !isempty(r)]
        κ_stab  = [parse(Float64, r[1]) for r in rows]      # col 1 = κ
        f0_stab = [parse(Float64, r[2]) for r in rows]      # col 2 = f0
        inrng   = (f0_stab .>= f0s[1]) .& (f0_stab .<= f0s[end]) .&
                  (κ_stab  .>= κs[1]) .& (κ_stab  .<= κs[end])
        any(inrng) || continue
        println("  Stability boundary loaded (N=$N_try).")
        return (f0_stab[inrng], κ_stab[inrng])
    end
    return (nothing, nothing)
end
stab_f0, stab_κ = load_stability_boundary()
overlay_stability!(ax) = stab_κ === nothing ? nothing :
    lines!(ax, stab_f0, stab_κ; color = :black, linewidth = 2, linestyle = :dash)

# ── Figure: three panels in one row, each with its own colorbar ───────────────
fig = Figure(size = (1650, 560))
Label(fig[0, 1:6],
      "N=$N circle IC — seed-aggregated phase diagram (basin balance)   " *
      "($(length(csv_seeds)) seeds: $(join(csv_seeds, ", "));  steady state t ≥ $(BURN_IN_T), last 5 sim-units)";
      fontsize = 15, font = :bold, tellwidth = false)

function panel!(col, mat, title_str; cmap, crange, cbar_label, stipple = false)
    ax = Axis(fig[1, 2col - 1];
              title  = title_str,
              xlabel = "f₀  (active force)", ylabel = "κ  (bending stiffness)",
              xscale = log10, yscale = log10,
              xticks = LogTicks(WilkinsonTicks(6)), yticks = LogTicks(WilkinsonTicks(6)),
              limits = ((f0s[1], f0s[end]), (κs[1], κs[end])))   # f₀ clamped to sampled range
    hm = heatmap!(ax, f0s, κs, mat';
                  colormap = cmap, colorrange = crange, nan_color = :white)
    Colorbar(fig[1, 2col], hm; label = cbar_label, width = 14)
    overlay_stability!(ax)
    if stipple && !isempty(dis_f)
        scatter!(ax, dis_f, dis_κ; color = :black, marker = :x, markersize = 6)
    end
    return ax
end

panel!(1, b_mean, "Asphericity  ⟨b⟩";
       cmap = :viridis, crange = (0.0, 1.0), cbar_label = "⟨b⟩")
panel!(2, balance, "Basin balance (P_out − P_in)   (blue = outward, red = inward)";
       cmap = :RdBu, crange = (-1.0, 1.0), cbar_label = "⟨P_out − P_in⟩", stipple = true)
panel!(3, k_std_deg, "Junction fluctuation  ⟨std(Kconn)⟩";
       cmap = :viridis, crange = (0.0, max(std_max, 1e-3)), cbar_label = "std(Kconn)  (°)")

colgap!(fig.layout, 10)

out_path = joinpath(RESULTS_DIR, "phase_diagram_seeds_aggregate_basin.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
