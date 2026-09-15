# phase_diagram_kconn_states.jl
# κ × f₀ phase diagram of the passive-junction turning angle Kconn, built to match
# the actual physics rather than force hard lines everywhere:
#
#   • Circle → Outward-ellipse is a *fade*, not a boundary.  As activity rises /
#     stiffness drops the circle smoothly elongates outward and (necessarily) moves.
#     This whole family is drawn as a GRADIENT in ⟨Kconn⟩−eq — no arbitrary internal
#     cut (so near-identical neighbours are never split by a threshold).
#
#   • Hard lines are drawn only where the physics is genuinely bistable:
#       – Inward-crumpled : the junction buckles inward and jams (⟨Kconn⟩ ≪ eq).
#       – Bistable/looping: the junction will not settle — it spends real time in
#         BOTH the inward and outward wells (e.g. flips +120°↔−135°).
#
# eq = 360/N.  Detection uses the times vector (≈200 Hz; the 60 fps is movie-only),
# restricted to the last TAIL_SECONDS of sim-time so the classification reflects the
# state the ring SETTLES into, not the route it took there.
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_kconn_states.jl
#   julia --project scripts/phase_diagram_kconn_states.jl Results_N17_circle
#   julia --project scripts/phase_diagram_kconn_states.jl Results_N17_circle 1 500

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT = joinpath(@__DIR__, "..")
const SKIP_SECONDS = 0.5    # drop brief IC-relaxation transient (sim-time)
const TAIL_SECONDS = 15.0   # classify on the LAST this-many seconds only (sim-time)
const IN_ABS_DEG   = 5.0    # Kconn below this            ⇒ sample is "well inward"
const OUT_OFF_DEG  = 20.0   # Kconn above eq+this         ⇒ sample is "well outward"
const BI_FRAC      = 0.15   # ≥ this fraction in EACH well ⇒ Bistable/looping
const INWARD_DEV   = -8.0   # ⟨Kconn⟩−eq below this        ⇒ settled Inward-crumpled

# ── Splitting the settled family by SIZE ─────────────────────────────────────
# ⟨Kconn⟩ alone cannot separate two physically different "outward" states: at very
# high activity the ring is driven so violently into itself that it folds onto
# itself, and the junction angle wraps round to read as strongly *outward* even
# though the shape is curled.  Radius of gyration separates them unambiguously:
#   open ring          Rg ≈ N/2π          (2.71 for N=17)
#   folded in half     Rg ≈ N/4π          (1.35 for N=17)
# The measured clusters sit at 2.67 and 1.33 with an empty valley between (only
# ~3 cells in Rg 1.7–2.1), so this cut is real, not imposed — any threshold from
# 55% to 75% of the circle value selects the same cells.
# The curled cells have Rg 1.33 vs 1.38 for the inward-crumpled (orange) family:
# geometrically the same folded state, opposite Kconn sign.
const RG_CURLED_FRAC = 0.70   # Rg < this × (N/2π) ⇒ folded onto itself

# ── Route to the crescent, not just the destination ──────────────────────────
# Every ring that folds ends in the crescent, so "is it folded at the end" says
# little.  What distinguishes the regimes is HOW it got there:
#   • forced straight in            — collapses within ~1 s, shape barely fluctuates
#   • slithered first               — snakes/writhes for seconds, then locks
#   • never gets there              — a steady gliding ellipse
# The separator is how much the shape WIGGLES before collapse — but it must be
# measured LOCALLY.  Taking std(b) over the whole pre-collapse window counts slow
# monotonic drift as if it were fluctuation, which wrongly paints two regions:
# the circle→ellipse transition line (b drifts as the ring elongates) and the
# slow low-f₀ collapses (b drifts as the ring folds).  Both look "high variance"
# while nothing is actually writhing.
#   median std(b) in 0.25 s windows:  slitherer 0.103 · slow drift 0.0155 · glider 0.0054
# The local measure separates by ~7×, where the global one does not separate at all.
const B_WIN_SECONDS = 0.25    # window for the local shape-fluctuation measure
const B_SLITHER     = 0.04    # median local std(b) above this ⇒ it slithered

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle_seed41"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Analysis window: the last TAIL_SECONDS of sim-time ───────────────────────
# Floored at SKIP_SECONDS so a run shorter than the tail still never feeds the
# IC-relaxation transient into the statistics.  Returns the first index in window.
function window_start(t)
    t0 = max(SKIP_SECONDS, t[end] - TAIL_SECONDS)
    something(findfirst(≥(t0), t), 1)
end

# ── Per-run state: classify by the ROUTE to the crescent ─────────────────────
# 1 = Slithering → crescent   (snakes about, then locks — or still snaking)
# 2 = Crescent, direct        (forced straight in, no slithering phase)
# 3 = Ellipse / glider        (steady, never folds)
#
# Note both folded outcomes are ONE state here.  Rg says the "inward-crumpled"
# (⟨Kconn⟩ ≈ −92°) and the "curled" (⟨Kconn⟩ ≈ +111°) cells are the same folded
# geometry (Rg 1.38 vs 1.33); Kconn's sign flips at extreme curvature and is not
# a physical distinction.
function ring_state(t, Kdeg, Rg, b, eq, N)
    i0   = window_start(t)
    nend = length(Rg)
    thr  = RG_CURLED_FRAC * (N / (2π))

    # collapse time: first entry into the folded state that LASTS to the end.
    # Deliberately scanned over the WHOLE trajectory, not just the tail window —
    # it is a diagnostic of when the fold began, which may predate the window.
    t_col = NaN
    if Rg[nend] < thr
        k = nend
        while k > 1 && Rg[k-1] < thr; k -= 1; end
        t_col = t[k]
    end

    seg   = @view Kdeg[i0:nend]
    f_in  = count(<(IN_ABS_DEG),       seg) / length(seg)
    f_out = count(>(eq + OUT_OFF_DEG), seg) / length(seg)
    dev   = mean(seg) - eq
    Rg_m  = median(@view Rg[i0:nend])

    # 1) Red, as originally detected — untouched.
    min(f_in, f_out) > BI_FRAC &&
        return (code = 1, dev = dev, Rg = Rg_m, bstd = NaN, t_col = t_col)

    # 2) Orange — untouched.  Folded by Rg and inward-by-Kconn are the same state
    #    (Rg 1.33 vs 1.38); Kconn's sign just flips at extreme curvature.
    (dev < INWARD_DEV || Rg_m < RG_CURLED_FRAC * (N / (2π))) &&
        return (code = 2, dev = dev, Rg = Rg_m, bstd = NaN, t_col = t_col)

    # 3) Everything left is the blue / ellipse family.  THIS is the only new step:
    #    split off the ones that are actually slithering rather than gliding.
    #    Median std(b) over short sliding windows — a slow drift in b as the ring
    #    elongates must not be mistaken for writhing (that mistake previously put
    #    red on the circle→ellipse transition line).
    dt   = nend > 1 ? (t[nend] - t[1]) / (nend - 1) : 1.0
    w    = max(3, round(Int, B_WIN_SECONDS / dt))
    bb   = @view b[i0:nend]
    bstd = length(bb) < 2w ? 0.0 :
           median(std(@view bb[i:i+w-1]) for i in 1:max(1, w ÷ 2):(length(bb) - w + 1))

    code = bstd > B_SLITHER ? 1 : 3
    return (code = code, dev = dev, Rg = Rg_m, bstd = bstd, t_col = t_col)
end

# ── Scan runs ─────────────────────────────────────────────────────────────────
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id_start ≤ parse(Int, m[1]) ≤ id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder…")

κ_raw = Float64[]; f0_raw = Float64[]; code_raw = Int[]; dev_raw = Float64[]
rat_raw = Float64[]                       # v_com / v_mean — translation fraction
win_lo = Inf; win_hi = -Inf               # effective analysis window actually used
for rd in run_dirs
    obs = joinpath(results_dir, rd, "observables.jld2")
    par = joinpath(results_dir, rd, "params.toml")
    (isfile(obs) && isfile(par)) || continue
    ep = TOML.parsefile(par)["params"]
    Nb = Int(ep["N"])
    eq = 360.0 / Nb
    d = jldopen(obs, "r") do f
        (haskey(f, "Kconn") && haskey(f, "Rg") && haskey(f, "b")) || return nothing
        m = length(f["Kconn"])
        t = haskey(f, "times") ? f["times"] : collect(1:m)
        (t = t, K = rad2deg.(f["Kconn"]), Rg = f["Rg"], b = f["b"],
         vc = haskey(f, "v_com")  ? f["v_com"]  : nothing,
         vm = haskey(f, "v_mean") ? f["v_mean"] : nothing)
    end
    d === nothing && continue
    length(d.K) < 10 && continue
    s = ring_state(d.t, d.K, d.Rg, d.b, eq, Nb)
    global win_lo = min(win_lo, d.t[window_start(d.t)])
    global win_hi = max(win_hi, d.t[end])

    # Translation fraction: COM speed / mean bead speed, over the same window.
    #   ≈1 → every bead moves together      (rigid translation — gliding ellipse)
    #   <1 → much of the motion is internal (rotation / churn — spinning crumple)
    # median, not mean: the periodic wrap shifts the whole ring by L in one frame
    # and would otherwise inject a huge spurious COM-speed spike.
    rat = NaN
    if d.vc !== nothing && d.vm !== nothing
        i0 = window_start(d.t)
        vcm = median(@view d.vc[i0:end]); vmm = median(@view d.vm[i0:end])
        rat = vmm > 0 ? vcm / vmm : NaN
    end

    push!(κ_raw, Float64(ep["kappa"])); push!(f0_raw, Float64(ep["f0"]))
    push!(code_raw, s.code); push!(dev_raw, s.dev); push!(rat_raw, rat)
end

n = length(code_raw)
println("Loaded $n runs with Kconn data.")
if n > 0
    @printf("Analysis window: last %.1f s requested → t ∈ [%.2f, %.2f] s used%s\n",
            TAIL_SECONDS, win_lo, win_hi,
            win_hi - win_lo < TAIL_SECONDS - 1e-9 ?
                "  (runs are shorter than the tail — whole trajectory after the transient)" : "")
end
n == 0 && error("No Kconn data found — was this sweep run with run_sweep.jl?")
labels = ["Slithering / chaotic", "Crescent (folded)", "Ellipse / glider"]
for c in 1:3
    @printf("  %-26s %d cells\n", labels[c], count(==(c), code_raw))
end

# ── κ × f₀ grid: gradient layer (settled dev) + categorical layer (1,2) ───────
kappa_vals = sort(unique(κ_raw)); f0_vals = sort(unique(f0_raw))
nκ = length(kappa_vals); nf = length(f0_vals)
kidx = Dict(v => i for (i, v) in enumerate(kappa_vals))
fidx = Dict(v => i for (i, v) in enumerate(f0_vals))

grad_mat = fill(NaN, nκ, nf)   # ⟨Kconn⟩−eq for the ELLIPSE family only
rat_mat  = fill(NaN, nκ, nf)   # v_com/v_mean for the ELLIPSE family only
cat_mat  = fill(NaN, nκ, nf)   # 1 = bistable, 2 = inward, 4 = curled (NaN else)
for i in 1:n
    ic = kidx[κ_raw[i]]; ir = fidx[f0_raw[i]]
    if code_raw[i] == 3
        grad_mat[ic, ir] = max(dev_raw[i], 0.0)   # circle(0) → outward(large)
        rat_mat[ic, ir]  = rat_raw[i]
    else
        cat_mat[ic, ir] = code_raw[i]
    end
end

# ── Figure ────────────────────────────────────────────────────────────────────
# Two views of the SAME classification.  The hard states are identical in both;
# only the shading of the ellipse family differs:
#   left  — SHAPE  : how far outward the ring sits   (⟨Kconn⟩−eq)
#   right — MOTION : how that shape moves            (v_com/v_mean)
# Those two stay gradients deliberately — each varies smoothly, so a hard cut
# inside them would split near-identical neighbours (the 496/526 problem).
# The curled/ellipse split is different: Rg is genuinely bimodal with an empty
# valley, so THAT one earns a hard boundary.
dev_hi = let s = filter(isfinite, grad_mat); isempty(s) ? 1.0 : quantile(s, 0.97) end
# 1 = slithering → crescent, 2 = crescent (direct).  3 = ellipse, drawn as the
# gradient instead, so it never appears in the categorical layer.
cat_colors = [:crimson, :darkorange, :white]

fig = Figure(size = (1750, 780))
Label(fig[0, 1:5], "Kconn states — $subfolder  (N runs = $n)";
      fontsize = 16, tellwidth = false)

mkax(col, ttl) = Axis(fig[1, col];
    xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ", title = ttl,
    xscale = log10, yscale = log10,
    xticks = LogTicks(WilkinsonTicks(6)), yticks = LogTicks(WilkinsonTicks(6)))

# ── Panel 1: shape — the circle→outward fade ─────────────────────────────────
ax = mkax(1, "SHAPE:  circle → outward fade")
hmg = heatmap!(ax, f0_vals, kappa_vals, grad_mat';
               colormap = :Blues, colorrange = (0.0, dev_hi),
               lowclip = :whitesmoke, nan_color = :transparent)
heatmap!(ax, f0_vals, kappa_vals, cat_mat';
         colormap = cgrad(cat_colors; categorical = true),
         colorrange = (0.5, 3.5), nan_color = :transparent)
Colorbar(fig[1, 2], hmg; label = "outward bend  ⟨Kconn⟩−eq  [deg]", width = 15)

# ── Panel 2: motion — spinning crumple → gliding ellipse ─────────────────────
ax2 = mkax(3, "MOTION:  rotating ↔ translating  (ellipse family)")
hmr = heatmap!(ax2, f0_vals, kappa_vals, rat_mat';
               colormap = :viridis, colorrange = (0.3, 1.0),
               lowclip = :black, nan_color = :transparent)
heatmap!(ax2, f0_vals, kappa_vals, cat_mat';
         colormap = cgrad(cat_colors; categorical = true),
         colorrange = (0.5, 3.5), nan_color = :transparent)
Colorbar(fig[1, 4], hmr;
         label = "v_com / v_mean    dark = rotates in place,  bright = glides", width = 15)

Legend(fig[1, 5],
       [PolyElement(color = cat_colors[c]) for c in (1, 2)],
       ["Slithering / chaotic", "Crescent (folded)"],
       "States"; framevisible = true)

out = joinpath(results_dir, "phase_diagram_kconn_states.png")
save(out, fig, px_per_unit = 2)
println("Saved: $out")
