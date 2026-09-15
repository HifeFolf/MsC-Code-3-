# phase_reliability.jl
# Scores a phase classifier on how RELIABLE its diagram is, instead of asking a
# human whether the picture looks right.
#
# Why this exists:
#   Every classifier this project has tried (amplitude → equipartition →
#   persistence → std → route) looked correct until someone examined neighbouring
#   cells by eye.  Eyeballing is the bottleneck, so reliability is made a number.
#   Any candidate classifier is then a pure function of a cached feature row and
#   can be compared against every other one on identical input.
#
# The two measurements, and the data each needs:
#
#   REPRODUCIBILITY — same (κ, f₀), different noise, same label?
#       Needs repeated seeds at one cell.  Results_N17_circle_targeted has
#       243 cells × 10 independent seeds (1001–1010), sitting on the contested
#       direction-selection cells.  Reported as the modal-label fraction per
#       cell: 1.0 = all ten seeds agree, 0.5 = a coin flip.
#       This is the number that says whether a SINGLE-seed label can be trusted —
#       which matters because the production diagram is single-seed per cell.
#
#   COHERENCE — does a cell agree with its 8 neighbours in (κ, f₀)?
#       Needs a full grid with independent noise per cell.  Results_N17_circle_indep
#       is post-seeding-fix (each run has its own seed_effective), so neighbouring
#       cells are independent samples and the speckle measured here is real, not
#       an artifact of a shared noise stream.
#
#   Reading them TOGETHER is what separates the two failure modes:
#       low reproducibility + low coherence → genuine stochasticity (physics)
#       high reproducibility + low coherence → real fine boundary structure
#       low reproducibility + high coherence → unstable classifier (our problem)
#
# Note the scoring deliberately REWARDS admitting undecidability.  Where the ring
# has not finished choosing at t = 7.5 s, a classifier that forces a binary label
# scores ~0.5 reproducibility; one that emits "Unsettled" scores ~1.0, because all
# ten seeds agree it hasn't settled.  Honest abstention beats a confident guess.
#
# Requires the feature cache — build it first:
#   julia --project scripts/reduce_phase_features.jl --all
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/phase_reliability.jl              # score all, plot best
#   julia --project scripts/phase_reliability.jl route        # plot a named one
#   julia --project scripts/phase_reliability.jl --no-plot
#
# Output:
#   results/Results_N17_circle_indep/phase_reliability_<clf>.png
#   results/Results_N17_circle_indep/phase_reliability_<clf>.csv

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_PR = joinpath(@__DIR__, "..")

# ── Data sources (overridable) ───────────────────────────────────────────────
#   --grid F      folder supplying the label map for neighbour coherence
#   --seeds F     folder supplying repeated seeds for reproducibility
#   --features N  which cached CSV to read (e.g. phase_features_t7.5.csv, so a
#                 campaign can be scored at several analysis cutoffs)
# The two folders may be THE SAME: a campaign with many seeds per cell supplies
# both roles at once, and then reproducibility is measured directly rather than
# inferred from coherence.
function argval_pr(flag::String, default::String)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i + 1]
end
const GRID_FOLDER   = argval_pr("--grid",  "Results_N17_circle_indep")
const SEED_FOLDER   = argval_pr("--seeds", "Results_N17_circle_targeted")
const FEATURES_FILE = argval_pr("--features", "phase_features.csv")
# Tag outputs with the cutoff so scoring the same folder at several analysis
# cutoffs does not overwrite itself ("phase_features_t7.5.csv" → "_t7.5").
const OUT_TAG = FEATURES_FILE == "phase_features.csv" ? "" :
    replace(replace(FEATURES_FILE, "phase_features" => ""), ".csv" => "")

# ── Canonical label vocabulary ───────────────────────────────────────────────
# Every classifier maps into this same set so their scores are comparable.
# 0 is the abstention class; a classifier that never abstains simply never emits it.
const LABEL_NAMES = Dict(0 => "Unsettled / undecided",
                         1 => "Slithering / chaotic",
                         2 => "Crescent (folded)",
                         3 => "Ellipse / glider")
const LABEL_COLORS = Dict(0 => :gray65, 1 => :crimson, 2 => :darkorange, 3 => :aliceblue)
const LABEL_ORDER  = [0, 1, 2, 3]

# ── Thresholds (mirrored from the scripts each classifier reproduces) ────────
const BI_FRAC        = 0.15   # ≥ this fraction in EACH Kconn well
const INWARD_DEV     = -8.0   # ⟨Kconn⟩−eq below this ⇒ settled inward
const RG_CURLED_FRAC = 0.70   # Rg < this × (N/2π) ⇒ folded onto itself
const B_SLITHER      = 0.04   # local std(b) above this ⇒ writhing, not gliding
const B_LOCKED       = 0.01   # local std(b) below this ⇒ genuinely at rest
const LEGACY_S_DYN   = 15.0   # the retired std(Kconn) = 15° "dynamic" cut
const CAPTURE_MARGIN = 1.0    # folded only within this many s of the end ⇒ still
                              # being captured, so its final label is a snapshot

# ══ Feature-cache reader (no CSV.jl in this project's deps) ═════════════════

const STRING_COLS = ("folder", "run")

function read_features(folder::String)
    path = joinpath(PROJECT_ROOT_PR, "results", folder, FEATURES_FILE)
    isfile(path) || error("Missing feature cache: $path\n" *
                          "Build it with: julia --project scripts/reduce_phase_features.jl --all")
    lines = readlines(path)
    length(lines) < 2 && error("Feature cache is empty: $path")
    hdr  = String.(split(lines[1], ','))
    syms = Tuple(Symbol.(hdr))
    rows = Vector{NamedTuple}()
    for ln in @view lines[2:end]
        isempty(strip(ln)) && continue
        parts = split(ln, ',')
        length(parts) == length(hdr) || continue
        vals = ntuple(i -> hdr[i] in STRING_COLS ? String(parts[i]) :
                           something(tryparse(Float64, parts[i]), NaN),
                      length(hdr))
        push!(rows, NamedTuple{syms}(vals))
    end
    return rows
end

# ══ Classifiers ═════════════════════════════════════════════════════════════
# Each is  row → Int  in the canonical vocabulary above.

rg_threshold(r) = RG_CURLED_FRAC * (r.N / (2π))
safe(x, fallback) = isnan(x) ? fallback : x

"""
Current model, as implemented in phase_diagram_kconn_states.jl — the baseline
every candidate has to beat.  Never abstains: every cell is forced into 1/2/3.
"""
function clf_route(r)
    min(r.f_in, r.f_out) > BI_FRAC              && return 1
    (r.dev < INWARD_DEV || r.Rg_med < rg_threshold(r)) && return 2
    return safe(r.bstd, 0.0) > B_SLITHER ? 1 : 3
end

"""
The route model with an explicit abstention class.

Two situations get 0 rather than a guess:
  • the ring is still changing shape at the buzzer and has NOT folded — whatever
    label it has is a snapshot of an ongoing process, not a resting state;
  • it IS folded, but capture happened inside the last CAPTURE_MARGIN seconds —
    the run ended mid-transition, so the same cell with different noise could
    plausibly still be unfolded.

This is the direct consequence of the capture result: in the high-Φ corner what a
cell shows at 7.5 s is largely WHEN capture happened, and no run length fixes it.
"""
function clf_route_undecided(r)
    folded = r.folded_end > 0.5
    if folded
        !isnan(r.t_col) && (r.t_end - r.t_col) < CAPTURE_MARGIN && return 0
    elseif safe(r.bstd_fin, 0.0) > B_SLITHER
        return 0
    end
    return clf_route(r)
end

"""
Final-state model (phase_diagram_final_state.jl): label only by what the ring is
doing in the last second, with no reference to the route it took.
"""
function clf_final(r)
    r.Rg_fin < rg_threshold(r) && return 2
    return safe(r.bstd_fin, 0.0) > B_LOCKED ? 1 : 3
end

"""
The retired std(Kconn) = 15° amplitude cut, kept as a control.  If the scoring is
measuring anything real, this should come out clearly worse — it is the model the
project already rejected on physical grounds.
"""
function clf_legacy_std(r)
    r.dev < INWARD_DEV && return 2
    return r.K_std > LEGACY_S_DYN ? 1 : 3
end

const CLASSIFIERS = ["route"            => clf_route,
                     "route_undecided"  => clf_route_undecided,
                     "final"            => clf_final,
                     "legacy_std"       => clf_legacy_std]

# ══ Scoring ═════════════════════════════════════════════════════════════════

cellkey(r) = (round(r.kappa, sigdigits = 8), round(r.f0, sigdigits = 8))

"Modal value and its fraction of `v`."
function modal(v::Vector{Int})
    best, bestn = v[1], 0
    for c in unique(v)
        n = count(==(c), v)
        n > bestn && ((best, bestn) = (c, n))
    end
    return best, bestn / length(v)
end

"""
Per-cell reproducibility over repeated seeds.
Returns (mean_agreement, cells) where each cell is (κ, f0, label, agreement, n).
"""
function reproducibility(rows, clf)
    byc = Dict{Tuple{Float64, Float64}, Vector{Int}}()
    for r in rows
        push!(get!(byc, cellkey(r), Int[]), clf(r))
    end
    cells = NamedTuple[]
    for (k, labs) in byc
        length(labs) < 2 && continue       # a lone run cannot measure agreement
        lab, frac = modal(labs)
        push!(cells, (kappa = k[1], f0 = k[2], label = lab,
                      agree = frac, n = length(labs)))
    end
    isempty(cells) && return (NaN, cells)
    return (mean(c.agree for c in cells), cells)
end

"""
Neighbour coherence on the full grid: for each cell, the fraction of its existing
8 neighbours carrying the same label.  Returns (mean, κ values, f₀ values,
label grid, coherence grid).  Empty cells are -1 in the label grid, NaN in coherence.
"""
function coherence(rows, clf)
    κs = sort(unique(round(r.kappa, sigdigits = 8) for r in rows))
    fs = sort(unique(round(r.f0,    sigdigits = 8) for r in rows))
    ki = Dict(v => i for (i, v) in enumerate(κs))
    fi = Dict(v => i for (i, v) in enumerate(fs))

    # Collapse to ONE label per cell. With a single seed per cell this is the
    # label itself; with several it is the modal vote — otherwise the last row
    # read would silently decide the cell.
    byc = Dict{Tuple{Float64,Float64}, Vector{Int}}()
    for r in rows
        push!(get!(byc, cellkey(r), Int[]), clf(r))
    end
    L = fill(-1, length(κs), length(fs))
    for (k, labs) in byc
        L[ki[k[1]], fi[k[2]]] = modal(labs)[1]
    end

    C = fill(NaN, size(L))
    for i in axes(L, 1), j in axes(L, 2)
        L[i, j] < 0 && continue
        same = tot = 0
        for di in -1:1, dj in -1:1
            (di == 0 && dj == 0) && continue
            ii, jj = i + di, j + dj
            (1 ≤ ii ≤ size(L, 1) && 1 ≤ jj ≤ size(L, 2)) || continue
            L[ii, jj] < 0 && continue
            tot += 1
            same += (L[ii, jj] == L[i, j])
        end
        tot > 0 && (C[i, j] = same / tot)
    end
    return (mean(filter(!isnan, C)), κs, fs, L, C)
end

# ══ Run the comparison ══════════════════════════════════════════════════════

# Positional args = classifier name. Skip flags AND the values they consume.
args = String[]
let skip = false
    for a in ARGS
        if skip
            skip = false
        elseif startswith(a, "--")
            a in ("--grid", "--seeds", "--features") && (skip = true)
        else
            push!(args, a)
        end
    end
end
do_plot = !("--no-plot" in ARGS)

println("Reading feature caches …")
grid_rows = read_features(GRID_FOLDER)
seed_rows = read_features(SEED_FOLDER)
@printf("  %-32s %5d runs\n", GRID_FOLDER, length(grid_rows))
@printf("  %-32s %5d runs\n", SEED_FOLDER, length(seed_rows))

nseeds = length(unique(r.seed for r in seed_rows))
ncells = length(unique(cellkey(r) for r in seed_rows))
@printf("  reproducibility set: %d cells × up to %d seeds\n\n", ncells, nseeds)

println("╔══════════════════════════════════════════════════════════════════════════╗")
println("║  classifier          reproducibility   coherence   abstained   n labels  ║")
println("╠══════════════════════════════════════════════════════════════════════════╣")

scores = Dict{String, Any}()
for (name, clf) in CLASSIFIERS
    rep, cells = reproducibility(seed_rows, clf)
    coh, κs, fs, L, C = coherence(grid_rows, clf)
    absts = count(r -> clf(r) == 0, grid_rows) / length(grid_rows)
    nlab  = length(unique(filter(≥(0), vec(L))))
    scores[name] = (rep = rep, cells = cells, coh = coh,
                    κs = κs, fs = fs, L = L, C = C, abst = absts)
    @printf("║  %-18s %11.3f %13.3f %10.1f%% %9d  ║\n",
            name, rep, coh, 100absts, nlab)
end
println("╚══════════════════════════════════════════════════════════════════════════╝")

# Per-classifier label census on the production grid
println("\nLabel census on $GRID_FOLDER:")
for (name, clf) in CLASSIFIERS
    counts = [(c, count(r -> clf(r) == c, grid_rows)) for c in LABEL_ORDER]
    parts = ["$(LABEL_NAMES[c]) $n" for (c, n) in counts if n > 0]
    @printf("  %-18s %s\n", name, join(parts, " · "))
end

# Where does the winner remain untrustworthy?
chosen = isempty(args) ? "route_undecided" : args[1]
haskey(scores, chosen) || error("Unknown classifier '$chosen'. " *
                                "Known: $(join(first.(CLASSIFIERS), ", "))")
s = scores[chosen]
weak = sort(filter(c -> c.agree < 0.7, s.cells), by = c -> c.agree)
@printf("\n'%s': %d / %d seeded cells below 0.7 agreement", chosen, length(weak), length(s.cells))
if !isempty(weak)
    println(" — least reliable:")
    for c in first(weak, min(8, length(weak)))
        @printf("    κ=%7.3f  f₀=%8.3f   agree %.2f (n=%d)  modal: %s\n",
                c.kappa, c.f0, c.agree, c.n, LABEL_NAMES[c.label])
    end
else
    println(".")
end

# ── Per-cell scores to CSV ───────────────────────────────────────────────────
out_dir = joinpath(PROJECT_ROOT_PR, "results", GRID_FOLDER)
csv_p = joinpath(out_dir, "phase_reliability_$(chosen)$(OUT_TAG).csv")
open(csv_p, "w") do io
    println(io, "kappa,f0,modal_label,label_name,agreement,n_seeds")
    for c in sort(s.cells, by = x -> (x.kappa, x.f0))
        @printf(io, "%.8g,%.8g,%d,%s,%.4f,%d\n",
                c.kappa, c.f0, c.label, LABEL_NAMES[c.label], c.agree, c.n)
    end
end
println("\n✓ per-cell reproducibility → $(relpath(csv_p, PROJECT_ROOT_PR))")

# ── Per-RUN labels to CSV (--dump-runs) ──────────────────────────────────────
# The per-cell file above says a cell is a coin flip; this one says WHICH seeds
# went which way, so specific trajectories can be pulled and watched.  Written
# with the same classifier the scores use, so the two can never disagree.
if "--dump-runs" in ARGS
    runs_p = joinpath(out_dir, "phase_runlabels_$(chosen)$(OUT_TAG).csv")
    clf_chosen = Dict(CLASSIFIERS)[chosen]
    open(runs_p, "w") do io
        println(io, "folder,run,seed,kappa,f0,label,label_name")
        for r in sort(seed_rows, by = x -> (x.kappa, x.f0, x.seed))
            c = clf_chosen(r)
            @printf(io, "%s,%s,%d,%.8g,%.8g,%d,%s\n",
                    r.folder, r.run, Int(r.seed), r.kappa, r.f0, c, LABEL_NAMES[c])
        end
    end
    println("✓ per-run labels    → $(relpath(runs_p, PROJECT_ROOT_PR))")
end

# ── Grid coherence to CSV ────────────────────────────────────────────────────
# Doubles as the cell list for a follow-up seed campaign: the low-coherence rows
# are exactly the cells where a single-seed label cannot yet be trusted, and
# run_targeted_seeds.jl consumes a (κ, f₀) list in this shape.
grid_csv = joinpath(out_dir, "phase_coherence_$(chosen)$(OUT_TAG).csv")
nlow = count(c -> !isnan(c) && c < 0.7, s.C)
open(grid_csv, "w") do io
    println(io, "kappa,f0,label,label_name,coherence")
    for (i, κ) in enumerate(s.κs), (j, f) in enumerate(s.fs)
        s.L[i, j] < 0 && continue
        @printf(io, "%.8g,%.8g,%d,%s,%.4f\n",
                κ, f, s.L[i, j], LABEL_NAMES[s.L[i, j]], s.C[i, j])
    end
end
@printf("✓ grid coherence → %s  (%d cells below 0.7)\n",
        relpath(grid_csv, PROJECT_ROOT_PR), nlow)

# ── Figure ───────────────────────────────────────────────────────────────────
if do_plot
    κs, fs, L, C = s.κs, s.fs, s.L, s.C
    logκ, logf = log10.(κs), log10.(fs)

    # Plotted in log10 space on linear axes: even visual spacing across decades,
    # with ticks relabelled to the original values.
    ticks(v) = (range(minimum(v), maximum(v), length = 5),
                [@sprintf("%.3g", 10^x) for x in range(minimum(v), maximum(v), length = 5)])

    fig = Figure(size = (1680, 520))

    # 1 — the diagram itself
    ax1 = Axis(fig[1, 1], xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
               title = "Phase diagram — $chosen",
               xticks = ticks(logf), yticks = ticks(logκ))
    for c in LABEL_ORDER
        M = [L[i, j] == c ? 1.0 : NaN for i in eachindex(κs), j in eachindex(fs)]
        all(isnan, M) && continue
        heatmap!(ax1, logf, logκ, permutedims(M);
                 colormap = [LABEL_COLORS[c], LABEL_COLORS[c]], colorrange = (0, 1))
    end
    Legend(fig[2, 1],
           [PolyElement(color = LABEL_COLORS[c]) for c in LABEL_ORDER
            if any(==(c), vec(L))],
           [LABEL_NAMES[c] for c in LABEL_ORDER if any(==(c), vec(L))],
           orientation = :horizontal, framevisible = false, labelsize = 11)

    # 2 — neighbour coherence
    ax2 = Axis(fig[1, 2], xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
               title = @sprintf("Neighbour coherence  (mean %.3f)", s.coh),
               xticks = ticks(logf), yticks = ticks(logκ))
    hm2 = heatmap!(ax2, logf, logκ, permutedims(C);
                   colormap = :viridis, colorrange = (0, 1))
    Colorbar(fig[1, 3], hm2, label = "fraction of 8 neighbours agreeing", width = 13)

    # 3 — reproducibility, only where repeated seeds exist
    ax3 = Axis(fig[1, 4], xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ",
               title = @sprintf("Cross-seed reproducibility  (mean %.3f)", s.rep),
               xticks = ticks(logf), yticks = ticks(logκ))
    # full grid extent as context, so the seeded cells are placed in the diagram
    xlims!(ax3, minimum(logf), maximum(logf)); ylims!(ax3, minimum(logκ), maximum(logκ))
    sc = scatter!(ax3, [log10(c.f0) for c in s.cells], [log10(c.kappa) for c in s.cells];
                  color = [c.agree for c in s.cells], colormap = :plasma,
                  colorrange = (0.3, 1.0), markersize = 9)
    Colorbar(fig[1, 5], sc, label = "modal-label fraction over seeds", width = 13)

    png_p = joinpath(out_dir, "phase_reliability_$(chosen)$(OUT_TAG).png")
    save(png_p, fig; px_per_unit = 3)
    CairoMakie.activate!(type = "pdf"); save(replace(png_p, ".png" => ".pdf"), fig); CairoMakie.activate!(type = "png")
    println("✓ figure → $(relpath(png_p, PROJECT_ROOT_PR))")
end
