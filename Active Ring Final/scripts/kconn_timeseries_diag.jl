# kconn_timeseries_diag.jl
# Two single-run diagnostics on the Kconn phase diagrams (read-only — does NOT
# touch the reduction, the CSV cache, or the composite figure).
#
#   D1  Circle, speckled high-f0 cell: is Kconn wrapping near ±π (artifact) or
#       genuinely erratic (real dynamic phase)?  → raw Kconn vs t, guides at ±180°.
#   D2  Quadkinked, high-κ / low-f0 corner: is the inward (red) mean a true
#       attractor or an unrelaxed transient?  → raw Kconn vs t, guide at 2π/N.
#   D3  Quadkinked, low-to-mid-κ / low-f0 (middle of the solidly-red inward
#       region): is the inward mean a settled attractor or a slow transient still
#       creeping toward 2π/N by t=7.5 s?  → raw Kconn vs t, guide at 2π/N.
#
# Kconn is stored in RADIANS (a wrapped turning angle, already in (-π, π]); plots
# are in degrees.  Runs are 7.5 s of sim-time (1500 frames × 0.005 s); the dashed
# vertical line marks the reduction's burn-in cutoff so you can see what the mean
# is computed over.
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/kconn_timeseries_diag.jl

using CairoMakie, Statistics, Printf

const PROJECT_ROOT_TD = joinpath(@__DIR__, "..")
include(joinpath(PROJECT_ROOT_TD, "src", "EngineRing.jl"))
using .EngineRing
include(joinpath(PROJECT_ROOT_TD, "src", "IOUtils.jl"))
using .IOUtils

const BURN_IN_T = 2.5      # matches reduce_kconn.jl
const OUT_DIR   = joinpath(PROJECT_ROOT_TD, "results", "kconn_phase")

# ── Plot one run's raw Kconn(t) ───────────────────────────────────────────────
function plot_run(folder, run_id; title, guides, outfile)
    dir = joinpath(PROJECT_ROOT_TD, "results", folder, run_id)
    times, obs = IOUtils.load_observables(dir)
    K = rad2deg.(obs["Kconn"])          # degrees

    fig = Figure(size = (1000, 520))
    ax  = Axis(fig[1, 1]; title = title,
               xlabel = "t  (s)   — full run = $(round(times[end]; digits=1)) s sim-time",
               ylabel = "Kconn at passive bead  (°)")
    lines!(ax, times, K; color = :black, linewidth = 0.8)   # raw, unsmoothed
    for (val, lab, col) in guides
        hlines!(ax, [val]; color = col, linestyle = :dash, linewidth = 1.5, label = lab)
    end
    vlines!(ax, [BURN_IN_T]; color = :grey50, linestyle = :dot, linewidth = 1.5,
            label = "burn-in cutoff ($(BURN_IN_T) s)")
    axislegend(ax; position = :rt, framevisible = true, labelsize = 10)

    save(outfile, fig, px_per_unit = 2)
    return times, K
end

println("=" ^ 70)
println("D1 — Circle, speckled high-f0 cell (run_1089, κ≈2.94, f0≈33.7)")
println("    Question: wrapping artifact (smooth swing crossing ±180°) vs real chaos?")
println("=" ^ 70)
t1, K1 = plot_run("Results_N17_circle", "run_1089";
    title = "D1  Circle  κ≈2.94, f₀≈33.7  —  raw Kconn (wrapping check)",
    guides = [(180.0, "+180° (wrap)", :red), (-180.0, "−180° (wrap)", :red)],
    outfile = joinpath(OUT_DIR, "diag1_circle_wrapping.png"))

# How close does it get to the ±180° wrap boundary, and how many wrap-like jumps?
absmax   = maximum(abs, K1)
near_pi  = count(>(170.0), abs.(K1))
dK       = abs.(diff(K1))
big_jump = count(>(180.0), dK)           # |Δ| > 180° between frames ⇒ a wrap flip
@printf("  min=%.1f°  max=%.1f°  range=%.1f°  max|K|=%.1f°\n",
        minimum(K1), maximum(K1), maximum(K1) - minimum(K1), absmax)
@printf("  frames with |K|>170° (hugging ±180): %d / %d\n", near_pi, length(K1))
@printf("  frame-to-frame jumps |ΔK|>180° (wrap flips): %d\n", big_jump)
@printf("  approaches ±π?  %s\n", absmax > 175.0 ? "YES (within 5° of ±180°)" :
                                  absmax > 160.0 ? "close (within 20°)" : "no")
println("  Saved: $(joinpath(OUT_DIR, "diag1_circle_wrapping.png"))")

println("\n" * "=" ^ 70)
println("D2 — Quadkinked, high-κ/low-f0 corner (run_0030, κ=50, f0=1)")
println("    Question: inward attractor vs unrelaxed transient (drift toward 2π/N)?")
println("=" ^ 70)
Keq = rad2deg(2π / 17)
t2, K2 = plot_run("Results_N17_quadkinked", "run_0030";
    title = "D2  Quadkinked  κ=50, f₀=1  —  raw Kconn (relaxation check)",
    guides = [(Keq, "2π/N = $(round(Keq; digits=1))° (circular)", :green)],
    outfile = joinpath(OUT_DIR, "diag2_quadkinked_relaxation.png"))

# Initial vs late: did it settle, and is it still drifting at the end?
init_val = K2[1]
early    = mean(K2[(t2 .>= BURN_IN_T) .& (t2 .< (BURN_IN_T + 1.5))])   # just after burn-in
late     = mean(K2[t2 .>= (t2[end] - 1.5)])                            # last 1.5 s
@printf("  initial Kconn (t=%.3f s): %.1f°\n", t2[1], init_val)
@printf("  mean over [%.1f, %.1f] s (early post-burn-in): %.1f°\n", BURN_IN_T, BURN_IN_T+1.5, early)
@printf("  mean over last 1.5 s:                          %.1f°\n", late)
@printf("  drift early→late: %.2f°   |   distance of late mean from 2π/N: %.1f°\n",
        late - early, late - Keq)
println("  Saved: $(joinpath(OUT_DIR, "diag2_quadkinked_relaxation.png"))")

println("\n" * "=" ^ 70)
println("D3 — Quadkinked, middle of the red inward region (run_0004, κ≈1.50, f0=1)")
println("    Question: settled inward attractor vs slow transient still drifting")
println("              toward 2π/N at t=7.5 s (the red region lives at LOW κ, so")
println("              bending is weak — is the kink held or is it just relaxing slowly)?")
println("=" ^ 70)
t3, K3 = plot_run("Results_N17_quadkinked", "run_0004";
    title = "D3  Quadkinked  κ≈1.50, f₀=1  —  raw Kconn (inward-attractor check)",
    guides = [(Keq, "2π/N = $(round(Keq; digits=1))° (circular)", :green)],
    outfile = joinpath(OUT_DIR, "diag3_quadkinked_inward_attractor.png"))

# Same read as D2: early (just after burn-in) vs late (last 1.5 s), and how far
# the late mean sits from the circular value 2π/N.  Inward is Kconn < 2π/N.
init_val3 = K3[1]
early3    = mean(K3[(t3 .>= BURN_IN_T) .& (t3 .< (BURN_IN_T + 1.5))])   # just after burn-in
late3     = mean(K3[t3 .>= (t3[end] - 1.5)])                            # last 1.5 s
@printf("  initial Kconn (t=%.3f s): %.1f°\n", t3[1], init_val3)
@printf("  mean over [%.1f, %.1f] s (early post-burn-in): %.1f°\n", BURN_IN_T, BURN_IN_T+1.5, early3)
@printf("  mean over last 1.5 s:                          %.1f°\n", late3)
@printf("  drift early→late: %.2f°   |   distance of late mean from 2π/N: %.1f° (negative ⇒ inward)\n",
        late3 - early3, late3 - Keq)
# The failure mode is specifically RELAXING toward 2π/N, so the read must be
# sign-aware: a large negative drift means the mean went DEEPER inward (away from
# circular), which supports the attractor, not against it.
gap_late    = Keq - late3                       # >0 ⇒ late mean sits inward of 2π/N
drift3      = late3 - early3                     # >0 ⇒ moving up toward 2π/N
toward_circ = drift3 > 5.0 && gap_late < 20.0    # creeping up and already near 2π/N
@printf("  read: %s\n",
        toward_circ  ? "STILL DRIFTING toward 2π/N ⇒ run length is the problem" :
        gap_late > 15.0 ? "CLEARLY INWARD & not relaxing toward 2π/N ⇒ genuine attractor (bistability solid)" :
                          "ambiguous — inspect the plot")
println("  Saved: $(joinpath(OUT_DIR, "diag3_quadkinked_inward_attractor.png"))")
