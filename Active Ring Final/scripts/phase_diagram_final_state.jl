# phase_diagram_final_state.jl
# κ × f₀ phase diagram based ONLY on the state the ring is in at the END of the run.
#
# No route, no history, no time gating: every quantity is measured over the last
# FINAL_SECONDS of the simulation.  Whatever shape the ring has settled into (or
# happens to be in) at that moment is what the cell is labelled with.
#
# Two panels:
#   LEFT  — FINAL SHAPE.  Crescent/folded is a hard state (Rg separates it with a
#           real gap: an open ring has Rg ≈ N/2π = 2.71 for N=17, one folded in
#           half has ≈ 1.35).  Everything else is the circle→outward fade, drawn
#           as a gradient in ⟨Kconn⟩−eq because it genuinely is a continuum.
#   RIGHT  — IS IT ACTUALLY LOCKED?  Local std(b) inside the final window.  Dark
#           means frozen (the "final state" is a real resting state); bright means
#           the ring is still changing shape at the buzzer, so its label is just a
#           snapshot of an ongoing process.  Read the left panel through this one.
#
# Usage (from the Active Ring Final/ project root):
#   julia --project scripts/phase_diagram_final_state.jl
#   julia --project scripts/phase_diagram_final_state.jl Results_N17_circle_indep
#   julia --project scripts/phase_diagram_final_state.jl Results_N17_circle_indep 1 500

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT   = joinpath(@__DIR__, "..")
const FINAL_SECONDS  = 1.0    # length of the window at the END of the run
const RG_CURLED_FRAC = 0.70   # Rg < this × (N/2π) ⇒ folded onto itself
const B_WIN_SECONDS  = 0.25   # sub-window for the local "is it still moving" test
const B_LOCKED       = 0.01   # local std(b) below this ⇒ genuinely locked

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle_indep"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id_start ≤ parse(Int, m[1]) ≤ id_end
end)

println("Scanning $(length(run_dirs)) runs from $subfolder…")
println("Final state measured over the last $(FINAL_SECONDS) s of each run.\n")

κ_raw = Float64[]; f0_raw = Float64[]
dev_raw = Float64[]; rg_raw = Float64[]; wig_raw = Float64[]; folded_raw = Bool[]
for rd in run_dirs
    obs = joinpath(results_dir, rd, "observables.jld2")
    par = joinpath(results_dir, rd, "params.toml")
    (isfile(obs) && isfile(par)) || continue
    ep = TOML.parsefile(par)["params"]
    N  = Int(ep["N"]); eq = 360.0 / N
    d = jldopen(obs, "r") do f
        (haskey(f, "Kconn") && haskey(f, "Rg") && haskey(f, "b")) || return nothing
        m = length(f["Kconn"])
        t = haskey(f, "times") ? f["times"] : collect(1:m)
        (t = t, K = rad2deg.(f["Kconn"]), Rg = f["Rg"], b = f["b"])
    end
    d === nothing && continue
    n = length(d.K); n < 10 && continue

    # ── the final window ─────────────────────────────────────────────────────
    t_end = d.t[n]
    i1 = something(findfirst(≥(t_end - FINAL_SECONDS), d.t), max(1, n - 99))
    i1 = min(i1, n - 1)

    rg  = median(@view d.Rg[i1:n])
    dev = mean(@view d.K[i1:n]) - eq

    # still changing shape at the end?  local std(b) inside the final window
    dt  = n > 1 ? (t_end - d.t[1]) / (n - 1) : 1.0
    w   = max(3, round(Int, B_WIN_SECONDS / dt))
    bb  = @view d.b[i1:n]
    wig = length(bb) < 2w ? std(bb) :
          median(std(@view bb[i:i+w-1]) for i in 1:max(1, w ÷ 2):(length(bb) - w + 1))

    push!(κ_raw, Float64(ep["kappa"])); push!(f0_raw, Float64(ep["f0"]))
    push!(dev_raw, dev); push!(rg_raw, rg); push!(wig_raw, wig)
    push!(folded_raw, rg < RG_CURLED_FRAC * (N / (2π)))
end

n = length(dev_raw)
println("Loaded $n runs.")
n == 0 && error("No usable runs found in $results_dir")

n_fold = count(folded_raw)
n_lock = count(<(B_LOCKED), wig_raw)
@printf("  Crescent / folded at the end : %4d  (%.0f%%)\n", n_fold, 100n_fold/n)
@printf("  Open (circle → outward)      : %4d  (%.0f%%)\n", n - n_fold, 100(n-n_fold)/n)
@printf("  Genuinely locked at the end  : %4d  (%.0f%%)  ← rest are still moving\n",
        n_lock, 100n_lock/n)

# ── κ × f₀ grid ───────────────────────────────────────────────────────────────
kappa_vals = sort(unique(κ_raw)); f0_vals = sort(unique(f0_raw))
kidx = Dict(v => i for (i, v) in enumerate(kappa_vals))
fidx = Dict(v => i for (i, v) in enumerate(f0_vals))
grad_mat = fill(NaN, length(kappa_vals), length(f0_vals))   # open rings: dev
cat_mat  = fill(NaN, length(kappa_vals), length(f0_vals))   # 1 = folded
wig_mat  = fill(NaN, length(kappa_vals), length(f0_vals))
for i in 1:n
    ic = kidx[κ_raw[i]]; ir = fidx[f0_raw[i]]
    wig_mat[ic, ir] = wig_raw[i]
    if folded_raw[i]
        cat_mat[ic, ir] = 1.0
    else
        grad_mat[ic, ir] = max(dev_raw[i], 0.0)
    end
end

# ── Figure ────────────────────────────────────────────────────────────────────
dev_hi = let s = filter(isfinite, grad_mat); isempty(s) ? 1.0 : quantile(s, 0.97) end
fig = Figure(size = (1750, 780))
Label(fig[0, 1:5], "Final state (last $(FINAL_SECONDS) s) — $subfolder  (N runs = $n)";
      fontsize = 16, tellwidth = false)

mkax(col, ttl) = Axis(fig[1, col];
    xlabel = "Active force  f₀", ylabel = "Bending stiffness  κ", title = ttl,
    xscale = log10, yscale = log10,
    xticks = LogTicks(WilkinsonTicks(6)), yticks = LogTicks(WilkinsonTicks(6)))

ax1 = mkax(1, "FINAL SHAPE:  circle → outward fade,  folded in orange")
hmg = heatmap!(ax1, f0_vals, kappa_vals, grad_mat';
               colormap = :Blues, colorrange = (0.0, dev_hi),
               lowclip = :whitesmoke, nan_color = :transparent)
heatmap!(ax1, f0_vals, kappa_vals, cat_mat';
         colormap = cgrad([:darkorange]; categorical = true),
         colorrange = (0.5, 1.5), nan_color = :transparent)
Colorbar(fig[1, 2], hmg; label = "outward bend  ⟨Kconn⟩−eq  [deg]  (final)", width = 15)

ax2 = mkax(3, "IS IT LOCKED?  shape fluctuation in the final window")
hmw = heatmap!(ax2, f0_vals, kappa_vals, wig_mat';
               colormap = :magma, colorrange = (0.0, quantile(filter(isfinite, wig_raw), 0.97)),
               nan_color = :lightgrey)
Colorbar(fig[1, 4], hmw;
         label = "local std(b), final window    dark = locked,  bright = still moving",
         width = 15)

Legend(fig[1, 5], [PolyElement(color = :darkorange)],
       ["Crescent / folded"], "Final state"; framevisible = true)

out = joinpath(results_dir, "phase_diagram_final_state.png")
save(out, fig, px_per_unit = 2)
println("\nSaved: $out")
