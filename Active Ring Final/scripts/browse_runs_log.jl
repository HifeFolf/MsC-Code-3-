# browse_runs_log.jl
# Interactive browser for a log-spaced κ × f₀ sweep (e.g. Results_log).
#
# The scatter is drawn in LOG10 space on LINEAR axes.  This gives even visual
# spacing across all decades and makes event.data return plain linear numbers,
# so click-to-point mapping is exact with a simple nearest-neighbour search.
# Tick labels are formatted to show the original f₀ / κ values.
#
# Points are coloured by their Kconn STATE — the same model as
# scripts/phase_diagram_kconn_states.jl, so the browser and that map agree:
#
#   • Settled circle→outward  — a GRADIENT in ⟨Kconn⟩−eq (pale circle → deep
#     outward).  This whole family is a smooth fade, no hard line.
#   • Inward-crumpled (orange) — settled inward, the junction jams.
#   • Bistable / looping (crimson) — will not settle; spends real time in BOTH
#     the inward and outward wells (e.g. flips +120°↔−135°).
#
# Click a crimson point and press R: you should see the ring loop in and out.
# A blue point just elongates outward and rotates — Kconn is SHAPE, not motion,
# so an outward-bent ring is a rigid rotator (moving) even at fixed Kconn.
#
# Click a point  →  print run details to terminal + info bar
# Press R        →  render preview.mp4 for selected run
# Press A        →  run analyze_motion.jl for selected run
#
# Usage (from the Active Ring Final/ project root):
#
#   julia --project scripts/browse_runs_log.jl
#   julia --project scripts/browse_runs_log.jl Results_log
#   julia --project scripts/browse_runs_log.jl Results_log 1 500

using GLMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_BRL = joinpath(@__DIR__, "..")
# —— identical to scripts/phase_diagram_kconn_states.jl so labels match ————————————
const SKIP_SECONDS = 0.5     # drop brief IC-relaxation transient (sim-time)
const IN_ABS_DEG   = 5.0     # Kconn below this      ⇒ sample is "well inward"
const OUT_OFF_DEG  = 20.0    # Kconn above eq+this   ⇒ sample is "well outward"
const BI_FRAC      = 0.15    # ≥ this fraction in EACH well ⇒ Bistable/looping
const INWARD_DEV   = -8.0    # ⟨Kconn⟩−eq below this  ⇒ settled Inward-crumpled

# Per-run state from Kconn(t) (deg).  1 = Bistable/looping  2 = Inward-crumpled
# 3 = Settled (circle→outward fade).   eq = 360/N.
function kconn_state(t::AbstractVector, Kdeg::AbstractVector, eq::Real)
    i0    = something(findfirst(≥(SKIP_SECONDS), t), 1)
    seg   = @view Kdeg[i0:end]
    f_in  = count(<(IN_ABS_DEG),      seg) / length(seg)
    f_out = count(>(eq + OUT_OFF_DEG), seg) / length(seg)
    dev   = mean(seg) - eq
    code  = min(f_in, f_out) > BI_FRAC ? 1 : (dev < INWARD_DEV ? 2 : 3)
    return (code = code, dev = dev, f_in = f_in, f_out = f_out, Kmean = mean(seg))
end

const HARD_COLORS  = Dict(1 => Makie.to_color(:crimson), 2 => Makie.to_color(:darkorange))
const MISSING_COL  = Makie.to_color(:gray70)
const GRAD_CMAP    = :Blues
const STATE_LABELS = ["Bistable / looping", "Inward-crumpled", "Settled (circle→outward)"]

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_circle_indep"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_BRL, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Load all runs ─────────────────────────────────────────────────────────────

println("Loading runs from $subfolder (IDs $id_start – $id_end)…")
    
run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

κ_vals     = Float64[]
f0_vals    = Float64[]
Kdev_vals  = Float64[]   # ⟨Kconn⟩ − eq (deg)  — gradient axis for settled runs
Kmean_vals = Float64[]   # ⟨Kconn⟩ (deg)
Kfin_vals  = Float64[]   # fraction of post-transient time well inward
Kfout_vals = Float64[]   # fraction of post-transient time well outward
code_vals  = Int[]       # 1 bistable, 2 inward, 3 settled  (0 = no Kconn / old run)
run_ids    = String[]
run_paths  = String[]

for rd in run_dirs
    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])
    eq = 360.0 / Int(ep["N"])

    st = jldopen(obs_file, "r") do f
        haskey(f, "Kconn") || return (code=0, dev=NaN, f_in=NaN, f_out=NaN, Kmean=NaN)
        m = length(f["Kconn"])
        m < 10 && return (code=0, dev=NaN, f_in=NaN, f_out=NaN, Kmean=NaN)
        t = haskey(f, "times") ? f["times"] : collect(1:m)   # fallback: frame index
        kconn_state(t, rad2deg.(f["Kconn"]), eq)
    end

    push!(κ_vals,     κ)
    push!(f0_vals,    f₀)
    push!(Kdev_vals,  st.dev)
    push!(Kmean_vals, st.Kmean)
    push!(Kfin_vals,  st.f_in)
    push!(Kfout_vals, st.f_out)
    push!(code_vals,  st.code)
    push!(run_ids,    rd)
    push!(run_paths,  run_path)
end

n_loaded = length(run_ids)
println("Loaded $n_loaded runs.")
n_loaded == 0 && error("No valid runs found in $results_dir")
for c in 1:3
    println("  $(STATE_LABELS[c]): $(count(==(c), code_vals))")
end
count(==(0), code_vals) > 0 && println("  (no Kconn / old run): $(count(==(0), code_vals))")
println()

# ── Pre-compute log10 plot coordinates ───────────────────────────────────────
# Everything is drawn in log10 space on linear axes.
# event.data from register_interaction! then returns log10 values directly.

lf0 = log10.(f0_vals)
lκ  = log10.(κ_vals)

lf0_range = let r = maximum(lf0) - minimum(lf0); r > 0 ? r : 1.0 end
lκ_range  = let r = maximum(lκ)  - minimum(lκ);  r > 0 ? r : 1.0 end

# ── Build tick positions + labels from the unique grid values ─────────────────
# Pick up to ~8 evenly-spaced ticks across the log range.

function make_log_ticks(vals::Vector{Float64}; max_ticks::Int = 8)
    uvals  = sort(unique(vals))
    n      = length(uvals)
    step   = max(1, round(Int, n / max_ticks))
    chosen = uvals[1:step:n]
    # Always include the last value
    chosen[end] != uvals[end] && push!(chosen, uvals[end])
    positions = log10.(chosen)
    labels    = [v >= 10 ? @sprintf("%.4g", v) : @sprintf("%.3g", v) for v in chosen]
    return positions, labels
end

f0_tick_pos, f0_tick_lbl = make_log_ticks(f0_vals)
κ_tick_pos,  κ_tick_lbl  = make_log_ticks(κ_vals)

# ── Build figure ──────────────────────────────────────────────────────────────

fig = Figure(size = (1150, 820))

ax = Axis(fig[1, 1];
    xlabel = "Active force  f₀",
    ylabel = "Bending stiffness  κ",
    title  = "$subfolder — click a point to inspect   [R] render   [A] analyse",
    xticks = (f0_tick_pos, f0_tick_lbl),
    yticks = (κ_tick_pos,  κ_tick_lbl),
)

# Colour: settled runs = circle→outward gradient in dev; hard states = solid.
settled_dev = [Kdev_vals[i] for i in eachindex(code_vals)
               if code_vals[i] == 3 && isfinite(Kdev_vals[i])]
dev_hi = isempty(settled_dev) ? 1.0 : max(quantile(settled_dev, 0.97), 1e-6)
grad = cgrad(GRAD_CMAP)
point_color(code, dev) =
    code == 0 ? MISSING_COL :
    code == 3 ? Makie.to_color(grad[clamp(max(dev, 0.0) / dev_hi, 0.0, 1.0)]) :
                HARD_COLORS[code]
point_colors = [point_color(code_vals[i], Kdev_vals[i]) for i in eachindex(code_vals)]

scatter!(ax, lf0, lκ;
         color       = point_colors,
         markersize  = 9,
         strokewidth = 0.3,
         strokecolor = :black)

Colorbar(fig[1, 2]; colormap = GRAD_CMAP, limits = (0.0, dev_hi),
         label = "outward bend  ⟨Kconn⟩−eq [deg]  (settled)", width = 15)
Legend(fig[1, 3],
       [MarkerElement(color = HARD_COLORS[c], marker = :circle, markersize = 12,
                      strokewidth = 0.3, strokecolor = :black) for c in 1:2],
       ["Bistable / looping", "Inward-crumpled"], "Hard states"; framevisible = true)

# Info panel
info_ax = Axis(fig[2, 1:3]; height = 90)
hidedecorations!(info_ax); hidespines!(info_ax)
xlims!(info_ax, 0, 1); ylims!(info_ax, 0, 1)

info_text = Observable("Click a point to see run details.")
text!(info_ax, 0.01, 0.72; text = info_text, fontsize = 12,
      align = (:left, :center), color = :black)

rowsize!(fig.layout, 2, Fixed(90))

# Highlight selected point — a hollow ring (so it reads over any phase colour)
sel_lx = Observable(Float64[])
sel_ly = Observable(Float64[])
scatter!(ax, sel_lx, sel_ly;
         color = :transparent, marker = :circle, markersize = 22,
         strokewidth = 3, strokecolor = :black)
scatter!(ax, sel_lx, sel_ly;
         color = :transparent, marker = :circle, markersize = 26,
         strokewidth = 1.5, strokecolor = :white)

# ── State ─────────────────────────────────────────────────────────────────────

selected_idx = Ref(0)

function find_nearest(fx, fy)
    # Axes are linear in log10 space, so event.data is already log10(f0), log10(κ).
    # Plain normalised Euclidean distance in log10 space.
    best_i = 0
    best_d = Inf
    for i in eachindex(lf0)
        d = ((lf0[i] - fx) / lf0_range)^2 +
            ((lκ[i]  - fy) / lκ_range)^2
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return best_i
end

function select_run(i)
    selected_idx[] = i
    sel_lx[] = [lf0[i]]
    sel_ly[] = [lκ[i]]

    ep  = TOML.parsefile(joinpath(run_paths[i], "params.toml"))["params"]
    κv  = ep["kappa"]
    f0v = ep["f0"]
    kBT = ep["kBT"]
    Φ   = κv > 0 ? f0v / κv : NaN

    st_str   = code_vals[i] == 0 ? "N/A (no Kconn)" : STATE_LABELS[code_vals[i]]
    dev_str  = isnan(Kdev_vals[i])  ? "N/A" : @sprintf("%+.1f°", Kdev_vals[i])
    km_str   = isnan(Kmean_vals[i]) ? "N/A" : @sprintf("%.1f°", Kmean_vals[i])
    fin_str  = isnan(Kfin_vals[i])  ? "N/A" : @sprintf("%.2f", Kfin_vals[i])
    fout_str = isnan(Kfout_vals[i]) ? "N/A" : @sprintf("%.2f", Kfout_vals[i])

    line1 = @sprintf(
        "Run: %s   f₀ = %.4g   κ = %.4g   Φ = f₀/κ = %.4g   kBT = %.3g",
        run_ids[i], f0v, κv, Φ, kBT)
    line2 = @sprintf(
        "STATE: %s    ⟨Kconn⟩ = %s  (dev %s)   time inward = %s   time outward = %s   [R] render   [A] analyse",
        st_str, km_str, dev_str, fin_str, fout_str)

    info_text[] = line1 * "\n" * line2

    println("\n── Selected ────────────────────────────────────────────────")
    println("  Path  : $(run_paths[i])")
    println("  ", line1)
    println("  ", line2)
end

# ── Mouse click handler ───────────────────────────────────────────────────────

register_interaction!(ax, :run_selector) do event::MouseEvent, _axis
    event.type == MouseEventTypes.leftclick || return
    idx = find_nearest(event.data[1], event.data[2])
    idx > 0 && select_run(idx)
end

# ── Keyboard handler ──────────────────────────────────────────────────────────

on(events(fig).keyboardbutton) do event
    event.action == Keyboard.press || return
    selected_idx[] == 0 && return

    i        = selected_idx[]
    run_path = run_paths[i]

    if event.key == Keyboard.r
        println("\n[R] Rendering movie for $(run_ids[i])…")
        render_script = joinpath(@__DIR__, "render_movie.jl")
        run(`julia --project=$(PROJECT_ROOT_BRL) $render_script $run_path`)

    elseif event.key == Keyboard.a
        println("\n[A] Running motion analysis for $(run_ids[i])…")
        analyse_script = joinpath(@__DIR__, "analyze_motion.jl")
        run(`julia --project=$(PROJECT_ROOT_BRL) $analyse_script $run_path`)
    end
end

# ── Launch ────────────────────────────────────────────────────────────────────

println("Browser ready  ($n_loaded points, log10 coordinates on linear axes).")
println("  Click any point  →  inspect run (terminal + info bar)")
println("  Press R          →  render preview.mp4 for selected run")
println("  Press A          →  run analyze_motion.jl for selected run")
println()

display(fig)
wait(fig.scene)
