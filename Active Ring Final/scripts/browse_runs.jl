

using GLMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT  = joinpath(@__DIR__, "..")
const SKIP_SECONDS  = 5.0
const FRAMERATE     = 60
const SKIP_FRAMES   = round(Int, SKIP_SECONDS * FRAMERATE)   # 300

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_05"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Load all runs ─────────────────────────────────────────────────────────────

println("Loading runs from $subfolder (IDs $id_start – $id_end)…")

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

κ_vals   = Float64[]
f0_vals  = Float64[]
b_vals   = Float64[]
run_ids  = String[]
run_paths = String[]

for rd in run_dirs
    run_path    = joinpath(results_dir, rd)
    obs_file    = joinpath(run_path, "observables.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(obs_file) && isfile(params_file)) || continue

    ep = TOML.parsefile(params_file)["params"]
    κ  = Float64(ep["kappa"])
    f₀ = Float64(ep["f0"])

    b_vec = jldopen(obs_file, "r") do f
        haskey(f, "b") ? f["b"] : nothing
    end
    b_vec === nothing && continue
    n = length(b_vec)
    (SKIP_FRAMES + 1 > n) && continue

    b_ss = mean(b_vec[(SKIP_FRAMES + 1):n])

    push!(κ_vals,    κ)
    push!(f0_vals,   f₀)
    push!(b_vals,    b_ss)
    push!(run_ids,   rd)
    push!(run_paths, run_path)
end

println("Loaded $(length(run_ids)) runs.\n")

# ── Build figure ──────────────────────────────────────────────────────────────

fig = Figure(size = (1100, 750))

ax = Axis(fig[1, 1];
    xlabel = "Active force  f₀",
    ylabel = "Bending stiffness  κ",
    title  = "$subfolder — click a point to inspect   [R] render movie   [A] analyze motion",
)

crange = (0.0, min(1.0, maximum(b_vals)))
sc = scatter!(ax, f0_vals, κ_vals;
              color       = b_vals,
              colormap    = :viridis,
              colorrange  = crange,
              markersize  = 8,
              strokewidth = 0.2,
              strokecolor = :black)

Colorbar(fig[1, 2], sc; label = "mean asphericity  b", width = 16)

# Info panel below
info_ax = Axis(fig[2, 1:2]; height = 80)
hidedecorations!(info_ax); hidespines!(info_ax)
xlims!(info_ax, 0, 1); ylims!(info_ax, 0, 1)

info_text = Observable("Click a point to see run details.")
text!(info_ax, 0.01, 0.65; text = info_text, fontsize = 13,
      align = (:left, :center), color = :black)

rowsize!(fig.layout, 2, Fixed(80))

# Highlight selected point — use explicit Observables so updates work correctly
sel_x = Observable(Float64[])
sel_y = Observable(Float64[])
scatter!(ax, sel_x, sel_y;
         color = :red, markersize = 16,
         strokewidth = 2, strokecolor = :white)

# ── State ─────────────────────────────────────────────────────────────────────

selected_idx = Ref(0)   # 0 = nothing selected

function find_nearest(fx, fy)
    # Normalise so both axes have equal weight regardless of their ranges
    f0_scale = maximum(f0_vals) - minimum(f0_vals)
    k_scale  = maximum(κ_vals)  - minimum(κ_vals)
    f0_scale = f0_scale > 0 ? f0_scale : 1.0
    k_scale  = k_scale  > 0 ? k_scale  : 1.0

    best_i = 0
    best_d = Inf
    for i in eachindex(f0_vals)
        d = ((f0_vals[i] - fx) / f0_scale)^2 + ((κ_vals[i] - fy) / k_scale)^2
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return best_i   # always return nearest — no threshold needed
end

function select_run(i)
    selected_idx[] = i
    sel_x[] = [f0_vals[i]]
    sel_y[] = [κ_vals[i]]

    ep  = TOML.parsefile(joinpath(run_paths[i], "params.toml"))["params"]
    κv  = ep["kappa"];  f0v = ep["f0"];  kBT = ep["kBT"]
    χ   = κv / 15
    Φ   = (f0v > 0 && κv > 0) ? f0v * 15 / κv : NaN

    line1 = @sprintf("Run: %s   f₀ = %.3g   κ = %.3g   kBT = %.3g   b = %.4f",
                     run_ids[i], f0v, κv, kBT, b_vals[i])
    line2 = isnan(Φ) ? @sprintf("χ = %.4f   Φ = N/A  (f₀ or κ = 0)   [R] render   [A] analyse", χ) :
                       @sprintf("χ = %.4f   Φ = %.4f   [R] render movie   [A] analyse motion", χ, Φ)

    info_text[] = line1 * "\n" * line2

    println("\n── Selected ────────────────────────────────────────────")
    println("  Path : $(run_paths[i])")
    println("  ", line1)
    println("  ", line2)
end

# ── Mouse click handler — use register_interaction! (correct GLMakie API) ────

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
        run(`julia --project=$(PROJECT_ROOT) $render_script $run_path`)

    elseif event.key == Keyboard.a
        println("\n[A] Running motion analysis for $(run_ids[i])…")
        analyse_script = joinpath(@__DIR__, "analyze_motion.jl")
        run(`julia --project=$(PROJECT_ROOT) $analyse_script $run_path`)
    end
end

# ── Launch ────────────────────────────────────────────────────────────────────

println("Browser ready.")
println("  Click any point  →  inspect run (printed to terminal + info bar)")
println("  Press R          →  render preview.mp4 for selected run")
println("  Press A          →  run analyze_motion.jl for selected run")
println()

display(fig)
wait(fig.scene)   # keep alive until window is closed
