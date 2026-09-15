

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_PO = joinpath(@__DIR__, "..")
const LAST_SECONDS    = 20.0
const FRAMERATE       = 60
const LAST_FRAMES     = round(Int, LAST_SECONDS * FRAMERATE)   # 1200

subfolder = length(ARGS) >= 1 ? ARGS[1] : "Results_N17_kinked2"
id_start  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
id_end    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)

results_dir = joinpath(PROJECT_ROOT_PO, "results", subfolder)
isdir(results_dir) || error("Results folder not found: $results_dir")

# ── Scan runs ─────────────────────────────────────────────────────────────────

run_dirs = sort(filter(readdir(results_dir, join = false)) do d
    m = match(r"^run_(\d+)$", d)
    m === nothing && return false
    id = parse(Int, m[1])
    return id_start <= id <= id_end
end)

n_total = length(run_dirs)
println("Scanning $n_total runs from $subfolder (IDs $id_start – $id_end)…")
println("Using last $(round(Int, LAST_SECONDS))s ($LAST_FRAMES frames) per run.")
println()

κ_raw  = Float64[]
f0_raw = Float64[]
φ_raw  = Float64[]
Γ_raw  = Float64[]

n_loaded = Ref(0)

for (ri, rd) in enumerate(run_dirs)
    ri % 500 == 0 && @printf("  %d / %d\n", ri, n_total)

    local run_path, traj_file, params_file, ep, κ, f₀, N, sub, dt, L
    local dt_frame, half_L, positions, n_frames, i1, pos_ss, n_ss, pos_uw
    local vx, vy, φ_sum, Γ_sum, n_valid

    run_path    = joinpath(results_dir, rd)
    traj_file   = joinpath(run_path, "traj.jld2")
    params_file = joinpath(run_path, "params.toml")
    (isfile(traj_file) && isfile(params_file)) || continue

    ep  = TOML.parsefile(params_file)
    κ   = Float64(ep["params"]["kappa"])
    f₀  = Float64(ep["params"]["f0"])
    N   = Int(ep["params"]["N"])
    sub = Int(ep["run"]["substeps"])
    dt  = Float64(ep["params"]["dt"])
    L   = Float64(ep["run"]["L"])

    dt_frame = sub * dt
    half_L   = L / 2.0

    positions = jldopen(traj_file, "r") do f
        haskey(f, "positions") ? f["positions"] : nothing
    end
    positions === nothing && continue

    n_frames = size(positions, 3)
    i1       = max(1, n_frames - LAST_FRAMES + 1)
    pos_ss   = positions[:, :, i1:end]   # steady-state slice
    n_ss     = size(pos_ss, 3)
    n_ss < 2 && continue

    # Unwrap per-bead positions over the steady-state window
    pos_uw = copy(pos_ss)
    for t in 2:n_ss
        for i in 1:N, d in 1:2
            δ = pos_ss[d, i, t] - pos_ss[d, i, t-1]
            δ >  half_L && (δ -= L)
            δ < -half_L && (δ += L)
            pos_uw[d, i, t] = pos_uw[d, i, t-1] + δ
        end
    end

    # Per-bead velocities via central differences
    vx = zeros(N, n_ss)
    vy = zeros(N, n_ss)
    for t in 1:n_ss
        tp   = min(t + 1, n_ss)
        tm   = max(t - 1, 1)
        dt_c = (tp - tm) * dt_frame
        for i in 1:N
            vx[i, t] = (pos_uw[1, i, tp] - pos_uw[1, i, tm]) / dt_c
            vy[i, t] = (pos_uw[2, i, tp] - pos_uw[2, i, tm]) / dt_c
        end
    end

    # Compute φ(t) and Γ(t) then average
    φ_sum   = 0.0
    Γ_sum   = 0.0
    n_valid = 0

    for t in 1:n_ss
        cmx = mean(pos_ss[1, :, t])
        cmy = mean(pos_ss[2, :, t])

        sum_vhx   = 0.0;  sum_vhy = 0.0
        sum_cross = 0.0

        for i in 1:N
            vn = sqrt(vx[i,t]^2 + vy[i,t]^2)
            vn < 1e-14 && continue
            vhx = vx[i,t] / vn
            vhy = vy[i,t] / vn
            sum_vhx   += vhx
            sum_vhy   += vhy
            rx = pos_ss[1, i, t] - cmx
            ry = pos_ss[2, i, t] - cmy
            sum_cross += rx * vhy - ry * vhx
        end

        φ_sum   += sqrt(sum_vhx^2 + sum_vhy^2) / N
        Γ_sum   += abs(sum_cross) / N
        n_valid += 1
    end

    n_valid == 0 && continue

    push!(κ_raw,  κ)
    push!(f0_raw, f₀)
    push!(φ_raw,  φ_sum / n_valid)
    push!(Γ_raw,  Γ_sum / n_valid)
    n_loaded[] += 1
end

println("\nLoaded $(n_loaded[]) / $n_total runs.")
isempty(φ_raw) && error("No data loaded.")

# ── Build regular κ × f₀ grid ────────────────────────────────────────────────

kappa_vals = sort(unique(κ_raw))
f0_vals    = sort(unique(f0_raw))
nκ = length(kappa_vals)
nf = length(f0_vals)
@printf("Grid: %d κ values  ×  %d f₀ values\n", nκ, nf)

φ_lookup = Dict(zip(zip(κ_raw, f0_raw), φ_raw))
Γ_lookup = Dict(zip(zip(κ_raw, f0_raw), Γ_raw))

φ_matrix = fill(NaN, nκ, nf)
Γ_matrix = fill(NaN, nκ, nf)

for (ic, κ) in enumerate(kappa_vals), (ir, f₀) in enumerate(f0_vals)
    key = (κ, f₀)
    haskey(φ_lookup, key) && (φ_matrix[ic, ir] = φ_lookup[key])
    haskey(Γ_lookup, key) && (Γ_matrix[ic, ir] = Γ_lookup[key])
end

# ── Figure: two heatmaps side by side ────────────────────────────────────────

fig = Figure(size = (1400, 700))

Label(fig[0, 1:4],
      "Order parameters — $subfolder   (last $(round(Int,LAST_SECONDS))s,  N=16)";
      fontsize = 14, tellwidth = false)

function make_ax(col, title_str)
    Axis(fig[1, col];
        xlabel = "Active force  f₀",
        ylabel = "Bending stiffness  κ",
        title  = title_str,
        xticks = LinearTicks(7),
        yticks = LinearTicks(7),
    )
end

# φ heatmap
ax_φ = make_ax(1, "Translational order  ⟨φ⟩  (1 = collective translation)")
hm_φ = heatmap!(ax_φ, f0_vals, kappa_vals, φ_matrix';
                colormap   = :plasma,
                colorrange = (0.0, 1.0),
                nan_color  = :white)
Colorbar(fig[1, 2], hm_φ; label = "⟨φ⟩", width = 18)

# Γ heatmap
ax_Γ = make_ax(3, "Rotational order  ⟨Γ⟩  (1 = collective rotation)")
hm_Γ = heatmap!(ax_Γ, f0_vals, kappa_vals, Γ_matrix';
                colormap   = :viridis,
                colorrange = (0.0, 1.0),
                nan_color  = :white)
Colorbar(fig[1, 4], hm_Γ; label = "⟨Γ⟩", width = 18)

out_path = joinpath(results_dir, "phase_diagram_order.png")
save(out_path, fig, px_per_unit = 2)
println("Saved: $out_path")
