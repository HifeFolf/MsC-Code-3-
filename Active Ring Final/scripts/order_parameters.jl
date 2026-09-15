

using CairoMakie, Statistics, Printf, TOML, JLD2

const PROJECT_ROOT_OP = joinpath(@__DIR__, "..")

# ── Run selection ─────────────────────────────────────────────────────────────
# ▼▼▼  Edit these two lines to choose a run when using the VS Code Run button  ▼▼▼
const DEFAULT_SUBFOLDER = "Results_N17_kinked2"
const DEFAULT_RUN_ID    = 200
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

run_dir = if length(ARGS) == 0
    joinpath(PROJECT_ROOT_OP, "results", DEFAULT_SUBFOLDER,
             @sprintf("run_%04d", DEFAULT_RUN_ID))
elseif length(ARGS) == 1
    abspath(ARGS[1])
else
    joinpath(PROJECT_ROOT_OP, "results", ARGS[1],
             @sprintf("run_%04d", parse(Int, ARGS[2])))
end

isdir(run_dir) || error("Run directory not found: $run_dir")
println("Loading: $run_dir")

# ── Load trajectory ───────────────────────────────────────────────────────────

traj_file   = joinpath(run_dir, "traj.jld2")
params_file = joinpath(run_dir, "params.toml")
isfile(traj_file)   || error("traj.jld2 not found")
isfile(params_file) || error("params.toml not found")

times, positions, bead_types = jldopen(traj_file, "r") do f
    f["times"], f["positions"], Symbol.(f["bead_types"])
end

ep        = TOML.parsefile(params_file)
N         = Int(ep["params"]["N"])
f0        = Float64(ep["params"]["f0"])
kappa     = Float64(ep["params"]["kappa"])
kBT       = Float64(ep["params"]["kBT"])
framerate = Int(ep["run"]["framerate"])
substeps  = Int(ep["run"]["substeps"])
dt        = Float64(ep["params"]["dt"])
L         = Float64(ep["run"]["L"])

n_frames  = size(positions, 3)
dt_frame  = substeps * dt
vtimes    = [(i - 1) / framerate for i in 1:n_frames]
half_L    = L / 2.0

println("  N=$N   f0=$f0   κ=$kappa   kBT=$kBT   frames=$n_frames")

# ── Per-bead velocities via central finite differences ────────────────────────
# Unwrap periodic jumps before differencing.

# First build unwrapped positions for each bead
pos_uw = copy(positions)
for t in 2:n_frames
    for i in 1:N
        for d in 1:2
            δ = positions[d, i, t] - positions[d, i, t-1]
            δ >  half_L && (δ -= L)
            δ < -half_L && (δ += L)
            pos_uw[d, i, t] = pos_uw[d, i, t-1] + δ
        end
    end
end

# Central differences (forward/backward at endpoints)
vx = zeros(N, n_frames)
vy = zeros(N, n_frames)
for t in 1:n_frames
    tp = min(t + 1, n_frames)
    tm = max(t - 1, 1)
    dt_cd = (tp - tm) * dt_frame
    for i in 1:N
        vx[i, t] = (pos_uw[1, i, tp] - pos_uw[1, i, tm]) / dt_cd
        vy[i, t] = (pos_uw[2, i, tp] - pos_uw[2, i, tm]) / dt_cd
    end
end

# ── Compute φ(t) and Γ(t) ────────────────────────────────────────────────────

phi_t   = Vector{Float64}(undef, n_frames)
gamma_t = Vector{Float64}(undef, n_frames)

for t in 1:n_frames
    # COM position at this frame
    cmx = mean(positions[1, :, t])
    cmy = mean(positions[2, :, t])

    sum_vhx = 0.0;  sum_vhy = 0.0   # for φ: Σ v̂ᵢ
    sum_cross = 0.0                  # for Γ: Σ r̄ᵢ,CM × v̂ᵢ

    for i in 1:N
        vn = sqrt(vx[i,t]^2 + vy[i,t]^2)
        vn < 1e-14 && continue       # bead not moving — skip (would be pure noise)

        vhx = vx[i,t] / vn
        vhy = vy[i,t] / vn

        # φ accumulator
        sum_vhx += vhx
        sum_vhy += vhy

        # Γ accumulator: 2-D cross product r̄ᵢ,CM × v̂ᵢ = rx*vhy − ry*vhx
        rx = positions[1, i, t] - cmx
        ry = positions[2, i, t] - cmy
        sum_cross += rx * vhy - ry * vhx
    end

    phi_t[t]   = sqrt(sum_vhx^2 + sum_vhy^2) / N
    gamma_t[t] = abs(sum_cross) / N
end

# ── PDF helper ────────────────────────────────────────────────────────────────

function hist_pdf(data::Vector{Float64}; n_bins::Int = 80)
    lo = 0.0;  hi = 1.0
    edges   = range(lo, hi; length = n_bins + 1)
    counts  = zeros(n_bins)
    for v in data
        i = clamp(searchsortedlast(edges, v), 1, n_bins)
        counts[i] += 1
    end
    centers = [(edges[i] + edges[i+1]) / 2 for i in 1:n_bins]
    width   = step(edges)
    pdf     = counts ./ (sum(counts) * width)
    return centers, pdf
end

# ── Figure ────────────────────────────────────────────────────────────────────

fig = Figure(size = (900, 420), backgroundcolor = :white)

title_str = @sprintf("f₀ = %.1f   κ = %.1f   kBT = %.3f   N = %d", f0, kappa, kBT, N)
Label(fig[0, 1:2], title_str; fontsize = 13, tellwidth = false)

# Left: time series
ax_t = Axis(fig[1, 1];
    xlabel  = "t  (video seconds)",
    ylabel  = "Order parameter",
    title   = "Time series",
    yticks  = 0:0.2:1.0,
)
ylims!(ax_t, -0.05, 1.05)
xlims!(ax_t, vtimes[1], vtimes[end])

lines!(ax_t, vtimes, phi_t;   color = :mediumpurple, linewidth = 1.2, label = "φ(t)")
lines!(ax_t, vtimes, gamma_t; color = :steelblue,    linewidth = 1.2, label = "Γ(t)")
axislegend(ax_t; position = :rb, labelsize = 12)

# Right: PDF (horizontal — density on x, value on y, matching paper layout)
ax_p = Axis(fig[1, 2];
    xlabel  = "PDF",
    ylabel  = "Order parameter",
    title   = "Probability density",
    yticks  = 0:0.2:1.0,
)
ylims!(ax_p, -0.05, 1.05)

phi_c,   phi_pdf   = hist_pdf(phi_t)
gamma_c, gamma_pdf = hist_pdf(gamma_t)

lines!(ax_p, phi_pdf,   phi_c;   color = :mediumpurple, linewidth = 1.5, label = "φ")
lines!(ax_p, gamma_pdf, gamma_c; color = :steelblue,    linewidth = 1.5, label = "Γ")
axislegend(ax_p; position = :rt, labelsize = 12)

colsize!(fig.layout, 1, Relative(0.65))
colsize!(fig.layout, 2, Relative(0.35))

out_path = joinpath(run_dir, "order_parameters.png")
save(out_path, fig, px_per_unit = 2)
println("\nSaved: $out_path")
