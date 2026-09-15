# hysteresis_movie.jl
# Run the hysteresis protocol and render an MP4.
#
# Left panel  : ring inside the simulation box.
# Right panel : b vs f₀ hysteresis curve traced in real time
#               (blue = up-ramp, red = down-ramp).
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/hysteresis_movie.jl
#   julia --project scripts/hysteresis_movie.jl configs/default.toml
#   julia --project scripts/hysteresis_movie.jl configs/default.toml kappa f0_max f0_min n_steps kBT
#
# Output:
#   results/hysteresis/hysteresis_movie_N{N}_kappa{k}_kBT{t}.mp4

using TOML, Random, Printf, Statistics, GLMakie

const PROJECT_ROOT_HM = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_HM, "src", "EngineRing.jl"));        using .EngineRing
include(joinpath(PROJECT_ROOT_HM, "src", "InitialConditions.jl")); using .InitialConditions
include(joinpath(PROJECT_ROOT_HM, "src", "Observables.jl"));       using .Observables

# ── Arguments ─────────────────────────────────────────────────────────────────
config_path = length(ARGS) >= 1 ? ARGS[1] :
              joinpath(PROJECT_ROOT_HM, "configs", "default.toml")
cfg = TOML.parsefile(config_path)
ep  = cfg["params"]
rc  = cfg["run"]

kappa_val = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
f0_max    = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 200.0
f0_min    = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
n_steps   = length(ARGS) >= 5 ? parse(Int,     ARGS[5]) : 25
kBT_val   = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : ep["kBT"]

frames_equil  = 400
frames_record = 200
frames_saved  = 20
movie_fps     = 30
substeps      = rc["substeps"]
seed          = rc["seed"]
N             = Int(ep["N"])
L             = Float64(rc["L"])
hw            = L / 2.0

println("=" ^ 60)
@printf("Hysteresis movie  |  N=%d  κ=%.2f  kBT=%.3f\n", N, kappa_val, kBT_val)
@printf("f₀: %.2f → %.2f → %.2f  (%d steps)\n", f0_min, f0_max, f0_min, n_steps)
println("=" ^ 60)

# ── Params ────────────────────────────────────────────────────────────────────
function make_params(f0_val)
    EngineRing.Params(
        N=N, l=ep["l"], ks=ep["ks"], κ=kappa_val,
        theta_max=ep["theta_max"], theta_buf=ep["theta_buf"],
        k_stop=ep["k_stop"], sigma=ep["sigma"], epsilon=ep["epsilon"],
        topology=Symbol(ep["topology"]),
        f0=f0_val, mu=ep["mu"], kBT=kBT_val, dt=ep["dt"],
    )
end

# ── Run one step; return (mean_b, saved_positions) ────────────────────────────
function run_segment!(state, f0_val, n_equil, n_record, n_save)
    p = make_params(f0_val)
    for _ in 1:n_equil
        for _ in 1:substeps; EngineRing.step!(state, p); end
    end
    save_every = max(1, n_record ÷ n_save)
    saved = Matrix{Float64}[]
    b_acc = 0.0
    for fr in 1:n_record
        for _ in 1:substeps; EngineRing.step!(state, p); end
        b_acc += Observables.compute_frame_obs(state.r).b
        if fr % save_every == 0 && length(saved) < n_save
            push!(saved, copy(state.r))
        end
    end
    while length(saved) < n_save; push!(saved, copy(state.r)); end
    return b_acc / n_record, saved
end

# ── Periodic wrap for display ─────────────────────────────────────────────────
function wrap_display(r)
    rw = copy(r)
    for i in 1:size(r,2)
        rw[1,i] = mod(r[1,i] + hw, L) - hw
        rw[2,i] = mod(r[2,i] + hw, L) - hw
    end
    return rw
end

# ── f₀ ramps ──────────────────────────────────────────────────────────────────
f0_up   = exp.(range(log(f0_min), log(f0_max), length = n_steps))
f0_down = reverse(f0_up)

Random.seed!(seed)
bead_types = InitialConditions.antipolar_bead_types(N)
p0 = make_params(f0_min)

# Up ramp
println("\nUp ramp ↑")
state_up = InitialConditions.create_ic("antipolar", p0, bead_types)
b_up   = Vector{Float64}(undef, n_steps)
pos_up = Vector{Vector{Matrix{Float64}}}(undef, n_steps)
for (i, f0) in enumerate(f0_up)
    b_up[i], pos_up[i] = run_segment!(state_up, f0, frames_equil, frames_record, frames_saved)
    @printf("  ↑ %2d/%d  f₀=%8.2f  b=%.4f\n", i, n_steps, f0, b_up[i])
end

# Down ramp
println("\nDown ramp ↓")
state_dn = deepcopy(state_up)
b_dn   = Vector{Float64}(undef, n_steps)
pos_dn = Vector{Vector{Matrix{Float64}}}(undef, n_steps)
for (i, f0) in enumerate(f0_down)
    b_dn[i], pos_dn[i] = run_segment!(state_dn, f0, frames_equil, frames_record, frames_saved)
    @printf("  ↓ %2d/%d  f₀=%8.2f  b=%.4f\n", i, n_steps, f0, b_dn[i])
end

b_dn_ordered = reverse(b_dn)

# Flatten frame list: (r, f0, b_inst_mean, direction)
frame_data = Tuple{Matrix{Float64}, Float64, Float64, Symbol}[]
for (i, f0) in enumerate(f0_up)
    for pos in pos_up[i]; push!(frame_data, (pos, f0, b_up[i], :up)); end
end
for (i, f0) in enumerate(f0_down)
    for pos in pos_dn[i]; push!(frame_data, (pos, f0, b_dn[i], :dn)); end
end
total_frames = length(frame_data)
@printf("\nTotal frames: %d  (%.1f s at %d fps)\n", total_frames, total_frames/movie_fps, movie_fps)

# ── Ring backbone (closed loop, no wrapping across box) ───────────────────────
const COLOUR_MAP = Dict(:active => :dodgerblue, :active_rev => :crimson, :passive => :gold)
bead_colours = [COLOUR_MAP[t] for t in bead_types]

function backbone_pts(r)
    pts = Point2f[]
    for i in 1:N
        j = i == N ? 1 : i+1
        push!(pts, Point2f(r[1,i], r[2,i]))
        push!(pts, Point2f(r[1,j], r[2,j]))
    end
    return pts
end

# ── Build figure ──────────────────────────────────────────────────────────────
r0 = wrap_display(frame_data[1][1])

backbone_obs = Observable(backbone_pts(r0))
xs_obs       = Observable(r0[1,:])
ys_obs       = Observable(r0[2,:])
title_obs    = Observable("")
trace_up     = Observable(Point2f[])
trace_dn     = Observable(Point2f[])

fig = Figure(size = (1200, 560))

Label(fig[0, 1:2], title_obs; fontsize = 13, tellwidth = false)

# Left: ring in box
ax_ring = Axis(fig[1, 1];
               xlabel = "x", ylabel = "y",
               aspect = DataAspect())
xlims!(ax_ring, -hw, hw)
ylims!(ax_ring, -hw, hw)

# Box border
box_xs = [-hw, hw, hw, -hw, -hw]
box_ys = [-hw, -hw, hw, hw, -hw]
lines!(ax_ring, box_xs, box_ys; color = :black, linewidth = 1.5)

linesegments!(ax_ring, backbone_obs; linewidth = 2, color = :gray30)
scatter!(ax_ring, xs_obs, ys_obs; markersize = 10, color = bead_colours)

# Right: hysteresis curve being traced
ax_b = Axis(fig[1, 2];
            xlabel  = "Active force  f₀",
            ylabel  = "Shape parameter  ⟨b⟩",
            title   = "κ = $kappa_val   kBT = $kBT_val",
            xscale  = log10,
            limits  = (nothing, (-0.02, 1.02)))

lines!(ax_b,   trace_up; color = :royalblue, linewidth = 2.5, label = "↑ up ramp")
scatter!(ax_b, trace_up; color = :royalblue, markersize = 7)
lines!(ax_b,   trace_dn; color = :crimson,   linewidth = 2.5, label = "↓ down ramp")
scatter!(ax_b, trace_dn; color = :crimson,   markersize = 7)
axislegend(ax_b; position = :lt)

display(fig)

# ── Build a step→b lookup for each ramp ──────────────────────────────────────
# We add a new point to the trace whenever we enter a new f₀ step.
up_points_added = Set{Int}()
dn_points_added = Set{Int}()

function step_index_up(f0)
    clamp(searchsortedfirst(f0_up, f0; lt = (a,b) -> a < b*0.9999), 1, n_steps)
end
function step_index_dn(f0)
    something(findfirst(x -> isapprox(x, f0; rtol=1e-5), f0_down), 1)
end

# ── Render ────────────────────────────────────────────────────────────────────
out_dir  = joinpath(PROJECT_ROOT_HM, "results", "hysteresis")
mkpath(out_dir)
tag      = @sprintf("N%d_kappa%.2f_kBT%.3f", N, kappa_val, kBT_val)
out_path = joinpath(out_dir, "hysteresis_movie_$(tag).mp4")

println("Rendering …")
t0 = time()

record(fig, out_path, 1:total_frames; framerate = movie_fps) do fi
    r, f0, b_step, dir = frame_data[fi]
    rw = wrap_display(r)

    backbone_obs[] = backbone_pts(rw)
    xs_obs[]       = rw[1,:]
    ys_obs[]       = rw[2,:]

    b_inst = Observables.compute_frame_obs(r).b

    if dir == :up
        title_obs[] = @sprintf("↑ UP RAMP   f₀ = %.2f   Fl = %.2f   b = %.4f",
                               f0, f0/kappa_val, b_inst)
        si = step_index_up(f0)
        if si ∉ up_points_added
            push!(trace_up[], Point2f(f0, b_step))
            notify(trace_up)
            push!(up_points_added, si)
        end
    else
        title_obs[] = @sprintf("↓ DOWN RAMP   f₀ = %.2f   Fl = %.2f   b = %.4f",
                               f0, f0/kappa_val, b_inst)
        si = step_index_dn(f0)
        if si ∉ dn_points_added
            push!(trace_dn[], Point2f(f0, b_step))
            notify(trace_dn)
            push!(dn_points_added, si)
        end
    end
end

@printf("Done in %.1f s\nSaved: %s\n", time()-t0, out_path)
