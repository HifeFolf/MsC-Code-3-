# render_hysteresis_movies.jl
# Render an MP4 movie for every hysteresis JLD2 file found in
# results/hysteresis/.  Loads once, loops through all runs — no
# recompilation overhead between movies.
#
# Left panel  : ring inside the simulation box.
# Right panel : b vs f₀ hysteresis curve traced in real time
#               (blue = up-ramp, red = down-ramp).
#
# Usage (from Active Ring Final/ project root):
#   julia --project scripts/render_hysteresis_movies.jl

using TOML, Random, Printf, Statistics, JLD2, GLMakie

const PROJECT_ROOT_RHM = joinpath(@__DIR__, "..")

include(joinpath(PROJECT_ROOT_RHM, "src", "EngineRing.jl"));        using .EngineRing
include(joinpath(PROJECT_ROOT_RHM, "src", "InitialConditions.jl")); using .InitialConditions
include(joinpath(PROJECT_ROOT_RHM, "src", "Observables.jl"));       using .Observables

const EP           = TOML.parsefile(joinpath(PROJECT_ROOT_RHM, "configs", "default.toml"))["params"]
const RC           = TOML.parsefile(joinpath(PROJECT_ROOT_RHM, "configs", "default.toml"))["run"]
const SUBSTEPS     = RC["substeps"]
const SEED         = RC["seed"]
const L_BOX        = Float64(RC["L"])
const HW           = L_BOX / 2.0
const FRAMES_EQUIL  = 400
const FRAMES_RECORD = 200
const FRAMES_SAVED  = 20
const MOVIE_FPS     = 30

const COLOUR_MAP = Dict(:active => :dodgerblue, :active_rev => :crimson, :passive => :gold)

# ── Display helpers ───────────────────────────────────────────────────────────
function wrap_display(r)
    rw = copy(r)
    for i in axes(r, 2)
        rw[1,i] = mod(r[1,i] + HW, L_BOX) - HW
        rw[2,i] = mod(r[2,i] + HW, L_BOX) - HW
    end
    return rw
end

function ring_backbone(r, N)
    pts = Point2f[]
    for i in 1:N
        j = i == N ? 1 : i+1
        push!(pts, Point2f(r[1,i], r[2,i]))
        push!(pts, Point2f(r[1,j], r[2,j]))
    end
    return pts
end

# ── Core simulation ───────────────────────────────────────────────────────────
function make_params(kappa_val, f0_val, kBT_val, N)
    EngineRing.Params(
        N=N, l=EP["l"], ks=EP["ks"], κ=kappa_val,
        theta_max=EP["theta_max"], theta_buf=EP["theta_buf"],
        k_stop=EP["k_stop"], sigma=EP["sigma"], epsilon=EP["epsilon"],
        topology=Symbol(EP["topology"]),
        f0=f0_val, mu=EP["mu"], kBT=kBT_val, dt=EP["dt"],
    )
end

function run_segment!(state, kappa_val, kBT_val, N, f0_val,
                      n_equil, n_record, n_save)
    p = make_params(kappa_val, f0_val, kBT_val, N)
    for _ in 1:n_equil
        for _ in 1:SUBSTEPS; EngineRing.step!(state, p); end
    end
    save_every = max(1, n_record ÷ n_save)
    saved = Matrix{Float64}[]
    b_acc = 0.0
    for fr in 1:n_record
        for _ in 1:SUBSTEPS; EngineRing.step!(state, p); end
        b_acc += Observables.compute_frame_obs(state.r).b
        if fr % save_every == 0 && length(saved) < n_save
            push!(saved, copy(state.r))
        end
    end
    while length(saved) < n_save; push!(saved, copy(state.r)); end
    return b_acc / n_record, saved
end

# ── Main render function ──────────────────────────────────────────────────────
function render_hysteresis_movie(kappa_val, kBT_val, f0_min, f0_max, n_steps, N,
                                  out_dir)
    tag      = @sprintf("N%d_kappa%.2f_kBT%.3f", N, kappa_val, kBT_val)
    out_path = joinpath(out_dir, "hysteresis_movie_$(tag).mp4")

    if isfile(out_path)
        println("  Skipping (movie exists): $tag")
        return
    end

    @printf("\n=== %s ===\n", tag)
    @printf("  κ=%.2f  kBT=%.3f  f₀: %.2f→%.2f  %d steps\n",
            kappa_val, kBT_val, f0_min, f0_max, n_steps)

    f0_up   = exp.(range(log(f0_min), log(f0_max), length = n_steps))
    f0_down = reverse(f0_up)

    Random.seed!(SEED)
    bead_types   = InitialConditions.antipolar_bead_types(N)
    bead_colours = [COLOUR_MAP[t] for t in bead_types]
    p0 = make_params(kappa_val, f0_min, kBT_val, N)

    # Up ramp
    println("  Up ramp ↑ …")
    state_up = InitialConditions.create_ic("antipolar", p0, bead_types)
    b_up   = Vector{Float64}(undef, n_steps)
    pos_up = Vector{Vector{Matrix{Float64}}}(undef, n_steps)
    for (i, f0) in enumerate(f0_up)
        b_up[i], pos_up[i] = run_segment!(state_up, kappa_val, kBT_val, N,
                                           f0, FRAMES_EQUIL, FRAMES_RECORD, FRAMES_SAVED)
        @printf("    ↑ %2d/%d  f₀=%8.2f  b=%.4f\n", i, n_steps, f0, b_up[i])
    end

    # Down ramp
    println("  Down ramp ↓ …")
    state_dn = deepcopy(state_up)
    b_dn   = Vector{Float64}(undef, n_steps)
    pos_dn = Vector{Vector{Matrix{Float64}}}(undef, n_steps)
    for (i, f0) in enumerate(f0_down)
        b_dn[i], pos_dn[i] = run_segment!(state_dn, kappa_val, kBT_val, N,
                                           f0, FRAMES_EQUIL, FRAMES_RECORD, FRAMES_SAVED)
        @printf("    ↓ %2d/%d  f₀=%8.2f  b=%.4f\n", i, n_steps, f0, b_dn[i])
    end

    # Flatten into frame sequence: (positions, f0, mean_b, direction)
    frame_data = Tuple{Matrix{Float64}, Float64, Float64, Symbol}[]
    for (i, f0) in enumerate(f0_up)
        for pos in pos_up[i]; push!(frame_data, (pos, f0, b_up[i], :up)); end
    end
    for (i, f0) in enumerate(f0_down)
        for pos in pos_dn[i]; push!(frame_data, (pos, f0, b_dn[i], :dn)); end
    end

    total_frames = length(frame_data)
    @printf("  Rendering %d frames (%.1f s) …\n", total_frames, total_frames/MOVIE_FPS)

    r0 = wrap_display(frame_data[1][1])
    backbone_obs = Observable(ring_backbone(r0, N))
    xs_obs       = Observable(r0[1,:])
    ys_obs       = Observable(r0[2,:])
    title_obs    = Observable("")
    trace_up     = Observable(Point2f[])
    trace_dn     = Observable(Point2f[])

    fig = Figure(size = (1200, 560))
    Label(fig[0, 1:2], title_obs; fontsize = 13, tellwidth = false)

    ax_ring = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect())
    xlims!(ax_ring, -HW, HW)
    ylims!(ax_ring, -HW, HW)

    box_xs = [-HW, HW, HW, -HW, -HW]
    box_ys = [-HW, -HW, HW, HW, -HW]
    lines!(ax_ring, box_xs, box_ys; color = :black, linewidth = 1.5)
    linesegments!(ax_ring, backbone_obs; linewidth = 2, color = :gray30)
    scatter!(ax_ring, xs_obs, ys_obs; markersize = 10, color = bead_colours)

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

    up_points_added = Set{Int}()
    dn_points_added = Set{Int}()

    function step_index_up(f0)
        clamp(searchsortedfirst(f0_up, f0; lt = (a,b) -> a < b*0.9999), 1, n_steps)
    end
    function step_index_dn(f0)
        something(findfirst(x -> isapprox(x, f0; rtol=1e-5), f0_down), 1)
    end

    t0 = time()
    record(fig, out_path, 1:total_frames; framerate = MOVIE_FPS) do fi
        r, f0, b_step, dir = frame_data[fi]
        rw = wrap_display(r)

        backbone_obs[] = ring_backbone(rw, N)
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
    @printf("  Done in %.1f s → %s\n", time()-t0, out_path)
end

# ── Discover and render all runs ──────────────────────────────────────────────
out_dir = joinpath(PROJECT_ROOT_RHM, "results", "hysteresis")

jld_files = sort(filter(f -> endswith(f, ".jld2"), readdir(out_dir)))
@printf("Found %d hysteresis data files.\n", length(jld_files))

for fname in jld_files
    d = jldopen(joinpath(out_dir, fname), "r") do fh
        (kappa = fh["kappa"],
         kBT   = fh["kBT"],
         f0    = fh["f0_vals"],
         N     = fh["N"])
    end
    render_hysteresis_movie(
        d.kappa, d.kBT,
        minimum(d.f0), maximum(d.f0), length(d.f0),
        d.N, out_dir)
end

println("\nAll movies done.")
