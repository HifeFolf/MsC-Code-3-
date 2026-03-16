# plot_wca_diagnostics.jl
# WCA overlap / "no-crossing" diagnostics:
#   Plot B: minimum non-excluded pair distance r_min(t)/σ vs time
#   Plot C: histogram of r/σ for WCA-interacting pairs at steady state
# All physics parameters come from Engine defaults.

using GLMakie
using LinearAlgebra

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function min_wca_distance(r::Matrix{Float64}, p)
    N = p.N
    rmin = Inf
    @inbounds for i in 1:N-1
        for j in i+1:N
            if p.topology == :open
                j - i == 1 && continue
            else # :ring
                min(j - i, N - (j - i)) == 1 && continue
            end
            dx = r[1,j] - r[1,i]
            dy = r[2,j] - r[2,i]
            dist = sqrt(dx*dx + dy*dy)
            if dist < rmin
                rmin = dist
            end
        end
    end
    return rmin
end

function all_wca_distances(r::Matrix{Float64}, p)
    N = p.N
    dists = Float64[]
    @inbounds for i in 1:N-1
        for j in i+1:N
            if p.topology == :open
                j - i == 1 && continue
            else # :ring
                min(j - i, N - (j - i)) == 1 && continue
            end
            dx = r[1,j] - r[1,i]
            dy = r[2,j] - r[2,i]
            push!(dists, sqrt(dx*dx + dy*dy))
        end
    end
    return dists
end

function main()
    frames   = 500
    substeps = 100

    p = Engine.Params()       # single source of truth
    N = p.N
    l = p.l
    σ = p.sigma

    # --- initial zig-zag chain ---
    r0 = zeros(2, N)
    zig_angle = π/3
    for i in 2:N
        θ = iseven(i) ? zig_angle : -zig_angle
        r0[1, i] = r0[1, i-1] + l * cos(θ)
        r0[2, i] = r0[2, i-1] + l * sin(θ)
    end
    state = Engine.State(copy(r0))

    # --- storage for Plot B ---
    times           = Float64[]
    rmin_over_sigma = Float64[]
    t = 0.0

    function record_rmin!()
        rm = min_wca_distance(state.r, p)
        push!(rmin_over_sigma, rm / σ)
        push!(times, t)
    end
    record_rmin!()

    # --- run simulation ---
    for _ in 1:frames
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            t += p.dt
        end
        record_rmin!()
    end

    # --- collect final-frame distances for Plot C ---
    final_dists = all_wca_distances(state.r, p) ./ σ

    # --- figure ---
    fig = Figure(size=(1000, 500))

    ax1 = Axis(fig[1, 1];
        xlabel="Time",
        ylabel="rₘᵢₙ / σ",
        title="Min non-excluded pair distance",
    )
    lines!(ax1, times, rmin_over_sigma; linewidth=2, color=:crimson)
    hlines!(ax1, [1.0]; color=:black, linestyle=:dash, linewidth=1, label="r = σ")
    hlines!(ax1, [0.8]; color=:orange, linestyle=:dot, linewidth=1, label="r = 0.8σ")
    axislegend(ax1; position=:rb)

    ax2 = Axis(fig[1, 2];
        xlabel="r / σ",
        ylabel="Count",
        title="Pair distance histogram (final frame)",
    )
    hist!(ax2, final_dists; bins=30, color=:dodgerblue)
    vlines!(ax2, [1.0]; color=:black, linestyle=:dash, linewidth=1)
    vlines!(ax2, [0.8]; color=:orange, linestyle=:dot, linewidth=1)

    display(fig)

    output_path = joinpath(@__DIR__, "wca_diagnostics.png")
    save(output_path, fig)
    println("Saved plot to: ", output_path)
end

main()
