# plot_relative_angle_pdf.jl
# PDF of the relative bond angle  Δθ = θ_{i+1} - θ_i
# where θ_i = atan2(Δy_i, Δx_i) is the tangent angle of bond i.
#
# Sweeps several f0 values (including f0=0 passive) to show how
# activity reshapes the turning-angle distribution.

using GLMakie
using Statistics

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

"""
    measure_relative_angles(r) → Vector{Float64}

For an N-bead chain with N-1 bonds, compute the N-2 relative angles
Δθ_i = θ_{i+1} - θ_i, wrapped to (-π, π].
"""
function measure_relative_angles(r::Matrix{Float64})
    N = size(r, 2)
    n_bonds = N - 1
    if n_bonds < 2
        return Float64[]
    end

    # bond tangent angles
    θ = Vector{Float64}(undef, n_bonds)
    for b in 1:n_bonds
        dx = r[1, b+1] - r[1, b]
        dy = r[2, b+1] - r[2, b]
        θ[b] = atan(dy, dx)
    end

    # relative angles (wrapped)
    Δθ = Vector{Float64}(undef, n_bonds - 1)
    for i in 1:n_bonds-1
        d = θ[i+1] - θ[i]
        # wrap to (-π, π]
        d = d - 2π * round(d / (2π))
        Δθ[i] = d
    end
    return Δθ
end

"""
    collect_relative_angles(; κ, f0, N, nsteps, burn_in, save_every)

Run a free (unclamped) active chain and collect relative angles over time.
"""
function collect_relative_angles(; κ::Float64=19.0, f0::Float64=0.0,
        N::Int=20, nsteps::Int=2_000_000,
        burn_in::Int=200_000, save_every::Int=100)

    p = Engine.Params(N=N, κ=κ, f0=f0)
    l = p.l

    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    all_Δθ = Float64[]
    n_samples = div(nsteps - burn_in, save_every) * (N - 2)
    sizehint!(all_Δθ, n_samples)

    for s in 1:nsteps
        Base.invokelatest(Engine.step!, state, p)
        if s > burn_in && (s - burn_in) % save_every == 0
            append!(all_Δθ, measure_relative_angles(state.r))
        end
    end
    return all_Δθ
end

function main()
    N = 20
    κ = 19.0
    f0_values = [0.0, 1.5, 5.0, 25.0]
    colors = [:gray50, :dodgerblue, :forestgreen, :crimson]

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1];
        xlabel = "Δθ  (rad)",
        ylabel = "P(Δθ)",
        title  = "Relative angle PDF   Δθ = θᵢ₊₁ − θᵢ   (N=$N, κ=$κ)",
    )

    for (k, f0) in enumerate(f0_values)
        label = f0 == 0 ? "passive (f0=0)" : "f0=$f0"
        println("Running f0=$f0 ...")
        Δθ = collect_relative_angles(; κ=κ, f0=f0, N=N)
        println("  collected $(length(Δθ)) samples,  ⟨Δθ²⟩ = $(round(mean(Δθ.^2), digits=5))")
        hist!(ax, Δθ; bins=80, normalization=:pdf,
              color=(colors[k], 0.4), label=label)
    end

    axislegend(ax; position=:rt)
    display(fig)

    output_path = joinpath(@__DIR__, "relative_angle_pdf.png")
    save(output_path, fig; px_per_unit=2)
    println("\nSaved to: ", output_path)
end

main()
