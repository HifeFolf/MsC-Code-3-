# plot_com_translation.jl
# Validation 2: straight chain, no clamp (translation).
#
# Setup: straight chain, active force on all beads along tangent.
# Graph:  Panel 1 — COM speed vs time   → should plateau at v∞ = μ f₀
#         Panel 2 — max bond strain |εᵢ| → should stay tiny (< 1 %)
#
# All physics parameters come from Engine defaults, except κ:
#   The straight configuration buckles when f₀ > f₀_c ~ π²κ/L².
#   Engine default κ=1 gives f₀_c ≈ 0.025 for N=20, well below f₀=1.
#   We use κ=100 so that f₀_c ≈ 8 >> f₀=1, keeping the chain straight.

using GLMakie
using LinearAlgebra

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function com_vec(r::Matrix{Float64})
    N = size(r, 2)
    return [sum(r[1, :]) / N, sum(r[2, :]) / N]
end

function max_bond_strain(r::Matrix{Float64}, l::Float64)
    N = size(r, 2)
    ε_max = 0.0
    for i in 1:N-1
        d = norm(r[:, i+1] - r[:, i])
        ε_max = max(ε_max, abs(d - l) / l)
    end
    return ε_max
end

function main()
    frames   = 400
    substeps = 100

    # κ=100 raises the buckling threshold above f₀=1 (see header comment)
    p = Engine.Params(κ=100.0)
    N = p.N
    l = p.l

    v_inf = p.mu * p.f0      # theoretical terminal COM speed

    # --- initial straight chain along x ---
    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    state = Engine.State(copy(r0))

    Δt_frame = substeps * p.dt

    times        = Float64[0.0]
    com_speeds   = Float64[0.0]
    bond_strains = Float64[max_bond_strain(state.r, l)]

    c_prev = com_vec(r0)
    t      = 0.0

    # --- run ---
    for _ in 1:frames
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
        end
        t += Δt_frame

        c_now = com_vec(state.r)
        speed = norm(c_now - c_prev) / Δt_frame
        c_prev = c_now

        push!(times,        t)
        push!(com_speeds,   speed)
        push!(bond_strains, max_bond_strain(state.r, l))
    end

    # --- figure ---
    fig = Figure(size=(1000, 450))

    ax1 = Axis(fig[1, 1];
        xlabel = "Time  t",
        ylabel = "COM speed  |ṙ_cm|",
        title  = "COM translation speed  (Validation 2)",
    )
    lines!(ax1, times, com_speeds;
        linewidth = 2, color = :dodgerblue, label = "simulation")
    hlines!(ax1, [v_inf];
        color = :crimson, linestyle = :dash, linewidth = 1.5,
        label = "v∞ = μ f₀ = $(v_inf)")
    ylims!(ax1, 0.0, 1.5 * v_inf)
    axislegend(ax1; position = :rb)

    ax2 = Axis(fig[1, 2];
        xlabel = "Time  t",
        ylabel = "max |εᵢ|  =  max |(dᵢ − l)| / l",
        title  = "Max bond strain  (should stay tiny)",
        yscale = log10,
    )
    lines!(ax2, times, max.(bond_strains, 1e-16);
        linewidth = 2, color = :forestgreen)
    hlines!(ax2, [0.01];
        color = :orange, linestyle = :dash, linewidth = 1.5, label = "1 %")
    axislegend(ax2; position = :rt)

    display(fig)

    output_path = joinpath(@__DIR__, "com_translation.png")
    save(output_path, fig)
    println("Saved to: ", output_path)
    println("Theoretical  v∞ = μ f₀  = $(v_inf)")
    println("Simulated    v  (final) = $(com_speeds[end])  (should ≈ $(v_inf))")
    println("Max bond strain (final) = $(bond_strains[end])")
end

main()
