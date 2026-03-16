# plot_harmonic_trap.jl
# Validation N2: stationary distribution in a harmonic trap.
#
# One bead in a harmonic potential U = ½ k x², no active force.
# Overdamped Langevin:  dr = -μ k r dt + √(2μk_BT dt) ξ
#
# At equilibrium the position is Gaussian with variance:
#   ⟨x²⟩ = k_BT / k
#
# Graph: histogram of x-positions compared to the Gaussian N(0, k_BT/k).
# This validates that noise + drift satisfy fluctuation-dissipation.
#
# Notes on potential pitfalls:
#   - Trap force must be F = -k·x (centred at origin), not F = -k·(x - x₀)
#   - Burn-in must be long enough to forget the initial condition
#   - Discretisation bias: need k·μ·dt ≪ 1  (here 1·1·5e-5 = 5e-5, fine)
#   - Multiple independent trajectories improve statistics over one long run

using GLMakie
using Statistics

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function main()
    n_traj     = 50          # independent trajectories
    nsteps     = 2_000_000   # steps per trajectory
    burn_in    = 200_000     # discard initial transient (10 %)
    save_every = 50          # save every N steps to reduce autocorrelation

    # Physics parameters from Engine defaults
    p   = Engine.Params()
    mu  = p.mu
    kBT = p.kBT
    dt  = p.dt

    k_trap = 1.0                        # trap stiffness
    var_theory = kBT / k_trap           # ⟨x²⟩ = k_BT / k
    sigma_theory = sqrt(var_theory)

    noise_std = sqrt(2.0 * mu * kBT * dt)

    println("k·μ·dt = $(k_trap * mu * dt)  (should be ≪ 1)")
    println("Running $n_traj trajectories × $nsteps steps ...")

    # --- collect samples from many independent trajectories ---
    n_save_per = div(nsteps - burn_in, save_every)
    xs = Vector{Float64}(undef, n_traj * n_save_per)
    ys = Vector{Float64}(undef, n_traj * n_save_per)
    offset = 0

    for traj in 1:n_traj
        x = 0.0
        y = 0.0

        for s in 1:nsteps
            # F = -k r  (trap centred at origin)
            x += -mu * k_trap * x * dt + noise_std * randn()
            y += -mu * k_trap * y * dt + noise_std * randn()

            if s > burn_in && (s - burn_in) % save_every == 0
                offset += 1
                xs[offset] = x
                ys[offset] = y
            end
        end
    end

    var_x = var(xs)
    var_y = var(ys)
    var_sim = (var_x + var_y) / 2
    err = abs(var_sim - var_theory) / var_theory

    # --- theory curve ---
    x_range = range(-4 * sigma_theory, 4 * sigma_theory, length = 300)
    pdf_theory = @. exp(-x_range^2 / (2 * var_theory)) / sqrt(2π * var_theory)

    # --- figure ---
    fig = Figure(size=(700, 500))

    ax = Axis(fig[1, 1];
        xlabel = "Position  x",
        ylabel = "Probability density  P(x)",
        title  = "Stationary distribution in harmonic trap  (Validation N2)",
    )

    hist!(ax, xs;
        bins = 100, normalization = :pdf, color = (:dodgerblue, 0.5),
        label = "simulation  ($(length(xs)) samples)")

    lines!(ax, x_range, pdf_theory;
        color = :crimson, linewidth = 2.5,
        label = "theory: N(0, k_BT/k)   σ = $(round(sigma_theory, digits=3))")

    axislegend(ax; position = :rt)

    display(fig)

    output_path = joinpath(@__DIR__, "harmonic_trap.png")
    save(output_path, fig)
    println("Saved to: ", output_path)
    println()
    println("Theory:   ⟨x²⟩ = k_BT / k = $var_theory")
    println("Sim:      ⟨x²⟩ = $(round(var_x, digits=6))  (x-component)")
    println("          ⟨x²⟩ = $(round(var_y, digits=6))  (y-component)")
    println("          avg  = $(round(var_sim, digits=6))")
    println("Error:    $(round(err * 100, digits=2)) %")
end

main()
