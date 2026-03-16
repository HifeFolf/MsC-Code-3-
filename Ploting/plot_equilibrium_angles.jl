# plot_equilibrium_angles.jl
# Validation N3: polymer equilibrium roughness (bending angles).
#
# Turn off activity (f0=0), run the chain with bending and bonds.
# The bending potential is  U_bend = (κ/l)(1 - cos θ),
# so each interior angle θ_i follows a von Mises distribution:
#   P(θ) ∝ exp(κ cos θ / (l k_BT))
#
# Panel 1: histogram of θ_i for several κ values (fixed kBT)
#   → width decreases as κ increases, Boltzmann PDF overlaid
# Panel 2: histogram of θ_i for several kBT values (fixed κ)
#   → width increases as kBT increases, Boltzmann PDF overlaid
# Panel 3: ⟨θ²⟩ vs kBT for fixed κ
#   → exact von Mises ⟨θ²⟩ (numerical) + small-angle approximation

using GLMakie
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

# ═══════════════════════════════════════════════════════════════
#  Von Mises theory
# ═══════════════════════════════════════════════════════════════

"""
    von_mises_pdf(θ_range, κ, kBT; l=1.0) → Vector{Float64}

Exact Boltzmann PDF: P(θ) = exp(κ cos θ / (l kBT)) / Z,
where Z = ∫_{-π}^{π} exp(κ cos θ / (l kBT)) dθ.
Normalised numerically via the trapezoid rule.
"""
function von_mises_pdf(θ_range, κ::Float64, kBT::Float64; l::Float64=1.0)
    β = κ / (l * kBT)   # concentration parameter
    unnorm = [exp(β * cos(θ)) for θ in θ_range]
    # numerical normalisation over [-π, π]
    dθ = 2π / 10000
    θ_fine = range(-π, π, length=10001)
    Z = sum(exp(β * cos(θ)) for θ in θ_fine) * dθ
    return unnorm ./ Z
end

"""
    von_mises_theta2(κ, kBT; l=1.0) → Float64

Exact ⟨θ²⟩ for P(θ) ∝ exp(κ cos θ / (l kBT)), computed numerically.
"""
function von_mises_theta2(κ::Float64, kBT::Float64; l::Float64=1.0)
    β = κ / (l * kBT)
    dθ = 2π / 10000
    θ_fine = range(-π, π, length=10001)
    weights = [exp(β * cos(θ)) for θ in θ_fine]
    Z   = sum(weights) * dθ
    θ2  = sum(θ^2 * w for (θ, w) in zip(θ_fine, weights)) * dθ
    return θ2 / Z
end

# ═══════════════════════════════════════════════════════════════
#  Simulation
# ═══════════════════════════════════════════════════════════════

"""
    measure_angles(r)

Compute signed bending angles θ_i at interior beads (i = 2 … N-1).
θ_i is the angle between consecutive bond vectors.
"""
function measure_angles(r::Matrix{Float64})
    N = size(r, 2)
    angles = Float64[]
    for i in 2:N-1
        ux = r[1, i] - r[1, i-1];  uy = r[2, i] - r[2, i-1]
        vx = r[1, i+1] - r[1, i];  vy = r[2, i+1] - r[2, i]
        nu = sqrt(ux*ux + uy*uy)
        nv = sqrt(vx*vx + vy*vy)
        (nu < 1e-14 || nv < 1e-14) && continue
        cosθ = clamp((ux*vx + uy*vy) / (nu * nv), -1.0, 1.0)
        sinθ = ux*vy - uy*vx
        push!(angles, atan(sinθ / (nu * nv), cosθ))
    end
    return angles
end

"""
    run_passive_chain(; N, κ, kBT, nsteps, burn_in, save_every)

Run a passive chain (f0=0) and collect all interior bending angles.
"""
function run_passive_chain(; N::Int=20, κ::Float64=10.0, kBT::Float64=1.0,
                             nsteps::Int=2_000_000, burn_in::Int=200_000,
                             save_every::Int=100)
    p = Engine.Params(N=N, κ=κ, f0=0.0, kBT=kBT, epsilon=0.0)

    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + p.l
    end
    state = Engine.State(copy(r0))

    all_angles = Float64[]
    sizehint!(all_angles, div(nsteps - burn_in, save_every) * (N - 2))

    for s in 1:nsteps
        Base.invokelatest(Engine.step!, state, p)
        if s > burn_in && (s - burn_in) % save_every == 0
            append!(all_angles, measure_angles(state.r))
        end
    end
    return all_angles
end

function main()
    N = 20

    # ──── Panel 1: vary κ at fixed kBT ────
    kBT_fixed = 1.0
    κ_values  = [2.0, 5.0, 10.0, 20.0]
    angles_by_κ = Dict{Float64, Vector{Float64}}()

    println("Panel 1: varying κ at kBT = $kBT_fixed")
    for κ in κ_values
        print("  κ = $κ ... ")
        angles_by_κ[κ] = run_passive_chain(N=N, κ=κ, kBT=kBT_fixed)
        println("⟨θ²⟩ = $(round(mean(angles_by_κ[κ].^2), digits=4))")
    end

    # ──── Panel 2: vary kBT at fixed κ ────
    κ_fixed    = 10.0
    kBT_values = [0.5, 1.0, 2.0, 4.0]
    angles_by_kBT = Dict{Float64, Vector{Float64}}()

    println("\nPanel 2: varying kBT at κ = $κ_fixed")
    for kBT in kBT_values
        print("  kBT = $kBT ... ")
        angles_by_kBT[kBT] = run_passive_chain(N=N, κ=κ_fixed, kBT=kBT)
        println("⟨θ²⟩ = $(round(mean(angles_by_kBT[kBT].^2), digits=4))")
    end

    # ──── Panel 3: ⟨θ²⟩ vs kBT ────
    κ_quant   = 10.0
    kBT_sweep = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    theta2_sim         = Float64[]
    theta2_exact       = Float64[]
    theta2_smallangle  = Float64[]

    println("\nPanel 3: ⟨θ²⟩ vs kBT at κ = $κ_quant")
    for kBT in kBT_sweep
        print("  kBT = $kBT ... ")
        angles = run_passive_chain(N=N, κ=κ_quant, kBT=kBT)
        m = mean(angles .^ 2)
        push!(theta2_sim, m)
        push!(theta2_exact, von_mises_theta2(κ_quant, kBT))
        push!(theta2_smallangle, 1.0 * kBT / κ_quant)
        println("⟨θ²⟩ = $(round(m, digits=4))  " *
                "(exact=$(round(theta2_exact[end], digits=4)), " *
                "small-angle=$(round(theta2_smallangle[end], digits=4)))")
    end

    # ──── Figure ────
    colors_κ   = [:royalblue, :forestgreen, :darkorange, :crimson]
    colors_kBT = [:royalblue, :forestgreen, :darkorange, :crimson]

    fig = Figure(size=(1400, 450))

    # Panel 1: vary κ — histograms + Boltzmann overlay
    ax1 = Axis(fig[1, 1];
        xlabel = "Bending angle  θ  (rad)",
        ylabel = "P(θ)",
        title  = "Vary κ  (kBT = $kBT_fixed)",
    )
    θ_plot = range(-π, π, length=500)
    for (idx, κ) in enumerate(κ_values)
        hist!(ax1, angles_by_κ[κ]; bins=80, normalization=:pdf,
              color=(colors_κ[idx], 0.3))
        pdf_theory = von_mises_pdf(θ_plot, κ, kBT_fixed)
        lines!(ax1, θ_plot, pdf_theory;
            linewidth=2, color=colors_κ[idx], label="κ=$κ")
    end
    axislegend(ax1; position=:rt)

    # Panel 2: vary kBT — histograms + Boltzmann overlay
    ax2 = Axis(fig[1, 2];
        xlabel = "Bending angle  θ  (rad)",
        ylabel = "P(θ)",
        title  = "Vary kBT  (κ = $κ_fixed)",
    )
    for (idx, kBT) in enumerate(kBT_values)
        hist!(ax2, angles_by_kBT[kBT]; bins=80, normalization=:pdf,
              color=(colors_kBT[idx], 0.3))
        pdf_theory = von_mises_pdf(θ_plot, κ_fixed, kBT)
        lines!(ax2, θ_plot, pdf_theory;
            linewidth=2, color=colors_kBT[idx], label="kBT=$kBT")
    end
    axislegend(ax2; position=:rt)

    # Panel 3: ⟨θ²⟩ vs kBT — simulation + exact + small-angle
    ax3 = Axis(fig[1, 3];
        xlabel = "k_BT",
        ylabel = "⟨θ²⟩",
        title  = "⟨θ²⟩ vs kBT  (κ = $κ_quant)",
    )
    scatter!(ax3, kBT_sweep, theta2_sim;
        markersize=10, color=:dodgerblue, label="simulation")
    # exact von Mises (dense curve)
    kBT_fine = range(0.1, 9.0, length=100)
    θ2_exact_fine = [von_mises_theta2(κ_quant, kBT) for kBT in kBT_fine]
    lines!(ax3, kBT_fine, θ2_exact_fine;
        color=:black, linewidth=2, label="exact von Mises")
    # small-angle approximation
    lines!(ax3, kBT_fine, collect(kBT_fine) ./ κ_quant;
        color=:crimson, linewidth=2, linestyle=:dash,
        label="small-angle: l·kBT/κ")
    axislegend(ax3; position=:lt)

    display(fig)

    output_path = joinpath(@__DIR__, "equilibrium_angles.png")
    save(output_path, fig; px_per_unit=2)
    println("\nSaved to: ", output_path)
end

main()
