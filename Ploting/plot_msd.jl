# plot_msd.jl
# Mean Squared Displacement of the ring centre of mass.
#
# Three scenarios on one log-log plot:
#   (blue)  active + pylons + walls  — default params
#   (green) active, no pylons        — pylon_spacing = 1e6 (disabled)
#   (gray)  passive (f0 = 0)         — thermal reference
#
# Time-averaged MSD:
#   MSD(τ) = (1/(T−τ)) Σ_t |r_com(t+τ) − r_com(t)|²

using GLMakie

include(joinpath(@__DIR__, "..", "Simulation", "Engine_ring.jl"))

function init_state(p::EngineRing.Params)
    block = [:active, :active, :active, :loaded, :passive]
    types = repeat(block, 3)   # length 15
    N = p.N;  l = p.l
    R  = l / (2 * sin(π / N))
    r0 = zeros(2, N)
    for i in 1:N
        φ = 2π * (i - 1) / N
        r0[1, i] = R * cos(φ)
        r0[2, i] = R * sin(φ)
    end
    return EngineRing.State(r0, types)
end

"""
Run simulation; discard `warmup` frames, then record COM every `substeps` steps.
Returns a 2×(frames+1) matrix of COM positions.
"""
function run_com(p::EngineRing.Params; warmup=500, frames=5000, substeps=100)
    state = init_state(p)
    for _ in 1:warmup, _ in 1:substeps
        EngineRing.step!(state, p)
    end
    com = zeros(2, frames + 1)
    com[:, 1] = vec(sum(state.r, dims=2)) ./ p.N
    for k in 1:frames
        for _ in 1:substeps
            EngineRing.step!(state, p)
        end
        com[:, k+1] = vec(sum(state.r, dims=2)) ./ p.N
    end
    return com
end

"""Time-averaged MSD from a 2×T COM trajectory."""
function tamsd(com::Matrix{Float64}, dt_frame::Float64)
    T       = size(com, 2)
    max_lag = T ÷ 2
    msd     = zeros(max_lag)
    @inbounds for lag in 1:max_lag
        n = T - lag
        s = 0.0
        for t in 1:n
            dx = com[1, t+lag] - com[1, t]
            dy = com[2, t+lag] - com[2, t]
            s += dx*dx + dy*dy
        end
        msd[lag] = s / n
    end
    return (1:max_lag) .* dt_frame, msd
end

function main()
    frames   = 5000
    substeps = 100
    warmup   = 500
    p0       = EngineRing.Params()
    dt_frame = substeps * p0.dt    # physical time per recorded frame

    p_full    = p0
    p_nopylon = EngineRing.Params(pylon_spacing = 1e6)   # n_grid=0 → no pylons
    p_passive = EngineRing.Params(f0 = 0.0)

    println("Case 1/3: active + pylons + walls …")
    com1 = run_com(p_full;    warmup=warmup, frames=frames, substeps=substeps)
    println("Case 2/3: active, no pylons …")
    com2 = run_com(p_nopylon; warmup=warmup, frames=frames, substeps=substeps)
    println("Case 3/3: passive …")
    com3 = run_com(p_passive; warmup=warmup, frames=frames, substeps=substeps)

    println("Computing MSD …")
    t1, msd1 = tamsd(com1, dt_frame)
    t2, msd2 = tamsd(com2, dt_frame)
    t3, msd3 = tamsd(com3, dt_frame)

    # --- figure ---
    fig = Figure(size = (820, 580))
    ax  = Axis(fig[1, 1];
        xlabel = "lag time  τ",
        ylabel = "MSD  ⟨|Δr_com|²⟩",
        title  = "Ring COM Mean Squared Displacement",
        xscale = log10,
        yscale = log10,
    )

    lines!(ax, t1, msd1; linewidth=2, color=:dodgerblue,  label="active + pylons")
    lines!(ax, t2, msd2; linewidth=2, color=:forestgreen, label="active, no pylons")
    lines!(ax, t3, msd3; linewidth=2, color=:gray60,      label="passive + pylons")

    # Guide lines: τ¹ and τ² anchored at geometric midpoint of time axis
    t_lo, t_hi = t1[1], t1[end]
    t_anch = exp((log(t_lo) + log(t_hi)) / 2)
    idx    = argmin(abs.(t1 .- t_anch))
    A      = msd1[idx] * 5.0   # place guides above data
    tref   = [t_lo, t_hi]
    lines!(ax, tref, A .* (tref ./ t_anch).^1;
        color=:black, linestyle=:dash, linewidth=1.2, label="∝ τ¹  (diffusive)")
    lines!(ax, tref, A .* (tref ./ t_anch).^2;
        color=:black, linestyle=:dot,  linewidth=1.2, label="∝ τ²  (ballistic)")

    axislegend(ax; position=:lt)

    display(fig)
    out = joinpath(@__DIR__, "msd.png")
    save(out, fig)
    println("Saved to: ", out)
end

main()
