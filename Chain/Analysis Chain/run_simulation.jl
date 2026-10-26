# Analysis/run_simulation.jl
# Run the physics engine, record all data/metadata, and save to Analysis.

using Dates
using LinearAlgebra
using Serialization

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))
import .Engine: Params, State, step!

function default_initial_chain(N::Int, l::Float64)
    r0 = zeros(2, N)
    for i in 2:N
        r0[1, i] = r0[1, i-1] + l
    end
    return r0
end

function compute_tangents(r::Matrix{Float64})
    N = size(r, 2)
    tangents = zeros(2, N - 1)
    for i in 1:(N - 1)
        d = r[:, i + 1] - r[:, i]
        n = norm(d)
        tangents[:, i] = n > 0 ? d / n : [1.0, 0.0]
    end
    return tangents
end

function simulate(; N::Int=12, l::Float64=1.0, ks::Float64=100.0, κ::Float64=10.0,
                    theta_max::Float64=π, theta_buf::Float64=0.1, k_stop::Float64=0.0,
                    sigma::Float64=1.0, epsilon::Float64=1.0, exclude_13::Bool=true,
                    topology::Symbol=:open, f0::Float64=0.0, mu::Float64=1.0,
                    kBT::Float64=0.0, dt::Float64=0.05,
                    steps::Int=200, outdir::String=@__DIR__)
    p = Params(N=N, l=l, ks=ks, κ=κ, theta_max=theta_max, theta_buf=theta_buf,
               k_stop=k_stop, sigma=sigma, epsilon=epsilon, exclude_13=exclude_13,
               topology=topology, f0=f0, mu=mu, kBT=kBT, dt=dt)
    state = State(default_initial_chain(N, l))

    times = collect(0.0:dt:steps * dt)

    positions = Array{Float64}(undef, 2, N, steps + 1)
    velocities = Array{Float64}(undef, 2, N, steps + 1)
    tangents = Array{Float64}(undef, 2, N - 1, steps + 1)

    positions[:, :, 1] = state.r
    velocities[:, :, 1] .= 0.0
    tangents[:, :, 1] = compute_tangents(state.r)

    for s in 1:steps
        Base.invokelatest(step!, state, p)
        positions[:, :, s + 1] = state.r
        tangents[:, :, s + 1] = compute_tangents(state.r)
        velocities[:, :, s + 1] = (positions[:, :, s + 1] - positions[:, :, s]) ./ dt
    end

    meta = Dict(
        "created_at" => string(now()),
        "julia_version" => string(VERSION),
        "platform" => string(Sys.KERNEL),
        "params" => Dict(
            "N" => N,
            "l" => l,
            "ks" => ks,
            "κ" => κ,
            "theta_max" => theta_max,
            "theta_buf" => theta_buf,
            "k_stop" => k_stop,
            "sigma" => sigma,
            "epsilon" => epsilon,
            "exclude_13" => exclude_13,
            "topology" => string(topology),
            "f0" => f0,
            "mu" => mu,
            "kBT" => kBT,
            "dt" => dt,
            "steps" => steps
        )
    )

    data = Dict(
        "positions" => positions,
        "velocities" => velocities,
        "tangents" => tangents,
        "time" => times,
        "meta" => meta
    )

    stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    outfile = joinpath(outdir, "active_chain_" * stamp * ".jls")

    open(outfile, "w") do io
        serialize(io, data)
    end

    println("Saved analysis to: ", outfile)
    return outfile
end

# Run with defaults if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    simulate()
end
