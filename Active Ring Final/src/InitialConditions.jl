module InitialConditions

export antipolar_bead_types, perfect_circle_ic

using Main.EngineRing

"""
    antipolar_bead_types(N) -> Vector{Symbol}

Return a bead-type vector for an anti-polar ring:
- beads 1 … N÷2   : :active     (push counter-clockwise)
- beads N÷2+1 … N : :active_rev (push clockwise)

N must be even.
"""
function antipolar_bead_types(N::Int)
    @assert iseven(N) "N must be even for an anti-polar layout (got N = $N)"
    return vcat(fill(:active, N ÷ 2), fill(:active_rev, N ÷ 2))
end

"""
    perfect_circle_ic(p, bead_types) -> EngineRing.State

Place N beads on a perfect circle whose radius is chosen so that the
equilibrium bond length l is exactly satisfied:

    R = l / (2 sin(π / N))

Bead i is placed at angle φ_i = 2π(i-1)/N (counter-clockwise, starting from +x).
"""
function perfect_circle_ic(p, bead_types::Vector{Symbol})
    N  = p.N
    l  = p.l
    R  = l / (2 * sin(π / N))
    r0 = zeros(2, N)
    for i in 1:N
        φ = 2π * (i - 1) / N
        r0[1, i] = R * cos(φ)
        r0[2, i] = R * sin(φ)
    end
    return EngineRing.State(copy(r0), copy(bead_types))
end

end # module InitialConditions
