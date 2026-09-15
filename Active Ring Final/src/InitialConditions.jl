module InitialConditions

export antipolar_bead_types,
       perfect_circle_ic,
       kinked_circle_ic,
       double_kinked_circle_ic,
       triple_kinked_circle_ic,
       quadruple_kinked_circle_ic,
       quintuple_kinked_circle_ic,
       create_ic

using Main.EngineRing

"""
    antipolar_bead_types(N) -> Vector{Symbol}

Return a bead-type vector for an anti-polar ring.

**Odd N** (e.g. N = 17) — passive junction **on**:
- beads 1 … (N-1)÷2       : :active     (push counter-clockwise)
- bead  (N-1)÷2 + 1        : :passive    (front junction — convergence point)
- beads (N-1)÷2 + 2 … N   : :active_rev (push clockwise)

The single passive bead sits at the "front" junction where the two active
chains converge. The "back" junction (divergence point) is a direct
active / active_rev bond with no passive bead.

**Even N** (e.g. N = 16) — passive junction **off**:
- beads 1 … N÷2            : :active
- beads N÷2 + 1 … N        : :active_rev

No passive bead. The front junction (between bead N÷2 and bead N÷2+1) is a
direct active / active_rev bond, mirroring the back junction.

N must be ≥ 2.
"""
function antipolar_bead_types(N::Int)
    @assert N >= 2 "N must be ≥ 2 (got N = $N)"
    if isodd(N)
        n_active = (N - 1) ÷ 2
        return vcat(fill(:active,     n_active),
                    [:passive],
                    fill(:active_rev, n_active))
    else
        n_half = N ÷ 2
        return vcat(fill(:active,     n_half),
                    fill(:active_rev, n_half))
    end
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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Reflect r0[:, target] across the chord spanned by r0[:, a] and r0[:, b].
function _reflect_across_chord!(r0, target::Int, a::Int, b::Int)
    ax = r0[1, a]; ay = r0[2, a]
    bx = r0[1, b]; by = r0[2, b]
    px = r0[1, target]; py = r0[2, target]
    vx = bx - ax; vy = by - ay
    t  = ((px - ax)*vx + (py - ay)*vy) / (vx*vx + vy*vy)
    fx = ax + t*vx; fy = ay + t*vy
    r0[1, target] = 2*fx - px
    r0[2, target] = 2*fy - py
end

# Build a perfect-circle layout and apply a k-level inward kink to every
# passive bead.  Two-phase construction:
#
# Phase 1 — inner kink, depth k_inner = min(k, 3), outside-in:
#   for j = k_inner-1 down to 1:
#     reflect i-j across chord(i-j-1, i-j+1)   [outer already done, inner on circle]
#     reflect i+j across chord(i+j-1, i+j+1)   [inner on circle, outer already done]
#   reflect passive i across chord(i-1', i+1')
#
# Phase 2 — outer spread, k > 3 only, inside-out (j = k_inner … k-1):
#   reflect i-j across chord(i-j-1, i-j+1')   [outer on circle, inner already folded]
#   reflect i+j across chord(i+j-1', i+j+1)
#
# For k ≤ 3 only Phase 1 runs (original behaviour).
# For k ≥ 4 the inner V is frozen at the k=3 depth; Phase 2 folds additional
# beads outward from the k=3 boundary, widening the kink without deepening it.
# This avoids the i±1 bead overlap that occurs for k ≥ 4 inner-only reflections
# (separation drops below σ when k=4: 0.55σ for N=17, l=σ=1).
#
# Beads moved: k=1→1, k=2→3, k=3→5, k=4→7, k=5→9.
function _kinked_circle_ic(p, bead_types::Vector{Symbol}, k::Int)
    N  = p.N
    l  = p.l
    R  = l / (2 * sin(π / N))
    r0 = zeros(2, N)
    for i in 1:N
        φ = 2π * (i - 1) / N
        r0[1, i] = R * cos(φ)
        r0[2, i] = R * sin(φ)
    end
    for i in 1:N
        bead_types[i] == :passive || continue

        # Phase 1: inner kink (outside-in, depth capped at 3)
        k_inner = min(k, 3)
        for j in (k_inner - 1):-1:1
            im     = mod1(i - j,     N)
            ip     = mod1(i + j,     N)
            im_out = mod1(i - j - 1, N)   # outer: already processed (or circle if j == k_inner-1)
            im_in  = mod1(i - j + 1, N)   # inner: still on circle
            ip_in  = mod1(i + j - 1, N)   # inner: still on circle
            ip_out = mod1(i + j + 1, N)   # outer: already processed (or circle if j == k_inner-1)
            _reflect_across_chord!(r0, im, im_out, im_in)
            _reflect_across_chord!(r0, ip, ip_in,  ip_out)
        end
        _reflect_across_chord!(r0, i, mod1(i - 1, N), mod1(i + 1, N))

        # Phase 2: outer spread (inside-out, only for k ≥ 4)
        for j in k_inner:(k - 1)
            im_bdy = mod1(i - j,     N)   # bead to fold
            im_out = mod1(i - j - 1, N)   # outer neighbour: still on circle
            im_in  = mod1(i - j + 1, N)   # inner neighbour: already folded
            ip_in  = mod1(i + j - 1, N)   # inner neighbour: already folded
            ip_bdy = mod1(i + j,     N)   # bead to fold
            ip_out = mod1(i + j + 1, N)   # outer neighbour: still on circle
            _reflect_across_chord!(r0, im_bdy, im_out, im_in)
            _reflect_across_chord!(r0, ip_bdy, ip_in,  ip_out)
        end
    end
    return EngineRing.State(copy(r0), copy(bead_types))
end

# ---------------------------------------------------------------------------
# Public IC constructors
# ---------------------------------------------------------------------------

"""
    kinked_circle_ic(p, bead_types) -> EngineRing.State

Like `perfect_circle_ic`, but the passive bead is reflected **inward** across
the chord that connects its two neighbours.

Geometry
--------
Start from the perfect circle. For the passive bead at position P, with
neighbours A (bead p-1) and B (bead p+1):

  1. Drop a perpendicular from P onto the chord AB; call the foot F.
  2. Place the passive bead at P' = 2F − P  (mirror image of P through chord AB).

Because reflection is an isometry, distances are exactly preserved:
  |P'−A| = |P−A| = l   and   |P'−B| = |P−B| = l

The angle ∠AP'B equals ∠APB in magnitude, so the bending stiffness sees
exactly the same-size deviation as everywhere else on the circle — it just
points inward (towards the ring centre) instead of outward.

Every other bead position, all bead types, and all simulation parameters are
unchanged.
"""
kinked_circle_ic(p, bead_types::Vector{Symbol}) = _kinked_circle_ic(p, bead_types, 1)

"""
    double_kinked_circle_ic(p, bead_types) -> EngineRing.State

Like `kinked_circle_ic`, but the kink spreads to the passive bead's immediate
neighbours, producing a deeper inward V centred on the passive bead.

Three-step outside-in construction (per passive bead at index i):

  1. Reflect bead i-1 across chord(i-2, i)   — pushes i-1 inward.
  2. Reflect bead i+1 across chord(i, i+2)   — pushes i+1 inward.
     (Steps 1 & 2 use the passive bead still on the circle.)
  3. Reflect bead i   across chord(i-1', i+1') — now the chord is inward,
     so the passive bead ends up much further in than a single kink.

Beads at distance ≥ 2 from the passive (i±2, i±3, …) remain on the circle.
"""
double_kinked_circle_ic(p, bead_types::Vector{Symbol}) = _kinked_circle_ic(p, bead_types, 2)

"""
    triple_kinked_circle_ic(p, bead_types) -> EngineRing.State

Outside-in kink of depth 3: beads i±2 are reflected first (using unmodified
neighbours), then i±1 (using the already-inward i±2), then the passive bead
(using the already-inward i±1).  Five beads total are moved; beads at distance
≥ 3 from the passive stay on the circle.
"""
triple_kinked_circle_ic(p, bead_types::Vector{Symbol}) = _kinked_circle_ic(p, bead_types, 3)

"""
    quadruple_kinked_circle_ic(p, bead_types) -> EngineRing.State

Wider variant of `triple_kinked_circle_ic`: the inner k=3 V (i±2, i±1, i)
is built first, then i±3 are additionally folded inward from the k=3 boundary
(inside-out, using the already-folded i±2' as the inner reference).

Seven beads total are moved; the passive bead depth is the same as k=3.
Beads at distance ≥ 4 from the passive stay on the circle.
"""
quadruple_kinked_circle_ic(p, bead_types::Vector{Symbol}) = _kinked_circle_ic(p, bead_types, 4)

"""
    quintuple_kinked_circle_ic(p, bead_types) -> EngineRing.State

Wider variant of `quadruple_kinked_circle_ic`: the inner k=3 V (i±2, i±1, i)
is built first, then i±3 and i±4 are folded inward in sequence from the k=3
boundary outward.

Nine beads total are moved; the passive bead depth is the same as k=3.
Beads at distance ≥ 5 from the passive stay on the circle.
"""
quintuple_kinked_circle_ic(p, bead_types::Vector{Symbol}) = _kinked_circle_ic(p, bead_types, 5)

"""
    create_ic(layout, p, bead_types) -> EngineRing.State

Dispatch helper: pick an initial condition by layout name.
Supported layouts:
  "antipolar"                — perfect circle
  "antipolar_kinked"         — single kink  (k=1, 1 bead moved)
  "antipolar_doublekinked"   — double kink  (k=2, 3 beads moved)
  "antipolar_triplekinked"   — triple kink  (k=3, 5 beads moved)
  "antipolar_quadkinked"     — quadruple kink (k=4, 7 beads moved)
  "antipolar_quintkinked"    — quintuple kink (k=5, 9 beads moved)
"""
function create_ic(layout::AbstractString, p, bead_types::Vector{Symbol})
    layout == "antipolar"              && return perfect_circle_ic(p, bead_types)
    layout == "antipolar_kinked"       && return kinked_circle_ic(p, bead_types)
    layout == "antipolar_doublekinked" && return double_kinked_circle_ic(p, bead_types)
    layout == "antipolar_triplekinked" && return triple_kinked_circle_ic(p, bead_types)
    layout == "antipolar_quadkinked"   && return quadruple_kinked_circle_ic(p, bead_types)
    layout == "antipolar_quintkinked"  && return quintuple_kinked_circle_ic(p, bead_types)
    error("Unknown bead_layout: \"$layout\"")
end

end # module InitialConditions
