# Engine_ring.jl
# Physics engine for an active ring polymer with heterogeneous bead types.
#
# Bead types (stored in State.bead_types):
#   :active  — self-propulsion force f₀ t̂_i along local tangent
#   :loaded  — constant external load force (magnitude f_load, angle load_angle)
#   :passive — no propulsion, no load; purely mechanical
#
# Ring fixes vs Engine.jl:
#   bond_forces  — includes the closing bond N↔1
#   active_forces! — all beads use wrap-around two-neighbour tangent average

module EngineRing

using LinearAlgebra
using Random

export Params, State, step!

"""
Simulation parameters.
- N: number of beads
- l: rest bond length (r₀ = b)
- ks: bond spring stiffness
- κ: bending stiffness  (U_bend = (κ/l)(1 - cos θ))
- theta_max: maximum allowed bond angle (radians)
- theta_buf: buffer width for the log-wall barrier (radians)
- k_stop: stiffness of the hinge-stop log-wall barrier
- sigma: WCA bead diameter σ
- epsilon: WCA interaction strength ε
- topology: :open or :ring
- f0: active propulsion force magnitude (F_i^act = f₀ t̂_i)
- f_load: magnitude of the constant external load force on :loaded beads
- load_angle: direction of the load force (radians, lab frame)
- L: side length of the square confinement box (beads confined to [-L/2, L/2]²)
- pylon_spacing: grid spacing between fixed WCA pylons
- pylon_sigma: effective WCA σ for pylon-bead interactions
- pylon_epsilon: WCA ε for pylon-bead interactions
- mu: mobility μ (1/γ)
- kBT: thermal energy (set to 0 to disable noise)
- dt: time step
"""
Base.@kwdef struct Params
    N::Int     = 15        # number of beads (3 × 5-bead block)
    l::Float64 = 1.0
    ks::Float64 = 2000
    κ::Float64 = 1.0
    theta_max::Float64 = π
    theta_buf::Float64 = 0.1
    k_stop::Float64 = 0.0
    sigma::Float64 = 1.0
    epsilon::Float64 = 1.0
    topology::Symbol = :ring   # default: closed loop
    f0::Float64 = 20.0
    f_load::Float64 = 4.0      # load force magnitude
    load_angle::Float64 = -π/2 # load force direction (downward by default)
    L::Float64 = 30.0           # square box side length
    pylon_spacing::Float64 = 6.0 # grid spacing between pylons
    pylon_sigma::Float64 = 1.0   # effective WCA σ for pylon-bead interactions
    pylon_epsilon::Float64 = 1.0 # WCA ε for pylon-bead interactions
    mu::Float64 = 1.0
    kBT::Float64 = 0.1
    dt::Float64 = 0.00005   # ks·μ·dt = 0.1; chain stability (zig-zag): μ·3ks·dt = 0.3 < 1
end

"""
Simulation state.
- r: 2xN matrix of bead positions (each column is a 2D position)
- bead_types: length-N vector of Symbols (:active, :loaded, or :passive)
"""
mutable struct State
    r::Matrix{Float64}
    bead_types::Vector{Symbol}
end

"""
Compute bond-stretching forces: U_bond = (ks/2) Σ (|r_{i+1} - r_i| - l)².
For :ring topology the closing bond N↔1 is included.
Returns a 2xN force matrix.
"""
function bond_forces(r::Matrix{Float64}, p::Params)
    N       = p.N
    F       = zeros(size(r))
    n_bonds = (p.topology == :ring) ? N : N - 1

    for i in 1:n_bonds
        j = (i == N) ? 1 : i + 1   # wrap-around for closing bond
        d = r[:, j] - r[:, i]
        n = norm(d)
        if n == 0
            continue
        end
        f = p.ks * (n - p.l) * (d / n)
        F[:, i] += f
        F[:, j] -= f
    end
    return F
end

"""
Compute bending forces in-place (Isele-Holder potential + log-wall hinge stop).
U_bend = (κ/l)(1 - cos θ), plus an optional barrier that diverges at θ_max.
Supports :open and :ring topologies.
"""
function bending_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params)
    N = p.N
    prefac = p.κ / p.l
    θmax   = p.theta_max
    δ      = p.theta_buf
    θ0     = θmax - δ
    kstop  = p.k_stop

    start_idx = (p.topology == :ring) ? 1 : 2
    end_idx   = (p.topology == :ring) ? N : N - 1

    @inbounds for i in start_idx:end_idx
        im = (i == 1) ? N : i - 1
        ip = (i == N) ? 1 : i + 1

        ux = r[1,i] - r[1,im];  uy = r[2,i] - r[2,im]
        vx = r[1,ip] - r[1,i];  vy = r[2,ip] - r[2,i]

        nu2 = ux*ux + uy*uy
        nv2 = vx*vx + vy*vy
        if nu2 < 1e-24 || nv2 < 1e-24
            continue
        end
        nu = sqrt(nu2); nv = sqrt(nv2)

        ax = ux/nu; ay = uy/nu
        bx = vx/nv; by = vy/nv

        c = clamp(ax*bx + ay*by, -1.0, 1.0)

        dcos_du_x = (bx - c*ax)/nu
        dcos_du_y = (by - c*ay)/nu

        dcos_dv_x = (ax - c*bx)/nv
        dcos_dv_y = (ay - c*by)/nv

        θ = acos(c)
        dUdc_wall = 0.0

        if kstop > 0 && θ > θ0
            φ = θ - θ0
            φ = min(φ, δ*(1 - 1e-9))
            dUdθ = kstop * φ / (δ*δ - φ*φ)
            s = sqrt(max(1.0 - c*c, 1e-18))
            dθdc = -1.0 / s
            dUdc_wall = dUdθ * dθdc
        end

        dUdc = (-prefac) + dUdc_wall

        fx_im =  dUdc * dcos_du_x
        fy_im =  dUdc * dcos_du_y

        fx_ip = -dUdc * dcos_dv_x
        fy_ip = -dUdc * dcos_dv_y

        fx_i  = -(fx_im + fx_ip)
        fy_i  = -(fy_im + fy_ip)

        F[1,im] += fx_im;  F[2,im] += fy_im
        F[1,i]  += fx_i;   F[2,i]  += fy_i
        F[1,ip] += fx_ip;  F[2,ip] += fy_ip
    end

    return nothing
end

"""
Add Weeks–Chandler–Andersen repulsion between bead pairs in-place.
Excludes only 1–2 bonded pairs. Supports :open and :ring topologies.
"""
function wca_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params)
    N  = p.N
    σ  = p.sigma
    ϵ  = p.epsilon
    rc2 = (2.0^(1.0/6.0) * σ)^2
    σ2  = σ * σ

    @inbounds for i in 1:N-1
        xi = r[1,i]; yi = r[2,i]

        for j in i+1:N
            # skip only directly bonded pairs (1–2)
            if p.topology == :open
                j - i == 1 && continue
            else # :ring
                min(j - i, N - (j - i)) == 1 && continue
            end

            dx = r[1,j] - xi
            dy = r[2,j] - yi
            r2 = dx*dx + dy*dy

            if r2 >= rc2 || r2 < 1e-24
                continue
            end

            inv_r2 = 1.0 / r2
            sr2    = σ2 * inv_r2
            sr6    = sr2^3
            sr12   = sr6 * sr6

            fac = 24.0 * ϵ * inv_r2 * (2.0*sr12 - sr6)

            fx = fac * dx
            fy = fac * dy

            F[1,i] -= fx;  F[2,i] -= fy
            F[1,j] += fx;  F[2,j] += fy
        end
    end

    return nothing
end

"""
Compute active propulsion forces in-place: F_i^act = f₀ t̂_i.
Only applied to beads with bead_types[i] == :active.
For :ring topology every active bead averages its two neighbouring bond tangents
(wrap-around). For :open topology end beads use a single bond tangent.
"""
function active_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params,
                        bt::Vector{Symbol})
    N  = p.N
    f0 = p.f0

    @inbounds for i in 1:N
        (bt[i] == :active || bt[i] == :active_rev) || continue
        sign = (bt[i] == :active_rev) ? -1.0 : 1.0

        if p.topology == :ring
            # all beads are interior — use wrap-around neighbours
            im = (i == 1) ? N : i - 1
            ip = (i == N) ? 1 : i + 1
            dx1 = r[1,ip] - r[1,i];  dy1 = r[2,ip] - r[2,i]
            n1  = sqrt(dx1*dx1 + dy1*dy1)
            dx2 = r[1,i] - r[1,im];  dy2 = r[2,i] - r[2,im]
            n2  = sqrt(dx2*dx2 + dy2*dy2)
            dx  = dx1/max(n1, 1e-14) + dx2/max(n2, 1e-14)
            dy  = dy1/max(n1, 1e-14) + dy2/max(n2, 1e-14)
        elseif i == 1
            dx = r[1,2] - r[1,1];  dy = r[2,2] - r[2,1]
        elseif i == N
            dx = r[1,N] - r[1,N-1];  dy = r[2,N] - r[2,N-1]
        else
            dx1 = r[1,i+1] - r[1,i];  dy1 = r[2,i+1] - r[2,i]
            n1  = sqrt(dx1*dx1 + dy1*dy1)
            dx2 = r[1,i] - r[1,i-1];  dy2 = r[2,i] - r[2,i-1]
            n2  = sqrt(dx2*dx2 + dy2*dy2)
            dx  = dx1/max(n1, 1e-14) + dx2/max(n2, 1e-14)
            dy  = dy1/max(n1, 1e-14) + dy2/max(n2, 1e-14)
        end
        n = sqrt(dx*dx + dy*dy)
        if n < 1e-14
            continue
        end
        F[1,i] += sign * f0 * dx / n
        F[2,i] += sign * f0 * dy / n
    end
    return nothing
end

"""
Apply constant external load force to :loaded beads in-place.
F_i^load = f_load * (cos(load_angle), sin(load_angle))
"""
function load_forces!(F::Matrix{Float64}, p::Params, bt::Vector{Symbol})
    fx = p.f_load * cos(p.load_angle)
    fy = p.f_load * sin(p.load_angle)
    @inbounds for i in 1:p.N
        if bt[i] == :loaded
            F[1,i] += fx
            F[2,i] += fy
        end
    end
    return nothing
end

"""
Apply WCA repulsion from the four walls of the square box [-L/2, L/2]² in-place.
Each wall is treated as a flat WCA surface with the bead's own σ and ε.
"""
function wall_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params)
    hw  = p.L / 2.0
    σ   = p.sigma
    ϵ   = p.epsilon
    rc  = 2.0^(1.0/6.0) * σ

    @inbounds for i in 1:p.N
        x = r[1, i];  y = r[2, i]

        # For each wall, d = gap between bead centre and wall surface.
        # Force magnitude = 24ε/d * (2(σ/d)^12 - (σ/d)^6), directed away from wall.
        for (d, dim, sign) in (
                (hw - x, 1, -1.0),   # right wall  → push -x
                (x + hw, 1, +1.0),   # left wall   → push +x
                (hw - y, 2, -1.0),   # top wall    → push -y
                (y + hw, 2, +1.0))   # bottom wall → push +y
            d <= 1e-14 && continue
            d >= rc    && continue
            inv_d = 1.0 / d
            sr    = σ * inv_d
            sr6   = sr^6
            F[dim, i] += sign * 24.0 * ϵ * inv_d * (2.0*sr6*sr6 - sr6)
        end
    end
    return nothing
end

"""
Apply WCA repulsion from fixed pylons on a grid of spacing pylon_spacing,
positioned at (i·d, j·d) for all integers (i,j) inside the box, skipping (0,0).
"""
function pylon_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params)
    d   = p.pylon_spacing
    hw  = p.L / 2.0
    σ   = p.pylon_sigma
    ϵ   = p.pylon_epsilon
    rc2 = (2.0^(1.0/6.0) * σ)^2
    σ2  = σ * σ
    n_grid = floor(Int, hw / d)

    @inbounds for gi in -n_grid:n_grid
        px = gi * d
        for gj in -n_grid:n_grid
            gi == 0 && gj == 0 && continue   # no pylon at origin
            py = gj * d
            for i in 1:p.N
                dx = r[1, i] - px
                dy = r[2, i] - py
                r2 = dx*dx + dy*dy
                (r2 >= rc2 || r2 < 1e-24) && continue
                inv_r2 = 1.0 / r2
                sr2    = σ2 * inv_r2
                sr6    = sr2^3
                fac    = 24.0 * ϵ * inv_r2 * (2.0*sr6*sr6 - sr6)
                F[1, i] += fac * dx
                F[2, i] += fac * dy
            end
        end
    end
    return nothing
end

"""
Apply one overdamped Langevin step:
    ṙ = μ F_total + √(2μk_BT) ξ
Noise is disabled when kBT = 0.
"""
function step!(state::State, p::Params)
    r  = state.r
    bt = state.bead_types

    F = bond_forces(r, p)
    bending_forces!(F, r, p)
    wca_forces!(F, r, p)
    active_forces!(F, r, p, bt)
    load_forces!(F, p, bt)
    wall_forces!(F, r, p)
    pylon_forces!(F, r, p)

    noise_std = sqrt(2.0 * p.mu * p.kBT * p.dt)
    r .= r .+ p.dt .* p.mu .* F .+ noise_std .* randn(size(r))

    return state
end

end # module EngineRing
