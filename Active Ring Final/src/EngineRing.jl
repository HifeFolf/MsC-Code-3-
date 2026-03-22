module EngineRing

using LinearAlgebra
using Random

export Params, State, step!

"""
Simulation parameters.
- N:         number of beads
- l:         rest bond length
- ks:        bond spring stiffness
- κ:         bending stiffness  (U_bend = (κ/l)(1 - cos θ))
- theta_max: maximum allowed bond angle (radians)
- theta_buf: buffer width for the log-wall barrier (radians)
- k_stop:    stiffness of the hinge-stop log-wall barrier
- sigma:     WCA bead diameter σ
- epsilon:   WCA interaction strength ε
- topology:  :ring (default) or :open
- f0:        active propulsion force magnitude
- mu:        mobility μ (1/γ)
- kBT:       thermal energy (set to 0 to disable noise)
- dt:        time step
"""
Base.@kwdef struct Params
    N::Int         = 16
    l::Float64     = 1.0
    ks::Float64    = 2000.0
    κ::Float64     = 10.0
    theta_max::Float64 = π
    theta_buf::Float64 = 0.1
    k_stop::Float64    = 0.0
    sigma::Float64     = 1.0
    epsilon::Float64   = 1.0
    topology::Symbol   = :ring
    f0::Float64    = 20.0
    mu::Float64    = 1.0
    kBT::Float64   = 1
    dt::Float64    = 0.00005
end

"""
Simulation state.
- r:          2×N matrix of bead positions (each column is a 2D position)
- bead_types: length-N vector of Symbols (:active or :passive)
"""
mutable struct State
    r::Matrix{Float64}
    bead_types::Vector{Symbol}
end

"""
Compute bond-stretching forces: U_bond = (ks/2) Σ (|r_{i+1} - r_i| - l)².
For :ring topology the closing bond N↔1 is included.
Returns a 2×N force matrix.
"""
function bond_forces(r::Matrix{Float64}, p::Params)
    N       = p.N
    F       = zeros(size(r))
    n_bonds = (p.topology == :ring) ? N : N - 1

    for i in 1:n_bonds
        j = (i == N) ? 1 : i + 1
        d = r[:, j] - r[:, i]
        n = norm(d)
        n == 0 && continue
        f = p.ks * (n - p.l) * (d / n)
        F[:, i] += f
        F[:, j] -= f
    end
    return F
end

"""
Compute bending forces in-place: U_bend = (κ/l)(1 - cos θ),
plus an optional log-wall barrier that diverges at theta_max.
Supports :open and :ring topologies.
"""
function bending_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params)
    N      = p.N
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
        (nu2 < 1e-24 || nv2 < 1e-24) && continue
        nu = sqrt(nu2);  nv = sqrt(nv2)

        ax = ux/nu;  ay = uy/nu
        bx = vx/nv;  by = vy/nv

        c = clamp(ax*bx + ay*by, -1.0, 1.0)

        dcos_du_x = (bx - c*ax)/nu;  dcos_du_y = (by - c*ay)/nu
        dcos_dv_x = (ax - c*bx)/nv;  dcos_dv_y = (ay - c*by)/nv

        θ = acos(c)
        dUdc_wall = 0.0
        if kstop > 0 && θ > θ0
            φ = min(θ - θ0, δ*(1 - 1e-9))
            dUdθ = kstop * φ / (δ*δ - φ*φ)
            s = sqrt(max(1.0 - c*c, 1e-18))
            dUdc_wall = dUdθ * (-1.0 / s)
        end

        dUdc = (-prefac) + dUdc_wall

        fx_im =  dUdc * dcos_du_x;  fy_im =  dUdc * dcos_du_y
        fx_ip = -dUdc * dcos_dv_x;  fy_ip = -dUdc * dcos_dv_y
        fx_i  = -(fx_im + fx_ip);   fy_i  = -(fy_im + fy_ip)

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
    N   = p.N
    σ   = p.sigma
    ϵ   = p.epsilon
    rc2 = (2.0^(1.0/6.0) * σ)^2
    σ2  = σ * σ

    @inbounds for i in 1:N-1
        xi = r[1,i];  yi = r[2,i]
        for j in i+1:N
            if p.topology == :open
                j - i == 1 && continue
            else
                min(j - i, N - (j - i)) == 1 && continue
            end

            dx = r[1,j] - xi;  dy = r[2,j] - yi
            r2 = dx*dx + dy*dy
            (r2 >= rc2 || r2 < 1e-24) && continue

            inv_r2 = 1.0 / r2
            sr2    = σ2 * inv_r2
            sr6    = sr2^3
            fac    = 24.0 * ϵ * inv_r2 * (2.0*sr6*sr6 - sr6)

            F[1,i] -= fac*dx;  F[2,i] -= fac*dy
            F[1,j] += fac*dx;  F[2,j] += fac*dy
        end
    end
    return nothing
end

"""
Compute active propulsion forces in-place: F_i^act = ±f₀ t̂_i.
Applied to :active beads (+t̂_i) and :active_rev beads (-t̂_i).
Each active bead averages its two neighbouring bond tangents (wrap-around for
:ring; end beads fall back to a single bond tangent for :open).
"""
function active_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params,
                        bt::Vector{Symbol})
    N  = p.N
    f0 = p.f0

    @inbounds for i in 1:N
        (bt[i] == :active || bt[i] == :active_rev) || continue
        sign = bt[i] == :active ? 1.0 : -1.0

        if p.topology == :ring
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
        n < 1e-14 && continue
        F[1,i] += sign * f0 * dx / n
        F[2,i] += sign * f0 * dy / n
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

    noise_std = sqrt(2.0 * p.mu * p.kBT * p.dt)
    r .= r .+ p.dt .* p.mu .* F .+ noise_std .* randn(size(r))

    return state
end

end # module EngineRing
