module Engine

using LinearAlgebra
using Random

export Params, State, step!

Base.@kwdef struct Params
    N::Int     = 20
    κ::Float64 = 5.0
    f0::Float64 = 25.0
    topology::Symbol = :open

    # Brownian units (fixed)
    l::Float64 = 1.0
    mu::Float64 = 1.0
    kBT::Float64 = 1.0

    # Excluded volume (fixed)
    sigma::Float64 = 1.0
    epsilon::Float64 = 1.0

    # Constraints (disabled)
    k_stop::Float64 = 0.0
    theta_max::Float64 = π
    theta_buf::Float64 = 0.1

    # Numerical (effectively inextensible bonds)
    ks::Float64 = 4444
    dt::Float64 = 0.00005
end

"""
- r: 2xN matrix of bead positions (each column is a 2D position)
"""
mutable struct State
    r::Matrix{Float64}
end

"""
Compute bond-stretching forces: U_bond = (ks/2) Σ (|r_{i+1} - r_i| - l)².
Returns a 2xN force matrix.
"""
function bond_forces(r::Matrix{Float64}, p::Params)
    N = p.N
    F = zeros(size(r))
    for i in 1:(N-1)
        d = r[:, i+1] - r[:, i]
        n = norm(d)
        if n == 0
            continue
        end
        # F = -dU/dr = -ks (|d| - l) * d̂
        f = p.ks * (n - p.l) * (d / n)
        F[:, i]   += f
        F[:, i+1] -= f
    end
    return F
end

"""
Compute bending forces in-place (potential + log-wall hinge stop).
U_bend = (κ/l)(1 - cos θ), plus an optional barrier that diverges at θ_max.
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

        # unit vectors
        ax = ux/nu; ay = uy/nu
        bx = vx/nv; by = vy/nv

        # cos(theta)
        c = clamp(ax*bx + ay*by, -1.0, 1.0)

        # ∂cos/∂u = (b - c a)/|u|
        dcos_du_x = (bx - c*ax)/nu
        dcos_du_y = (by - c*ay)/nu

        # ∂cos/∂v = (a - c b)/|v|
        dcos_dv_x = (ax - c*bx)/nv
        dcos_dv_y = (ay - c*by)/nv

        # ---------- hinge-stop barrier (inactive until θ > θ0) ----------
        θ = acos(c)
        dUdc_wall = 0.0

        if kstop > 0 && θ > θ0
            φ = θ - θ0
            φ = min(φ, δ*(1 - 1e-9))

            # U_wall(φ) = -(kstop/2) * log(1 - (φ/δ)^2)
            # dU/dφ = kstop * φ / (δ^2 - φ^2)
            dUdθ = kstop * φ / (δ*δ - φ*φ)

            # dθ/dc = -1/sinθ
            s = sqrt(max(1.0 - c*c, 1e-18))
            dθdc = -1.0 / s

            dUdc_wall = dUdθ * dθdc
        end

        # total dU/dc: U_bend = prefac*(1 - c) => dU/dc = -prefac
        dUdc = (-prefac) + dUdc_wall

        # Forces: F_im = dUdc * ∂c/∂u,  F_ip = -dUdc * ∂c/∂v
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

            # F = 24ε/r² * [2(σ/r)¹² - (σ/r)⁶] · r̂
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
The force is split equally between the two beads sharing the bond:
    F_i   += -(f₀/2) t̂_{i,i+1}
    F_{i+1} += -(f₀/2) t̂_{i,i+1}

"""
function active_forces!(F::Matrix{Float64}, r::Matrix{Float64}, p::Params)
    N  = p.N
    f0 = p.f0
    n_bonds = (p.topology == :ring) ? N : N - 1

    @inbounds for b in 1:n_bonds
        j = (b == N) ? 1 : b + 1          # wrap-around for ring closing bond

        dx = r[1,j] - r[1,b]
        dy = r[2,j] - r[2,b]
        d  = sqrt(dx*dx + dy*dy)
        if d < 1e-14
            continue
        end

        # backward tangent: force points from bead j toward bead b (toward head)
        fx = -0.5 * f0 * dx / d
        fy = -0.5 * f0 * dy / d

        F[1,b] += fx;  F[2,b] += fy
        F[1,j] += fx;  F[2,j] += fy
    end
    return nothing
end

"""
Apply one overdamped Langevin step:
    ṙ = μ F_total + √(2μk_BT) ξ
Noise is disabled when kBT = 0.
"""
function step!(state::State, p::Params)
    r = state.r

    # Total forces
    F = bond_forces(r, p)
    bending_forces!(F, r, p)
    wca_forces!(F, r, p)
    active_forces!(F, r, p)

    # Overdamped Langevin update: dr = μF dt + √(2μk_BT dt) ξ
    noise_std = sqrt(2.0 * p.mu * p.kBT * p.dt)
    r .= r .+ p.dt .* p.mu .* F .+ noise_std .* randn(size(r))

    return state
end

end # module Engine
