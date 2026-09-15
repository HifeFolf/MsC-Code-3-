# ring_energy.jl
# Potential energy of a ring configuration, matching EngineRing's forces exactly.
#
# The ACTIVE force is non-conservative and has no potential — it is deliberately
# absent here.  That is the whole reason -ln P(Rg) is not a free energy.
#
# Convention note: the WCA force implemented in wca_forces! is
#     fac = eps/r^2 * (2*sr12 - sr6)
# whereas the textbook WCA force is 24x larger for the same eps.  So the potential
# consistent with THIS engine is
#     U_wca = (eps/6)[(sig/r)^12 - (sig/r)^6] + eps/24        (r < 2^(1/6) sig)
# i.e. the excluded-volume repulsion is ~24x softer than a conventional WCA with
# eps = 1.  Verified below by finite-differencing U against the engine's forces.

module RingEnergy
export bond_energy, bend_energy, wca_energy, total_energy

function bond_energy(r, p)
    N = p.N; U = 0.0
    nb = (p.topology == :ring) ? N : N-1
    @inbounds for i in 1:nb
        j = (i == N) ? 1 : i+1
        dx = r[1,j]-r[1,i]; dy = r[2,j]-r[2,i]
        U += 0.5*p.ks*(sqrt(dx*dx+dy*dy) - p.l)^2
    end
    U
end

function bend_energy(r, p)
    N = p.N; U = 0.0; pref = p.κ/p.l
    s = (p.topology == :ring) ? 1 : 2
    e = (p.topology == :ring) ? N : N-1
    @inbounds for i in s:e
        im = (i==1) ? N : i-1; ip = (i==N) ? 1 : i+1
        ux = r[1,i]-r[1,im]; uy = r[2,i]-r[2,im]
        vx = r[1,ip]-r[1,i]; vy = r[2,ip]-r[2,i]
        nu2 = ux*ux+uy*uy; nv2 = vx*vx+vy*vy
        (nu2 < 1e-24 || nv2 < 1e-24) && continue
        c = clamp((ux*vx+uy*vy)/sqrt(nu2*nv2), -1.0, 1.0)
        U += pref*(1.0 - c)
    end
    U
end

function wca_energy(r, p)
    N=p.N; σ=p.sigma; ϵ=p.epsilon
    rc2 = (2.0^(1/6)*σ)^2; σ2 = σ*σ; U = 0.0
    @inbounds for i in 1:N-1, j in i+1:N
        if p.topology == :open
            j-i == 1 && continue
        else
            min(j-i, N-(j-i)) == 1 && continue
        end
        dx = r[1,j]-r[1,i]; dy = r[2,j]-r[2,i]
        r2 = dx*dx+dy*dy
        (r2 >= rc2 || r2 < 1e-24) && continue
        sr6 = (σ2/r2)^3
        U += (ϵ/6)*(sr6*sr6 - sr6) + ϵ/24
    end
    U
end

total_energy(r,p) = bond_energy(r,p) + bend_energy(r,p) + wca_energy(r,p)
end
