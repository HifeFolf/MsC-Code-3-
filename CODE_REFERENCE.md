# Code reference — the parts of the code an article needs

The snippets below are copied verbatim from the source, with the file named above each one. Paths are relative
to `Active Ring Final/`. For the physics write-up and all measured numbers, see `LATEX_EXPORT.md`.

---

## 1. Model in one equation

Overdamped Langevin dynamics, integrated with Euler–Maruyama:

```
r_i(t+dt) = r_i(t) + μ F_i dt + sqrt(2 μ kBT dt) ξ_i ,      ξ_i ~ N(0, 1) per component
F_i = −∇_i (U_bond + U_bend + U_WCA) + F_i^act
```

| Term | Potential / force | Source function |
|---|---|---|
| Bond | U = (ks/2) Σ (\|r_{i+1} − r_i\| − l)², ring closed (bond N↔1 included) | `bond_forces` |
| Bending | U = (κ/l) Σ (1 − cos θ_i), θ_i = turning angle at bead i | `bending_forces!` |
| Excluded volume | WCA, cut at 2^(1/6) σ, all pairs except bonded neighbours | `wca_forces!` |
| Active | F_i = ±f₀ t̂_i, where t̂_i is the normalised sum of the two adjacent bond unit vectors. **No potential.** | `active_forces!` |

**WCA convention.** The engine's force prefactor is `ε/r² (2 sr¹² − sr⁶)`, which is 24× softer than textbook
WCA at the same ε. The potential consistent with it is `U = (ε/6)[(σ/r)¹² − (σ/r)⁶] + ε/24`. This is a property
of the model, not a bug. The energy module in §7 matches the engine forces.

---

## 2. Parameters — `src/EngineRing.jl`

```julia
Base.@kwdef struct Params
    N::Int         = 16
    l::Float64     = 1.0
    ks::Float64    = 4444.0
    κ::Float64     = 10.0
    theta_max::Float64 = π
    theta_buf::Float64 = 0.1
    k_stop::Float64    = 0.0
    sigma::Float64     = 1.0
    epsilon::Float64   = 1.0
    topology::Symbol   = :ring
    f0::Float64    = 20.0
    mu::Float64    = 1.0
    kBT::Float64   = 0.1
    dt::Float64    = 0.00005
end

mutable struct State
    r::Matrix{Float64}          # 2×N, column i = position of bead i
    bead_types::Vector{Symbol}  # :active | :active_rev | :passive
end
```

These are code defaults only. Every campaign sets them from a TOML config. **Values used in all published
campaigns:**

| Symbol | Code field | Value | Meaning |
|---|---|---|---|
| N | `N` | 17 | beads (8 active + 1 passive junction + 8 active_rev) |
| l | `l` | 1.0 | bond rest length (length unit) |
| k_s | `ks` | 4444.0 | bond stiffness (nearly inextensible) |
| κ | `κ` (`kappa` in TOML) | **swept** | bending stiffness |
| f₀ | `f0` | **swept** | propulsion force per active bead |
| k_BT | `kBT` | 0.1 | temperature |
| μ | `mu` | 1.0 | mobility (1/γ) |
| dt | `dt` | 5×10⁻⁵ | integration time step |
| σ, ε | `sigma`, `epsilon` | 1.0, 1.0 | WCA diameter and strength |
| θ_max, θ_buf, k_stop | | π, 0.1, 0 | optional hinge-stop wall, **disabled** (k_stop = 0) |

Sweep ranges per campaign are in `LATEX_EXPORT.md` §1.2 and §5.

---

## 3. Forces and integrator — `src/EngineRing.jl`

### Bond springs
```julia
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
```

### Bending, U = (κ/l)(1 − cos θ)
The derivative of cos θ with respect to the two bond vectors u = r_i − r_{i−1} and v = r_{i+1} − r_i is
distributed onto beads i−1, i and i+1 (the log-wall branch is inactive because k_stop = 0).
```julia
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
```

### WCA excluded volume (all non-bonded pairs)
```julia
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
            fac    = ϵ * inv_r2 * (2.0*sr6*sr6 - sr6)

            F[1,i] -= fac*dx;  F[2,i] -= fac*dy
            F[1,j] += fac*dx;  F[2,j] += fac*dy
        end
    end
    return nothing
end
```

### Active propulsion, F = ±f₀ t̂ (ring branch)
The local tangent at bead i is the normalised sum of its two adjacent bond unit vectors. `:active` beads push
along +t̂ and `:active_rev` along −t̂. The `:passive` junction gets no active force.
```julia
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
        # (… :open-topology end-bead branches omitted …)
        end

        n = sqrt(dx*dx + dy*dy)
        n < 1e-14 && continue
        F[1,i] += sign * f0 * dx / n
        F[2,i] += sign * f0 * dy / n
    end
```

### One time step (Euler–Maruyama)
```julia
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
```

---

## 4. Bead layout and initial condition — `src/InitialConditions.jl`

```julia
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
```
For N = 17 this gives beads 1–8 `:active`, bead 9 `:passive` and beads 10–17 `:active_rev`. Both arcs push
towards bead 9, so the junction is permanently under compression.

Default initial condition: a perfect circle with exact bond lengths, R = l / (2 sin(π/N)) (`perfect_circle_ic`).
The code also has kinked-circle initial conditions (`_kinked_circle_ic`), used by the `sweep_N17_*kinked*`
configs, and restarts from a snapshot (`bead_layout = "from_snapshot"`).

---

## 5. Run loop — `scripts/run_single.jl`

A saved frame is `substeps` engine steps, so frame spacing = substeps × dt. The 300 s campaigns use 1000
substeps, which gives 0.05 s per frame. After each frame the whole ring is rigidly shifted by L if it has left
the box. This is a **recentring only**: there are no periodic interactions. Unwrap the COM by detecting jumps
larger than L/2.

```julia
dt_frame = substeps * p.dt     # simulation time per saved frame

for frame in 1:frames
    for _ in 1:substeps
        EngineRing.step!(state, p)
    end
    global t_sim += dt_frame

    hw = L / 2.0
    min_x = minimum(@view state.r[1, :]);  max_x = maximum(@view state.r[1, :])
    min_y = minimum(@view state.r[2, :]);  max_y = maximum(@view state.r[2, :])
    min_x >  hw && (state.r[1, :] .-= L)
    max_x < -hw && (state.r[1, :] .+= L)
    min_y >  hw && (state.r[2, :] .-= L)
    max_y < -hw && (state.r[2, :] .+= L)

    times[frame]           = t_sim
    positions[:, :, frame] = state.r
    obs = Observables.compute_frame_obs(state.r)     # Rg, b, Kconn, A1…A6, P1…P6 …
    # … stored per frame …
end
```

Each run writes `params.toml`, `metadata.json` (seed, timing), `traj.jld2` (2 × N × frames positions, only for
some campaigns) and `observables.jld2` (time series) into `results/<campaign>/run_XXXX/`. Campaign sweeps are
driven by `scripts/run_sweep.jl` and `scripts/run_targeted_seeds.jl`.

---

## 6. Observables — `src/Observables.jl`

### Radius of gyration and asphericity
```julia
function gyration_vals(r::Matrix{Float64})
    N   = size(r, 2)
    xcm = sum(r[1, :]) / N
    ycm = sum(r[2, :]) / N
    Gxx = sum((r[1, i] - xcm)^2 for i in 1:N) / N
    Gyy = sum((r[2, i] - ycm)^2 for i in 1:N) / N
    Gxy = sum((r[1, i] - xcm) * (r[2, i] - ycm) for i in 1:N) / N
    mid  = (Gxx + Gyy) / 2.0
    half = sqrt(((Gxx - Gyy) / 2.0)^2 + Gxy^2)
    λ1   = mid + half
    λ2   = mid - half
    rg2  = λ1 + λ2
    b    = rg2 > 1e-30 ? (λ1 - λ2)^2 / rg2^2 : 0.0
    return (Gxx = Gxx, Gyy = Gyy, Gxy = Gxy,
            λ1 = λ1, λ2 = λ2,
            Rg = sqrt(max(rg2, 0.0)), b = b)
end
```
Perfect circle: Rg = 1/(2 sin(π/17)) ≈ 2.72. Folded: Rg ≈ 1.5.

### Kconn — signed turning angle at the junction (excerpt of `contour_vals`)
```julia
    # bond angles ψ_j for bond j → (j mod N)+1
    for j in 1:N
        jp = (j == N) ? 1 : j + 1
        ψ[j] = atan(r[2, jp] - r[2, j], r[1, jp] - r[1, j])
    end
    # wrapped turning angles
    for j in 1:N
        jp = (j == N) ? 1 : j + 1
        d  = ψ[jp] - ψ[j]
        d -= 2π * round(d / (2π))   # wrap to (-π, π]
        dψ[j] = d
    end
    # dψ[j] is the turning AT bead j+1, so turning at bead b = dψ[mod1(b-1, N)].
    Kconn = dψ[mod1(connector_idx - 1, N)]      # connector_idx = 9 for N = 17
```
The turning angles of a closed curve sum to +2π, so a perfect circle has Kconn = 360°/17 ≈ +21.18°.
**Kconn > 0**: convex, the junction sits at the tip of a hairpin (fold-out). **Kconn < 0**: concave, the
junction is pushed into the ring (fold-in). Rg cannot tell these apart.

`contour_vals` also returns the contour Fourier amplitudes A1…A6 of dψ and the kurtosis-like localisation
`Kloc = Σdψ⁴ / (Σdψ²)²`. `position_vals` returns the Fourier amplitudes P1…P6 of the centred positions.

---

## 7. Potential energy — `scripts/ring_energy.jl`

This matches the engine's forces exactly (checked by finite differences). The active force is deliberately absent
because it has no potential.
```julia
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
```
Divide by kBT = 0.1 to express in units of kBT.

---

## 8. State definition — hysteretic tracker

Every analysis uses this tracker, so all numbers are mutually comparable.

| Constant | Value |
|---|---|
| reference size `thr` | 0.70 · N/(2π) = 1.894 |
| folded | Rg < 0.9·thr = 1.705 |
| open | Rg > 1.1·thr = 2.083 |
| in between | state unchanged (deadband) |
| minimum dwell | 0.5 s before a switch counts |
| burn-in | first 5.0 s discarded |
| fold-out / fold-in (folded frames only) | Kconn > +30° / Kconn < −30° |

Two-state version, from `scripts/analyze_kramers_rates.jl`:
```julia
function segment(t, Rg, thr, need)
    segs = Tuple{Symbol,Float64,Float64}[]
    state = Rg[1] < thr ? :folded : :open
    t_enter = t[1]; rl = 0; pend = :none
    for (j, x) in enumerate(Rg)
        s = x < 0.9thr ? :folded : (x > 1.1thr ? :open : :none)
        (s == :none || s == state) && (rl = 0; pend = :none; continue)
        (s == pend) ? (rl += 1) : (pend = s; rl = 1)
        if rl >= need
            push!(segs, (state, t_enter, t[j]))
            state = s; t_enter = t[j]; rl = 0; pend = :none
        end
    end
    push!(segs, (state, t_enter, t[end]))
    segs
end
```
The three-state labelling, from `scripts/analyze_tristate.jl`, applies the same debounce to:
```julia
raw(j) = Rg[j] > 1.1thr ? :O :
         (Rg[j] < 0.9thr ? (K[j] > K0 ? :Fo : (K[j] < -K0 ? :Fi : :none)) : :none)   # K0 = 30°
```

**Escape rate** (censored-exponential MLE): k = completed escapes / total time in the state. The first and
last segments of each run add exposure time but are never counted as events.

---

## 9. Kramers–Moyal effective potential and the Kramers test — `scripts/analyze_kramers_rates.jl`

```
D₁(x) = ⟨ΔRg⟩/τ ,   D₂(x) = ⟨ΔRg²⟩/(2τ) ,   U_eff(x) = −∫ D₁/D₂ dx ,   τ = frame spacing (0.05 s)
k_Kramers = (D₂(top)/2π) √(|U″(well)| |U″(top)|) exp(−ΔU)          (no fitted parameter)
```
```julia
    D1 = (s1[best] ./ nb[best]) ./ c.dt
    D2 = (s2[best] ./ nb[best]) ./ (2*c.dt)
    integ = -D1 ./ D2
    U = zeros(length(x))
    for i in 2:length(x); U[i] = U[i-1] + 0.5*(integ[i]+integ[i-1])*(x[i]-x[i-1]); end
    # … locate barrier top bt and the two wells li, ri …
    function curv(i)                       # least-squares parabola over ±2 bins
        lo = max(1,i-2); hi = min(length(x),i+2)
        hi-lo < 2 && return NaN
        xs = x[lo:hi] .- x[i]
        coef = [xs.^2 xs ones(length(xs))] \ U[lo:hi]
        2*coef[1]
    end
    pre(cw) = (D2[bt]/(2π)) * sqrt(abs(cw*ct))
    kpred = (folded = pre(cw_f)*exp(-(U[bt]-U[li])), open = pre(cw_o)*exp(-(U[bt]-U[ri])))
```
Markov validity (lag dependence, Chapman–Kolmogorov) is checked in `scripts/markov_check.jl`.

---

## 10. Script → result map

| Script (`scripts/`) | Produces |
|---|---|
| `run_single.jl`, `run_sweep.jl`, `run_targeted_seeds.jl` | simulation runs |
| `reduce_phase_features.jl` | per-run features → `phase_features.csv` |
| `phase_reliability.jl` | reproducibility and coherence maps |
| `phase_diagram_regimes.jl`, `plot_regimes.jl`, `phase_diagram_confidence.jl` | regime map, confidence map |
| `analyze_dwell.jl` | dwell times, survival curves |
| `analyze_tristate.jl` | three-state analysis |
| `explain_fold_asymmetry.jl` | per-state energies, contacts, force projections, snapshots |
| `ring_energy.jl` | the energy module |
| `analyze_energy_landscape.jl` | −ln P vs U_eff vs ⟨U\|Rg⟩ |
| `analyze_kramers_rates.jl` | escape rates, Kramers test |
| `markov_check.jl` | Markov / Chapman–Kolmogorov check |
| `showcase_nonequilibrium.jl` | headline figure |
| `plot_traces_simple.jl` | single-panel trace figure |
| `plot_well_cuts.jl`, `plot_cut_map.jl` | six fixed-parameter cuts and their map |
| `plot_model_schematic.jl` | model schematic |

`PROJECT_SUMMARY.md` §3 describes every remaining script.
