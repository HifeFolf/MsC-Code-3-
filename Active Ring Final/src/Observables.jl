module Observables

export gyration_vals, contour_vals, inertia_vals, speed_vals, force_directions, compute_frame_obs

"""
    gyration_vals(r) -> NamedTuple

Compute the 2D gyration tensor and its scalar observables from a 2×N position matrix.

Returns a NamedTuple with fields:
- Gxx, Gyy, Gxy  : gyration tensor components
- λ1, λ2         : eigenvalues (λ1 ≥ λ2)
- Rg             : radius of gyration = sqrt(λ1 + λ2)
- b              : asphericity = (λ1 - λ2)² / (λ1 + λ2)²  ∈ [0, 1]
"""
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

"""
    contour_vals(r; connector_idx) -> NamedTuple

Compute contour-based shape observables from a 2×N position matrix (closed ring).

Steps:
1. Bond vectors along contour: bond j connects bead j to bead j+1 (mod N).
2. Bond angles ψ_j = atan(dy, dx).
3. Wrapped turning angles dψ_j = wrap(ψ_{j+1} - ψ_j) ∈ (-π, π].
4. Kloc = sum(dψ^4) / sum(dψ^2)^2   (0 if denominator ≈ 0).
5. Contour Fourier coefficients:
       c_m = (1/N) Σ_j dψ_j exp(-2πi m (j-1)/N)
   A1 = |c_1|,  A2 = |c_2|,  A3 = |c_3|,  A4 = |c_4|.
6. Kconn = signed turning angle at bead connector_idx (the passive junction).
   dψ[j] is the turning angle AT bead j+1, so the turning at bead b is dψ[mod1(b-1, N)].
   Positive = CCW (outward buckling); negative = CW (inward buckling).

The turning-angle sequence dψ sums to ≈ 2π for a simply-wound ring, so c_0 is
the mean curvature per bead (non-zero) while c_1…c_4 are shape deformation modes:
  A1 — polar / tear-drop    (one-lobe)
  A2 — elliptical           (two-lobe)
  A3 — triangular           (three-lobe)
  A4 — square               (four-lobe)
  A5 — pentagonal           (five-lobe)
  A6 — hexagonal            (six-lobe)

connector_idx defaults to the antipolar passive bead: (N-1)÷2 + 1.

Returns (A1, A2, A3, A4, A5, A6, Kloc, Kconn).
"""
function contour_vals(r::Matrix{Float64};
                      connector_idx::Int = (size(r, 2) - 1) ÷ 2 + 1)
    N = size(r, 2)

    # bond angles ψ_j for bond j → (j mod N)+1
    ψ = Vector{Float64}(undef, N)
    for j in 1:N
        jp = (j == N) ? 1 : j + 1
        ψ[j] = atan(r[2, jp] - r[2, j], r[1, jp] - r[1, j])
    end

    # wrapped turning angles
    dψ = Vector{Float64}(undef, N)
    for j in 1:N
        jp = (j == N) ? 1 : j + 1
        d  = ψ[jp] - ψ[j]
        d -= 2π * round(d / (2π))   # wrap to (-π, π]
        dψ[j] = d
    end

    # Kloc
    s2   = sum(x^2 for x in dψ)
    s4   = sum(x^4 for x in dψ)
    Kloc = s2^2 > 1e-30 ? s4 / s2^2 : 0.0

    # Contour Fourier amplitudes (index j runs 0-based inside the exponent)
    c1 = sum(dψ[j] * exp(-2π * im * 1 * (j - 1) / N) for j in 1:N) / N
    c2 = sum(dψ[j] * exp(-2π * im * 2 * (j - 1) / N) for j in 1:N) / N
    c3 = sum(dψ[j] * exp(-2π * im * 3 * (j - 1) / N) for j in 1:N) / N
    c4 = sum(dψ[j] * exp(-2π * im * 4 * (j - 1) / N) for j in 1:N) / N
    c5 = sum(dψ[j] * exp(-2π * im * 5 * (j - 1) / N) for j in 1:N) / N
    c6 = sum(dψ[j] * exp(-2π * im * 6 * (j - 1) / N) for j in 1:N) / N
    A1 = abs(c1)
    A2 = abs(c2)
    A3 = abs(c3)
    A4 = abs(c4)
    A5 = abs(c5)
    A6 = abs(c6)

    # Signed turning angle at the connector (passive junction bead).
    # dψ[j] is the turning AT bead j+1, so turning at bead b = dψ[mod1(b-1, N)].
    Kconn = dψ[mod1(connector_idx - 1, N)]

    return (A1 = A1, A2 = A2, A3 = A3, A4 = A4, A5 = A5, A6 = A6,
            Kloc = Kloc, Kconn = Kconn)
end

"""
    position_vals(r) -> NamedTuple

Compute Fourier amplitudes of the centered bead positions.

Input: zⱼ = (xⱼ − x_CM) + i(yⱼ − y_CM)   (complex centered position of bead j)

Fourier coefficients:
    p_m = (1/N) Σ_j zⱼ exp(-2πi m (j-1)/N)
    Pₘ = |p_m|

For a perfect circle with N equally-spaced beads:
    P1 = R   (the radius — always non-zero for any ring)
    Pm = 0   for m ≥ 2

Deformations from a circle appear as P2 (elliptical), P3 (triangular), P4 (square),
P5 (pentagonal), P6 (hexagonal) on top of the always-present P1 background.

Returns (P1, P2, P3, P4, P5, P6).
"""
function position_vals(r::Matrix{Float64})
    N   = size(r, 2)
    xcm = sum(r[1, :]) / N
    ycm = sum(r[2, :]) / N

    # centered complex positions
    z = [(r[1, j] - xcm) + im * (r[2, j] - ycm) for j in 1:N]

    p1 = sum(z[j] * exp(-2π * im * 1 * (j - 1) / N) for j in 1:N) / N
    p2 = sum(z[j] * exp(-2π * im * 2 * (j - 1) / N) for j in 1:N) / N
    p3 = sum(z[j] * exp(-2π * im * 3 * (j - 1) / N) for j in 1:N) / N
    p4 = sum(z[j] * exp(-2π * im * 4 * (j - 1) / N) for j in 1:N) / N
    p5 = sum(z[j] * exp(-2π * im * 5 * (j - 1) / N) for j in 1:N) / N
    p6 = sum(z[j] * exp(-2π * im * 6 * (j - 1) / N) for j in 1:N) / N

    return (P1 = abs(p1), P2 = abs(p2), P3 = abs(p3), P4 = abs(p4),
            P5 = abs(p5), P6 = abs(p6))
end

"""
    inertia_vals(r) -> NamedTuple

Compute the 2D planar inertia tensor for unit-mass beads from a 2×N position matrix.

Tensor components (standard mechanics convention, z-axis out of plane):
  Ixx = Σ (yᵢ − y_cm)²          (moment about x-axis)
  Iyy = Σ (xᵢ − x_cm)²          (moment about y-axis)
  Ixy = −Σ (xᵢ − x_cm)(yᵢ − y_cm)   (product of inertia)
  Izz = Ixx + Iyy                (polar moment, perpendicular-axis theorem)

Principal moments I1 ≥ I2 are eigenvalues of [[Ixx, Ixy],[Ixy, Iyy]].

Note: these relate to the gyration tensor G by Ixx = N·Gyy, Iyy = N·Gxx, Ixy = −N·Gxy.
Returns (Ixx, Iyy, Ixy, Izz, I1, I2).
"""
function inertia_vals(r::Matrix{Float64})
    N   = size(r, 2)
    xcm = sum(r[1, :]) / N
    ycm = sum(r[2, :]) / N
    Ixx = sum((r[2, i] - ycm)^2 for i in 1:N)
    Iyy = sum((r[1, i] - xcm)^2 for i in 1:N)
    Ixy = -sum((r[1, i] - xcm) * (r[2, i] - ycm) for i in 1:N)
    Izz = Ixx + Iyy
    mid  = (Ixx + Iyy) / 2.0
    half = sqrt(((Ixx - Iyy) / 2.0)^2 + Ixy^2)
    I1   = mid + half
    I2   = mid - half
    return (Ixx = Ixx, Iyy = Iyy, Ixy = Ixy, Izz = Izz, I1 = I1, I2 = I2)
end

"""
    speed_vals(r_curr, r_prev, dt_frame) -> NamedTuple

Compute per-frame speed observables from two consecutive 2×N position matrices
and the inter-frame time interval dt_frame = substeps × dt.

  v_mean : mean instantaneous bead speed  (1/N) Σ |Δrᵢ| / dt_frame
  v_max  : maximum instantaneous bead speed
  v_com  : speed of the center of mass   |Δr_cm| / dt_frame

Returns (v_mean, v_max, v_com).
"""
function speed_vals(r_curr::Matrix{Float64}, r_prev::Matrix{Float64},
                    dt_frame::Float64)
    N     = size(r_curr, 2)
    v_sum = 0.0
    v_max = 0.0
    for i in 1:N
        dx = r_curr[1, i] - r_prev[1, i]
        dy = r_curr[2, i] - r_prev[2, i]
        v  = sqrt(dx*dx + dy*dy) / dt_frame
        v_sum += v
        v_max  = max(v_max, v)
    end
    xcm_c = sum(r_curr[1, :]) / N;  ycm_c = sum(r_curr[2, :]) / N
    xcm_p = sum(r_prev[1, :]) / N;  ycm_p = sum(r_prev[2, :]) / N
    v_com = sqrt((xcm_c - xcm_p)^2 + (ycm_c - ycm_p)^2) / dt_frame
    return (v_mean = v_sum / N, v_max = v_max, v_com = v_com)
end

"""
    force_directions(r, bt, N) -> (tx, ty)

Compute the unit propulsion tangent vector for each bead.
For :active beads the tangent points along the averaged bond direction.
For :active_rev beads the tangent is reversed.
For :passive beads the tangent is zero.

Uses wrap-around neighbours.
"""
function force_directions(r::Matrix{Float64}, bt::Vector{Symbol}, N::Int)
    tx = zeros(N)
    ty = zeros(N)
    for i in 1:N
        bt[i] == :passive && continue   # passive bead: no propulsion, leave tx/ty = 0
        im = (i == 1) ? N : i - 1
        ip = (i == N) ? 1 : i + 1
        dx1 = r[1, ip] - r[1, i];  dy1 = r[2, ip] - r[2, i]
        n1  = sqrt(dx1*dx1 + dy1*dy1)
        dx2 = r[1, i]  - r[1, im]; dy2 = r[2, i]  - r[2, im]
        n2  = sqrt(dx2*dx2 + dy2*dy2)
        dx  = dx1 / max(n1, 1e-14) + dx2 / max(n2, 1e-14)
        dy  = dy1 / max(n1, 1e-14) + dy2 / max(n2, 1e-14)
        n   = sqrt(dx*dx + dy*dy)
        s   = (bt[i] == :active_rev) ? -1.0 : 1.0
        tx[i] = s * dx / max(n, 1e-14)
        ty[i] = s * dy / max(n, 1e-14)
    end
    return tx, ty
end

"""
    compute_frame_obs(r; connector_idx) -> NamedTuple

Compute all per-frame scalar observables from a 2×N position matrix.
Returns the merged result of gyration_vals, contour_vals, position_vals, and inertia_vals:
  Gxx, Gyy, Gxy, λ1, λ2, Rg, b,
  A1, A2, A3, A4, A5, A6, Kloc, Kconn,
  P1, P2, P3, P4, P5, P6,
  Ixx, Iyy, Ixy, Izz, I1, I2

Speed observables (v_mean, v_max, v_com) require two frames and are computed
separately via speed_vals.

connector_idx: 1-based index of the passive junction bead (default: (N-1)÷2+1).
"""
function compute_frame_obs(r::Matrix{Float64};
                           connector_idx::Int = (size(r, 2) - 1) ÷ 2 + 1)
    gv = gyration_vals(r)
    cv = contour_vals(r; connector_idx = connector_idx)
    pv = position_vals(r)
    iv = inertia_vals(r)
    return merge(gv, cv, pv, iv)
end

end # module Observables
