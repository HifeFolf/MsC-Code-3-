module Observables

export gyration_vals, contour_vals, force_directions, compute_frame_obs

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
    contour_vals(r) -> NamedTuple

Compute contour-based shape observables from a 2×N position matrix (closed ring).

Steps:
1. Bond vectors along contour: bond j connects bead j to bead j+1 (mod N).
2. Bond angles ψ_j = atan(dy, dx).
3. Wrapped turning angles dψ_j = wrap(ψ_{j+1} - ψ_j) ∈ (-π, π].
4. Kloc = sum(dψ^4) / sum(dψ^2)^2   (0 if denominator ≈ 0).
5. Contour Fourier coefficients:
       c_m = (1/N) Σ_j dψ_j exp(-2πi m (j-1)/N)
   A2 = |c_2|,  A3 = |c_3|.

Returns (A2, A3, Kloc).
"""
function contour_vals(r::Matrix{Float64})
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
    c2 = sum(dψ[j] * exp(-2π * im * 2 * (j - 1) / N) for j in 1:N) / N
    c3 = sum(dψ[j] * exp(-2π * im * 3 * (j - 1) / N) for j in 1:N) / N
    A2 = abs(c2)
    A3 = abs(c3)

    return (A2 = A2, A3 = A3, Kloc = Kloc)
end

"""
    force_directions(r, bt, N) -> (tx, ty)

Compute the unit propulsion tangent vector for each bead.
For :active beads the tangent points along the averaged bond direction.
For :active_rev beads the tangent is reversed.
For :passive beads the tangent is zero.

Uses wrap-around neighbours (ring topology assumed).
"""
function force_directions(r::Matrix{Float64}, bt::Vector{Symbol}, N::Int)
    tx = zeros(N)
    ty = zeros(N)
    for i in 1:N
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
    compute_frame_obs(r) -> NamedTuple

Compute all per-frame scalar observables from a 2×N position matrix.
Returns the merged result of gyration_vals and contour_vals:
  Gxx, Gyy, Gxy, λ1, λ2, Rg, b, A2, A3, Kloc
"""
function compute_frame_obs(r::Matrix{Float64})
    gv = gyration_vals(r)
    cv = contour_vals(r)
    return merge(gv, cv)
end

end # module Observables
