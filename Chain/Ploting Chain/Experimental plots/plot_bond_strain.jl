# plot_bond_strain.jl
# Bond inextensibility diagnostics: max and RMS bond strain vs time.
# All physics parameters come from Engine defaults.

using GLMakie
using LinearAlgebra

include(joinpath(@__DIR__, "..", "Simulation", "Engine.jl"))

function main()
    frames   = 500
    substeps = 100

    p = Engine.Params()       # single source of truth
    N = p.N
    l = p.l

    # --- initial zig-zag chain ---
    r0 = zeros(2, N)
    zig_angle = π/3
    for i in 2:N
        θ = iseven(i) ? zig_angle : -zig_angle
        r0[1, i] = r0[1, i-1] + l * cos(θ)
        r0[2, i] = r0[2, i-1] + l * sin(θ)
    end
    state = Engine.State(copy(r0))

    # --- storage ---
    times      = Float64[]
    max_strain = Float64[]
    rms_strain = Float64[]
    t = 0.0

    function record_strain!()
        strains = [norm(state.r[:, i+1] - state.r[:, i]) - l for i in 1:N-1] ./ l
        push!(max_strain, maximum(abs, strains))
        push!(rms_strain, sqrt(sum(s -> s*s, strains) / length(strains)))
        push!(times, t)
    end
    record_strain!()

    # --- run simulation ---
    for _ in 1:frames
        for _ in 1:substeps
            Base.invokelatest(Engine.step!, state, p)
            t += p.dt
        end
        record_strain!()
    end

    # --- plot ---
    fig = Figure(size=(800, 500))
    ax  = Axis(fig[1, 1];
        xlabel="Time",
        ylabel="Bond strain",
        title="Bond Inextensibility Diagnostics",
        yscale=log10,
    )

    lines!(ax, times, max_strain; label="max |εᵢ|", linewidth=2)
    lines!(ax, times, rms_strain; label="RMS εᵢ",   linewidth=2)
    axislegend(ax; position=:rt)

    display(fig)

    output_path = joinpath(@__DIR__, "bond_strain.png")
    save(output_path, fig)
    println("Saved plot to: ", output_path)
end

main()
