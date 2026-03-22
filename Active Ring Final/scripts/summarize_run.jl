# summarize_run.jl
# Generate a static 5-panel summary PNG from a saved run directory.
# Uses CairoMakie (headless — no display or window required).
#
# Usage (standalone, from the Active Ring Final/ project root):
#
#   julia --project scripts/summarize_run.jl results/run_0001
#
# Or called programmatically from run_single.jl:
#
#   include("scripts/summarize_run.jl")
#   summarize_run("results/run_0001")

using CairoMakie

const PROJECT_ROOT_SR = joinpath(@__DIR__, "..")

# Load IOUtils if not already loaded (standalone use)
if !isdefined(Main, :IOUtils)
    include(joinpath(PROJECT_ROOT_SR, "src", "IOUtils.jl"))
    using .IOUtils
end
if !isdefined(Main, :EngineRing)
    include(joinpath(PROJECT_ROOT_SR, "src", "EngineRing.jl"))
    using .EngineRing
end

"""
    summarize_run(run_dir)

Load observables from `run_dir/observables.jld2` and `run_dir/params.toml`,
then write a 5-panel static figure to `run_dir/summary.png`.
Panels: Rg, b  (top row);  A2, A3  (middle row);  Kloc  (bottom, full width).
"""
function summarize_run(run_dir::String)
    _, obs  = IOUtils.load_observables(run_dir)
    p, rc   = IOUtils.load_params(run_dir)
    meta    = IOUtils.load_metadata(run_dir)

    n       = length(obs["Rg"])
    fr      = Float64(rc["framerate"])
    vtimes  = [(i - 1) / fr for i in 1:n]   # video seconds, matches MP4 x-axis

    fig = Figure(size = (900, 900))

    title_str = "Run: $(meta["run_id"])   N=$(p.N)   f₀=$(p.f0)   κ=$(p.κ)   kBT=$(p.kBT)"
    Label(fig[0, 1:2], title_str; fontsize = 14, tellwidth = false)

    ax_Rg  = Axis(fig[1, 1]; ylabel = "Rg",             xlabel = "video time (s)")
    ax_b   = Axis(fig[1, 2]; ylabel = "b (asphericity)", xlabel = "video time (s)")
    ax_A2  = Axis(fig[2, 1]; ylabel = "A₂",             xlabel = "video time (s)")
    ax_A3  = Axis(fig[2, 2]; ylabel = "A₃",             xlabel = "video time (s)")
    ax_K   = Axis(fig[3, 1:2]; ylabel = "Kloc",         xlabel = "video time (s)")

    lines!(ax_Rg, vtimes, obs["Rg"];   color = :dodgerblue,  linewidth = 1.5)
    lines!(ax_b,  vtimes, obs["b"];    color = :green,        linewidth = 1.5)
    lines!(ax_A2, vtimes, obs["A2"];   color = :mediumpurple, linewidth = 1.5)
    lines!(ax_A3, vtimes, obs["A3"];   color = :darkorange,   linewidth = 1.5)
    lines!(ax_K,  vtimes, obs["Kloc"]; color = :teal,         linewidth = 1.5)

    save(joinpath(run_dir, "summary.png"), fig)
    return nothing
end

# ── Standalone entry point ────────────────────────────────────────────────────

# ▼▼▼  Edit this line to choose which run to summarise when using the Run button  ▼▼▼
const RUN_DIR_SR = joinpath(@__DIR__, "..", "results", "run_0006")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

if abspath(PROGRAM_FILE) == @__FILE__
    run_dir_sr = isempty(ARGS) ? RUN_DIR_SR : ARGS[1]
    summarize_run(run_dir_sr)
    println("Saved: ", joinpath(run_dir_sr, "summary.png"))
end
