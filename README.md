# Active Anti-Polar Ring — MSc project

2-D overdamped Brownian-dynamics simulation of a closed ring of **N = 17** beads (Julia). Beads 1–8 self-propel
along +t̂, beads 10–17 along −t̂, and bead 9 is a passive **junction** that both driven arcs push towards.
The project maps the ring's states over bending stiffness κ and propulsion f₀. It also shows that the driven
steady state is **not** Boltzmann: the ring spends long times in states its own potential energy does not favour.

## Where to start (article writing)

| Read | For |
|---|---|
| [`LATEX_EXPORT.md`](LATEX_EXPORT.md) | **Canonical, most recent (2026-10-06).** Model, exact parameters, potentials, observables, state definition, every measured number, corrections, limitations, figure inventory + caption drafts, LaTeX notation, safe sentences, and **§13: what the data does NOT support**. |
| [`CODE_REFERENCE.md`](CODE_REFERENCE.md) | The code that matters, verbatim: engine (forces + integrator), parameters, bead layout, run loop, observables (Rg, Kconn), energy module, state tracker, Kramers–Moyal barrier. |
| [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | Repository layout and what every script does. |
| [`PRESENTATION_OUTLINE.md`](PRESENTATION_OUTLINE.md) | Talk structure. |
| [`AI_CONTEXT.md`](AI_CONTEXT.md) | Older context file. **§5–§8 are stale.** It predates the three-state result, the passive control and the kinetics, so prefer `LATEX_EXPORT.md` wherever they disagree. |

## Key figures (committed)

These are under `Active Ring Final/results/`, PNG plus a vector PDF where one exists. `LATEX_EXPORT.md` §9
has the full table and caption drafts.

| Figure | Shows |
|---|---|
| `campaign/model_schematic` | the 17-bead ring, drive directions, the junction |
| `campaign/showcase_nonequilibrium` | **headline**: energy vs occupancy, passive vs active trace, Boltzmann vs measured |
| `campaign/traces_simple` | Rg(t), one passive and one active run |
| `Results_N17_phi_long/phase_diagram_regimes` | regime map over (κ, f₀) |
| `Results_N17_phi_long/phase_diagram_confidence` | same, shaded by confidence |
| `Results_N17_phi_long/phase_reliability_route` | seed agreement and neighbour coherence |
| `Results_N17_bistable_long/tristate` | three states (open / fold-out / fold-in), transition matrix, survival |
| `Results_N17_bistable_long/fold_shapes` | snapshots of the three states with active-force arrows |
| `Results_N17_bistable_long/{bistability_evidence, dwell_analysis, seed_ensemble}` | switching evidence, dwell times, seed-to-seed variation |
| `campaign/energy_landscape` | −ln P vs U_eff vs ⟨U\|Rg⟩ |
| `campaign/{well_cuts, cut_map}` | effective double wells along six cuts, and where they sit |
| `campaign/kramers_rates` | escape rates and the parameter-free Kramers test |
| `campaign/markov_check` | lag dependence, Chapman–Kolmogorov |
| `campaign/{validation_map, before_after}` | movie-validated cells; diagram before/after corrections |

The small CSVs behind the campaign figures (`campaign/*.csv`) are committed as well. Raw run data (trajectories,
the per-run JLD2 files) and movies are **not** in git because they are too large.

## Layout

```
Active Ring Final/        current project
  src/                    EngineRing (physics), Observables, InitialConditions, MotionAnalysis, IOUtils
  configs/                TOML configs, one per campaign
  scripts/                drivers, analysis and plotting (run: julia --project scripts/<name>.jl)
  results/                ignored except the figures and CSVs listed above
Chain/                    earlier work: open / clamped active filament (code + plots; movies not tracked)
Rings/                    earlier ring prototypes, incl. a box + pylons variant (code + plots; movies not tracked)
```

## Running

```
cd "Active Ring Final"
julia --project -e "using Pkg; Pkg.instantiate()"
julia --project scripts/run_single.jl configs/default.toml
```

Dependencies: CairoMakie, GLMakie, JLD2, JSON3, TOML (`Project.toml`).
