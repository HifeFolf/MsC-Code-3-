# Project Summary: Active Anti-Polar Ring Simulation (Julia)

## 1. What this project is

This is a **2D overdamped Brownian-dynamics simulation of an "active anti-polar ring"** — a closed chain of beads connected by springs, where beads carry self-propulsion forces. It is a computational soft-matter / active-matter physics research code written in **Julia**. The central scientific question is: **how does an elastic ring buckle, rotate, oscillate, or drift as a function of its activity (propulsion strength `f₀`), bending stiffness `κ`, and temperature `kBT`?** The key dimensionless control parameter is the **flexure number `Fl = f₀ / κ`** (activity-to-bending ratio).

The "anti-polar" design: half the ring's beads push **counter-clockwise** (`:active`) and the other half push **clockwise** (`:active_rev`). The two active arcs meet at junctions. For odd `N` there is a single **`:passive`** bead at the "front" junction (where the two active chains converge) — this passive junction bead is the main object of study (it's where the ring buckles inward/outward).

## 2. Repository layout

The repo (`New folder/`) contains **three sub-projects**. Only the first is current; the other two are earlier/legacy work.

```
Active Ring Final/        ← THE MAIN, CURRENT PROJECT
├── Project.toml          deps: CairoMakie, GLMakie, JLD2, JSON3, TOML
├── src/                  reusable modules (the "library")
│   ├── EngineRing.jl         physics engine (forces + Langevin integrator)
│   ├── InitialConditions.jl  bead-type layouts + circle/kinked geometries
│   ├── Observables.jl        per-frame shape descriptors
│   ├── MotionAnalysis.jl     motion observables + motion classifier
│   └── IOUtils.jl            save/load runs (TOML/JSON/JLD2)
├── configs/              TOML config files (one per experiment/sweep)
├── scripts/              ~35 driver + analysis + plotting scripts
└── results/              output run directories + figures (large)

Chain/                    LEGACY: open/clamped active filament ("chain") work
Rings/                    LEGACY: earlier ring simulations (+ box/pylon variant)
```

The code style is **module + scripts**: shared physics lives in `src/` modules; each script in `scripts/` is a standalone entry point run via `julia --project scripts/<name>.jl [args]`. Scripts `include()` the `src/` modules directly (they are NOT a registered package; modules live under `Main`).

## 3. The physics engine — `src/EngineRing.jl`

This is the heart of everything. It defines:

**`Params`** (a `@kwdef struct`) — all physical/numerical parameters:
- `N` = number of beads (default 16; **17** is the standard production value)
- `l` = rest bond length (1.0)
- `ks` = bond spring stiffness (**4444.0** — very stiff, makes bonds ~inextensible)
- `κ` (`kappa`) = bending stiffness
- `theta_max`, `theta_buf`, `k_stop` = optional log-wall "hinge stop" barrier that prevents bond angles exceeding `theta_max` (disabled by default, `k_stop=0`)
- `sigma`, `epsilon` = WCA (excluded-volume) bead diameter and strength
- `topology` = `:ring` (closed, default) or `:open` (filament)
- `f0` = active propulsion force magnitude
- `mu` = mobility μ = 1/γ (1.0)
- `kBT` = thermal energy (0 disables noise)
- `dt` = time step (5e-5)

**`State`** — mutable: `r` is a `2×N` matrix (each column is one bead's 2D position), and `bead_types` is a length-`N` `Vector{Symbol}` (`:active`, `:active_rev`, or `:passive`).

**Forces** (each is a separate function, summed every step):
1. `bond_forces` — Hookean springs: `U_bond = (ks/2) Σ(|r_{i+1}−r_i| − l)²`. For `:ring`, the closing bond N↔1 is included.
2. `bending_forces!` — bending energy `U_bend = (κ/l)(1 − cos θ)` at each interior vertex, plus the optional divergent log-wall barrier near `theta_max`. Computed analytically via the derivative of `cos θ` w.r.t. bead positions.
3. `wca_forces!` — Weeks–Chandler–Andersen purely-repulsive excluded volume between non-bonded bead pairs (cutoff at `2^(1/6)·σ`). Skips 1–2 bonded neighbors.
4. `active_forces!` — self-propulsion: `F_i = ±f₀ · t̂_i`, where `t̂_i` is the unit tangent (average of the two neighboring bond directions). `:active` → `+t̂`, `:active_rev` → `−t̂`, `:passive` → no force.

**Integrator** — `step!` applies one **overdamped Langevin (Euler–Maruyama) step**:
```
r ← r + dt·μ·F_total + sqrt(2·μ·kBT·dt)·ξ      (ξ = standard normal noise)
```
There is **no inertia** (overdamped). Noise is off when `kBT=0`.

## 4. Initial conditions — `src/InitialConditions.jl`

- `antipolar_bead_types(N)` builds the bead-type vector:
  - **Odd N (e.g. 17):** `(N−1)/2` `:active`, then 1 `:passive` (the front junction), then `(N−1)/2` `:active_rev`. → "passive junction ON".
  - **Even N (e.g. 16):** `N/2` `:active` + `N/2` `:active_rev`, no passive bead → "passive junction OFF".
- `perfect_circle_ic` places beads on a circle of radius `R = l/(2 sin(π/N))` so bonds are exactly at rest length.
- **Kinked ICs** (`kinked_circle_ic` … `quintuple_kinked_circle_ic`, k=1…5): start from the circle, then reflect the passive bead (and k−1 neighbors on each side) **inward** across neighbor chords. Reflection is an isometry so bond lengths stay exactly `l`. These create pre-buckled "V" shapes of increasing width to test basin-of-attraction / hysteresis. For k≥4 a two-phase scheme widens rather than deepens the kink (to avoid bead overlaps).
- `create_ic(layout, p, bead_types)` dispatches on a string name: `"antipolar"`, `"antipolar_kinked"`, `"antipolar_doublekinked"`, `"antipolar_triplekinked"`, `"antipolar_quadkinked"`, `"antipolar_quintkinked"`. A run can also start `"from_snapshot"` (load positions from a previous run's trajectory).

## 5. Observables — `src/Observables.jl`

Per-frame scalar shape descriptors computed from the `2×N` positions:
- **Gyration tensor** → `Rg` (radius of gyration), `b` (asphericity ∈[0,1], 0=circle, 1=line), eigenvalues `λ1,λ2`, components `Gxx,Gyy,Gxy`.
- **Contour Fourier amplitudes** `A1…A6`: FFT of the **turning-angle sequence** `dψ_j` around the ring. `A1`=polar/tear-drop, `A2`=elliptical, `A3`=triangular, `A4`=square, `A5`=pentagonal, `A6`=hexagonal.
- **`Kloc`** = `Σdψ⁴ / (Σdψ²)²` — curvature localization (how concentrated the bending is).
- **`Kconn`** = the **signed turning angle at the passive junction bead** — THE key order parameter. Positive = outward (CCW) buckling, negative = inward, near `2π/N` = circular. (See §10.)
- **Position Fourier amplitudes** `P1…P6`: FFT of centered complex positions (`P1`≈radius, `P2…` are shape deformations).
- **Inertia tensor** → `Ixx,Iyy,Ixy,Izz,I1,I2`.
- **Speeds** (`speed_vals`, needs two frames): `v_mean`, `v_max` (bead speeds), `v_com` (center-of-mass speed).
- `compute_frame_obs(r)` merges gyration+contour+position+inertia into one NamedTuple.

## 6. Data formats & I/O — `src/IOUtils.jl`

Each simulation run is one directory `results/<subfolder>/run_XXXX/` containing:
- **`params.toml`** — exact `Params` (under `[params]`) + run config (under `[run]`).
- **`metadata.json`** — run_id, seed, timestamp, Julia version, frame count, `L`, bead layout, sweep combo index, etc.
- **`traj.jld2`** — `times` (Vector), `positions` (`2×N×n_frames` Array), `bead_types` (Vector of strings). JLD2 = Julia's HDF5-based format.
- **`observables.jld2`** — `times` + one vector per observable key (`Rg`, `b`, `lambda1/2`, `Gxx/Gyy/Gxy`, `A1…A6`, `Kloc`, `Kconn`, `P1…P6`, `Ixx…I2`, `v_mean/v_max/v_com`).
- **`summary.png`** — auto-generated static 5-panel figure (Rg, b, A2, A3, Kloc vs time).
- Motion analysis adds: `motion_observables.jld2`, `motion_windows.jld2/.csv`, `motion_summary.json`.

`IOUtils` has matching `save_*`/`load_*` for every artifact. `init_run_dir` refuses to overwrite an existing run.

## 7. Run drivers

- **`scripts/run_single.jl [config.toml] [run_id]`** — runs one simulation from a config, saves all outputs, generates `summary.png`. The loop runs `substeps` physics steps per saved frame, applies **periodic wrapping** (shifts the whole ring back into a box `[−L/2, L/2]²` — a pure coordinate shift that doesn't affect forces; this is for visualization/COM tracking, not inter-bead PBC), and records observables.
- **`scripts/run_sweep.jl [sweep.toml] [--resume|--skip-existing]`** — runs a **parameter sweep**. A sweep config has `[sweep]` (pointing at a `base` config), `[sweep.run]` (output dir, frames, seed, layout), and `[sweep.params]` with either plain arrays `kappa=[1,2,5]` or **`linspace`/`logspace`** table syntax `logspace=[1, 50.0, 30]`. It takes the **Cartesian product** of all swept params, optionally crossed with multiple IC variants (`[[sweep.ics]]`). Runs are numbered from `start_id`; `--resume`/`--skip-existing` allow restarting interrupted sweeps. Writes `sweep_index.csv`. A typical production sweep is **30 κ × 50 f₀ = 1500 runs** (e.g. `sweep_N17_circle.toml`).

There are many config files representing different experiments: by N (16/17), by IC (circle, kinked, doublekinked…quintkinked), by seed (41/43/123/456), and large log-spaced sweeps.

## 8. Motion analysis — `src/MotionAnalysis.jl` + `scripts/analyze_motion.jl`

Post-processes a saved trajectory to classify **what kind of motion** the ring exhibits:
- **Per-frame** (`compute_motion_frame_obs`): unwrapped center-of-mass path & COM speed (corrects ±L box-wrap jumps), principal-axis angle `θ_shape` and its rate `Ω_shape` (with a reliability flag based on asphericity `b`), **contour circulation** `(1/N)Σ vᵢ·t̂ᵢ` (internal "treadmilling" flow) split into the two active halves, and **shape similarity** vs frame 1.
- **Per-window** (`compute_motion_windows`): chunks the run into windows (default 150 frames) and computes means/stds + a label.
- **Classifier** (`classify_motion_window`): assigns `:oscillatory`, `:rotating`, `:locked`, `:drifting`, or `:arrested` based on tunable thresholds in `MOTION_THRESHOLDS`. `classify_motion_run` does majority vote → a run is `:switching` if no single label dominates.
- **Morphology classifier** (`classify_morphology`): majority-votes each run into `:circular`, `:curled`, `:elongated`, or `:triangular` using `(b, A2, A3)` thresholds/centroids.
- `analyze_motion.jl` writes 4 figures (morphology, rotation, activity, window-timeline).

## 9. Analysis & visualization scripts (catalogue)

All in `scripts/`, all CairoMakie (headless PNG) or GLMakie (interactive/MP4):

| Script | Purpose |
|---|---|
| `summarize_run.jl` | static 5-panel time-series PNG per run |
| `render_movie.jl` | replays `traj.jld2` → MP4 (ring animation + 5 time-series); no physics recomputed |
| `browse_runs.jl` / `browse_runs_log.jl` | interactive GLMakie browser over a sweep folder |
| `check_ic.jl` | diagnostic: prints bond lengths, angles, per-force-component magnitudes at t=0 for circle vs kinked ICs (verifies no spurious overlaps) |
| `phase_diagram*.jl` (many variants) | 2D phase diagrams over (Fl or f₀)×(κ or kBT), colored by `b`, COM speed, mode fractions, local curvature, etc. Variants: `_01`, `_log`, `_modes`, `_modes_pos`, `_mode_fractions(_pos)`, `_order`, `_overlay`, `_seeds`, `_local_curvature`, `_passive_angle`, `_kinked_comparison`, `_collapsed` |
| `phase_classification.jl` | **the principled phase map** (see §10) |
| `transition_plot.jl` | `b` and COM speed vs `f₀/κ` (continuous transition curves) |
| `bifurcation_diagram.jl` | 1-D bifurcation: ⟨Kconn⟩ vs Φ=f₀/κ, plus A2–A6; supports fixed-κ or fixed-f₀ slices |
| `linear_stability.jl` | **linear stability analysis**: builds Jacobian `J = μ[H_bond + κ·H_bend + f₀·H_act]` by finite differences (H_act is non-symmetric → can give Hopf bifurcations); finds instability threshold f₀_c(κ), critical mode shape, and checks `Fl=f₀/κ` scaling |
| `fl_origin.jl` | exact derivation of critical flexure number `Fl_c = −1/λ_min(B⁻¹H_act)/κ`; shows it's ~constant when bonds are stiff |
| `normal_modes.jl` | normal-mode (Hessian eigen) analysis of the **passive** ring; filters 3 rigid-body zero modes |
| `pca_dynamics.jl`, `pca_structures.jl`, `pca_isele_holder.jl` | **PCA on curvature fluctuations**, adapting Isele-Holder et al. (Soft Matter 2016, 12, 8495). Uses a χ² deviation-from-Rayleigh test of `p(λ₁,λ₂)` as a buckling criterion; classifies runs as circular/kinked/dynamic |
| `hysteresis.jl` / `hysteresis_movie.jl` | ramps `f₀` up then down **without reinitializing state** → traces a hysteresis loop in `b` vs `f₀` to prove **bistability** |
| `order_parameters.jl` | order-parameter analysis of a single run |

## 10. Phase classification method (important — established convention)

The project's preferred way to classify **static vs dynamic** phases is NOT by fluctuation amplitude (which is a featureless continuum), but by **persistence**: `phase_classification.jl` uses two axes —
- `amplitude = log10 std(Kconn)` (how much the junction moves), and
- `persistence = autocorrelation of Kconn at lag ≈1 s` (whether the motion is sustained: thermal jitter decorrelates fast → ~0; oscillation/rotation stays correlated → →1).

The **static↔dynamic** cut is an **Otsu threshold on the persistence distribution** (a real valley in the data). The **structural** split (inward / circular / outward buckling) uses **⟨Kconn⟩ vs 360/N**. This "classify by Kconn persistence/autocorrelation, not by fluctuation amplitude" is a deliberate design decision.

## 11. Legacy sub-projects (for context, not active)

- **`Chain/`** — earlier work on **open/clamped active filaments** ("chains"). `Simulation Chain 2/Engine.jl` is a near-identical engine but defaults to `:open` topology. Has a large `Ploting Chain/` library (MSD, diffusion, beating frequency, terminal velocity, limit cycles, phase diagrams, harmonic trap, WCA diagnostics, etc.).
- **`Rings/`** — earlier ring simulations, including a **box+pylons** variant (`Simulation Ring Box + Pylons/`) and anti-polar ring plotting/sweeps. The current `Active Ring Final/EngineRing.jl` is the cleaned-up descendant of `Rings/Simulation Ring/Engine_ring.jl` (which had `ks=2000`, `kBT=1` defaults vs the production `ks=4444`, `kBT=0.1`).

## 12. Key physics facts to carry over

- **Units are reduced/Brownian**: `l=σ=μ=1`. Bonds are stiff (`ks=4444`) → effectively inextensible.
- **Control parameters**: `Fl = f₀/κ` (primary), with `kBT` as a secondary (noise) axis. Stability boundary is roughly `f₀_c ≈ 75κ` (so `Fl_c ≈ 75`) at the default geometry — picked so hysteresis sweeps `f₀ ∈ [1,200]` cross it.
- **N parity matters**: odd N has a passive junction bead (symmetry-broken front/back); even N is symmetric. N=17 is standard production.
- **Order parameter**: `Kconn` (signed turning angle at the passive bead) for buckling direction; `b`, `A2`, `Rg` for morphology; circulation/Ω for dynamics.
- **Phases observed**: static (circular / inward-kinked / outward-kinked) vs dynamic (oscillating, rotating, drifting, locked-circulation, switching/chaotic).
- Typical run: `N=17`, `1500 frames × 100 substeps`, `dt=5e-5`, `framerate=60`, seed `42`.

## 13. How to run

```bash
cd "Active Ring Final"
julia --project scripts/run_single.jl configs/default.toml        # one run
julia --project scripts/run_sweep.jl  configs/sweep_N17_circle.toml --resume   # a sweep
julia --project scripts/analyze_motion.jl results/Results_N17_circle/run_0042
julia --project scripts/phase_classification.jl                   # phase map over a sweep
julia --project scripts/render_movie.jl results/Results_01/run_5089
```
