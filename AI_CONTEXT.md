# Active Anti-Polar Ring — AI Context Brief

Self-contained handoff for an AI assistant with no access to the repo. Written 2026-09-05.
Everything below is either measured or explicitly flagged as an assumption.

---

## 1. The system

2D **overdamped Brownian dynamics** of a closed elastic ring of `N` beads, in Julia.

The ring is **anti-polar**: half the beads push counter-clockwise (`:active`), half push
clockwise (`:active_rev`). Net torque is zero by construction, so — unlike a *polar* active
ring, which relieves its drive by treadmilling — this ring has nowhere to put the activity
except into **shape**. It is a deliberately frustrated system.

For **odd N** there is one extra `:passive` bead at the junction where the two active arcs
meet. This is the engineered weak point where frustration localises. **N = 17 is production**
(8 active + 1 passive + 8 active_rev). N = 16 (8+8, no passive bead) exists as the
junction-OFF control.

### Physics

Forces summed each step:
1. **Bonds** — Hookean, `U = (ks/2) Σ(|r_{i+1}−r_i| − l)²`, closing bond included for `:ring`.
2. **Bending** — `U = (κ/l)(1 − cos θ)` at each vertex; optional log-wall barrier (disabled by default, `k_stop = 0`).
3. **WCA** — purely repulsive excluded volume, cutoff `2^(1/6)σ`, skips bonded neighbours.
4. **Active** — `F_i = ±f₀ t̂_i`, `t̂` = unit tangent (mean of the two adjacent bond directions). `:passive` gets none.

Integrator — **Euler–Maruyama, overdamped, no inertia**:
`r ← r + dt·μ·F + sqrt(2·μ·kBT·dt)·ξ`

### Production parameters

| param | value | note |
|---|---|---|
| `N` | 17 | odd ⇒ passive junction present |
| `l`, `σ`, `μ` | 1.0 | reduced units |
| `ks` | 4444.0 | very stiff ⇒ bonds effectively inextensible |
| `kBT` | 0.1 | noise on |
| `dt` | 5e-5 | |
| `κ` | swept 1 → 50 (30 log-spaced) | bending stiffness |
| `f₀` | swept 1 → 120 (50 log-spaced) | activity |

**Control parameter: `Φ = Fl = f₀/κ`** (flexure number). Standard grid = 30 κ × 50 f₀ = **1500 cells**.

---

## 2. Order parameters

- **`Kconn`** — signed turning angle at the passive junction bead. *The* primary order
  parameter. Equilibrium (perfect circle) = `360/N` = **21.176°** for N=17. Above ⇒ outward
  bend, below ⇒ inward.
- **`Rg`** — radius of gyration. Open ring ≈ `N/2π` = **2.71**; folded in half ≈ `N/4π` = **1.35**.
  Clean bimodal separation with an empty valley — this cut is real, not imposed.
- **`b`** — asphericity ∈ [0,1] (0 = circle, 1 = line).
- **`A1…A6`** — Fourier amplitudes of the turning-angle sequence (A2 elliptical, A3 triangular…).
- **`v_com`, `v_mean`** — COM speed and mean bead speed. `v_com/v_mean` ≈ 1 ⇒ rigid translation;
  ≪ 1 ⇒ internal churn.

---

## 3. Critical gotchas — get these wrong and every result is wrong

**⚠️ Run length is 7.5 s, not 25 s.** 1500 frames × 100 substeps × dt 5e-5 = 0.005 s/frame
⇒ 7.5 s. The `metadata.json` field `"framerate": 60` is the **MP4 playback rate only** and says
nothing about data spacing. **Always slice time windows from the `times` vector**, never from a
frame count. Several older scripts define `LAST_FRAMES = 20*60` and call it "last 20 s" — at
0.005 s/frame that is really the last 6 s.

**⚠️ `Kconn` measures SHAPE, not motion.** A fixed outward-bent (elliptical) ring has constant
`Kconn` yet is a rigidly *rotating/translating* ring with high `v_mean`. Verified: median
`v_mean` rises monotonically with ⟨Kconn⟩−eq (inward 5.4, circular 5.5, mild-out 14, strong-out
31, folded 65). So **outward-bent ⟺ moving; circle & inward ⟺ still.** Do not infer stillness
from a steady `Kconn`.

**⚠️ The two folded families are DIFFERENT states, even though `Rg` cannot tell them apart.**
The "inward-crumpled" (⟨Kconn⟩ ≈ −92°) and "curled outward" (⟨Kconn⟩ ≈ +111°) folds have nearly
the same `Rg` (1.38 vs 1.33), and an earlier version of these notes called them "the same folded
geometry" with the sign of `Kconn` "not a physical distinction". **That was measured and is wrong.**
Complete folded dwells with `Kconn > 0` last **4.1 s** on average (CV 1.00); with `Kconn < 0`,
**50.4 s** (CV 1.22) — a ~12× lifetime difference. The ring is therefore at least **tristable**
(open / fold-out / fold-in). Identify folding with `Rg`, then split it by the **sign of `Kconn`**;
any analysis using `Rg` alone merges two kinetically distinct states.

Transitions between the three run **one way** — open → fold-in → fold-out → open — with
`fold-in → open` **never observed** (0 of 455 exits) and every reverse step suppressed (0 vs 626,
3 vs 455, 219 vs 684). A net cycle is direct evidence of **broken detailed balance**. The outward
fold reached from the inward fold escapes memorylessly (CV 1.14), so it is a genuine intermediate:
**the inward fold must evert before it can open.** (`scripts/analyze_tristate.jl`.)

**⚠️ The old seeding bug.** Sweeps before the fix called `Random.seed!(42)` once per
combination, so all 1500 runs replayed *one* noise stream — one realisation sampled at 1500
parameter points, not 1500 independent samples. Fixed by deriving `seed_effective = seed*1e6 +
combo_index`. **Only folders whose metadata contains `seed_effective` are post-fix.** Folders
named `*_indep` are post-fix. `Results_N17_circle` (original), `*_seed41/43/123/456` are **pre-fix**.

**⚠️ Equipartition normalisation of Kconn variance does not work.** The junction angle is
constraint-stiffened (huge `ks` + closed-ring coupling), so its real variance sits far below
`kBT/κ` and `Var/(kBT/κ)` lives below 1. Otsu on that ratio grabs the wrong valley.

---

## 4. Repo layout

```
Active Ring Final/          ← current project
├── src/
│   ├── EngineRing.jl           forces + Langevin integrator (Params field is `κ`, NOT `kappa`)
│   ├── InitialConditions.jl    bead layouts; circle + kinked ICs (k=1…5)
│   ├── Observables.jl          per-frame shape descriptors
│   ├── MotionAnalysis.jl       motion observables + motion classifier
│   └── IOUtils.jl              save/load
├── configs/                TOML, one per experiment
├── scripts/                ~40 drivers + analysis
└── results/                run dirs + figures (gitignored, large)
Chain/  Rings/              LEGACY — earlier filament and ring work, not current
```

Scripts `include()` the `src/` modules directly — this is **not** a registered package.
Run as `julia --project scripts/<name>.jl [args]` from the `Active Ring Final/` root.

**Per-run output directory** (`results/<folder>/run_XXXX/`): `params.toml`, `metadata.json`,
`traj.jld2` (`times`, `positions` 2×N×frames, `bead_types`), `observables.jld2` (`times` + one
vector per observable), and usually `summary.png`.

### Key datasets

| folder | contents | run length | seeding |
|---|---|---|---|
| `Results_N17_circle_indep` | 1500 cells × 1 seed (the production grid) | 7.5 s | post-fix ✅ |
| `Results_N17_circle_targeted` | 243 cells × 10 seeds (1001–1010) | 7.5 s | post-fix ✅ |
| `Results_N17_phi_long` | 1079 cells × 10 seeds — **campaign in progress** | 30 s | post-fix ✅ |
| `Results_N17_circle`, `*_seed41/43/123/456` | 1500 cells each | 7.5 s | **pre-fix ⚠️** |
| `Results_N17_{,double,triple,quad,quint}kinked{,_indep}` | kinked-IC basin tests | 7.5 s | mixed |

---

## 5. The central methodological result

### The problem

Five successive classifiers were tried and each failed at the phase boundary: amplitude →
equipartition → Kconn-persistence → std(Kconn) → route-based. Each looked right until
neighbouring cells were inspected by eye. **Eyeballing was the bottleneck**, so reliability was
made a number.

### Two measurements (`scripts/phase_reliability.jl`)

- **Reproducibility** — same (κ, f₀), different noise, same label? Modal-label fraction over
  repeated seeds. Needs a multi-seed dataset. *This is the number that says whether a
  single-seed label can be trusted — and the production grid is single-seed per cell.*
- **Coherence** — does a cell agree with its 8 neighbours in (κ, f₀)? Needs a full grid with
  independent per-cell noise (hence post-fix data only).

Read together they separate the failure modes:

| | low coherence | high coherence |
|---|---|---|
| **low reproducibility** | genuine stochasticity (physics) | unstable classifier (our problem) |
| **high reproducibility** | real fine boundary structure | reliable |

### Always score against the chance null, not against 1.0

Expected modal fraction from labels drawn **at random**:

| label mix | n=5 | n=10 | n=20 |
|---|---|---|---|
| 50/50 two-way | 0.688 | **0.623** | 0.588 |
| three-way even | 0.556 | 0.493 | 0.444 |
| 70/30 two-way | 0.746 | 0.712 | 0.702 |

An agreement of 0.71 at n=10 is barely above the two-way null. This is the fix for the
project's recurring arbitrary-threshold problem.

### Classifier scores (measured)

| classifier | reproducibility | coherence | abstained |
|---|---|---|---|
| `route` (current model) | 0.761 | 0.826 | 0% |
| `route_undecided` | 0.758 | 0.810 | 3.7% |
| `final` (final-state) | 0.817 | 0.791 | 0% |
| `legacy_std` (retired 15° cut) | 0.712 | 0.767 | 0% |

`legacy_std` — already rejected on physical grounds — scoring worst on both axes is the
instrument's sanity check. Note `route` and `final` score similarly but produce **very
different diagrams** (86 vs 526 "slithering" cells): they disagree about what the boundary *is*.

---

## 6. Findings (measured, with caveats)

### 6a. Where the diagram is unreliable — **not where previously assumed**

Cross-seed reproducibility vs Φ, on 243 cells × 10 seeds:

| Φ = f₀/κ | cells | mean agreement | cells < 0.7 |
|---|---|---|---|
| < 2 | 13 | 0.808 | 1 |
| **2–4** | 50 | **0.610** | 29 |
| **4–8** | 49 | **0.606** | 36 |
| 8–16 | 46 | 0.713 | 13 |
| 16–32 | 29 | 0.876 | 0 |
| **> 32** | 57 | **0.984** | 0 |

**The high-Φ capture corner is the MOST reproducible region, not the least.** Capture into the
curled state is deterministic enough by 7.5 s that all ten seeds agree. A prior working
assumption — that the high-Φ corner is where the diagram can't be trusted — is **false**.

The unreliable band is the **buckling transition at Φ ≈ 2–8**, where agreement (0.606–0.610)
sits *at or below* the 50/50 chance null of 0.623. Those cells carry no information about the
final state.

⚠️ *Caveat:* the 243 seeded cells were selected from a pre-fix disagreement analysis and span
only κ ∈ [1.0, 2.25]. These are worst-case numbers on contested low-κ cells, not diagram-wide.

### 6b. Cell-level undecidedness cannot be detected from a single run

Two distinct notions were being conflated:
- **Run-level** — this trajectory hadn't converged by t_end. Visible in one run.
- **Cell-level** — this (κ, f₀) doesn't determine a unique outcome; noise picks a basin. A
  property of the *ensemble*.

Measured at the unreliable cells:

```
cells      runs   bstd_fin   spread(dev_fin)   folded
STRONG     1653     0.0075             70.5°     0.80
WEAK        780     0.0113             90.3°     0.59
```

`bstd_fin ≈ 0.011` against a slither cut of 0.04 — **the runs ARE settled.** They each picked a
state and locked in; they just picked *different* ones. A threshold sweep confirms no
separation exists (at 0.010: flags 47.8% of weak runs but also 22.0% of strong). Margin-to-
boundary also fails (mean relative margin 0.257 weak vs 0.272 strong).

**Conclusion: the information is not in a single trajectory.** Cell-level reliability must be
*measured* with replicate seeds. This is a cost statement, not a dead end — and it is why an
abstention/"unsettled" class built from single-run features fired on only 3.7% of cells and
improved nothing.

**Design consequence:** don't make "undecided" a *class* with a threshold. Modal fraction is
continuous — render it as a **confidence channel** (opacity/stipple) over the phase colours.
The band then draws itself at whatever width the data supports, with no cut to defend.

### 6c. Φ vs f₀ — an unresolved confound

| predictor | corr with reproducibility |
|---|---|
| log₁₀ Φ | **+0.761** |
| log₁₀ f₀ | +0.735 |
| coherence | +0.625 |
| log₁₀ κ | −0.108 |

Φ predicts better than the classifier-derived coherence — appealing, because Φ is the theory's
own control parameter and is non-circular.

⚠️ **But the calibration cannot support that claim.** The seeded strip has κ varying only 2.2×
while f₀ varies 67×, giving `corr(log f₀, log Φ) = +0.974` — **collinear**. The data cannot
distinguish "Φ governs" from "f₀ governs". This matters: a Φ band is a *diagonal* stripe, an f₀
band is *horizontal*; at κ = 50 they point to opposite corners. Weak supporting evidence for Φ:
the full-grid coherence map (which does cover the upper-left) shows 99% of incoherent cells at
Φ > 1. Resolving it properly needs ~160 runs seeding high-κ cells where the hypotheses diverge.

**Note this is also the flexure-number scaling claim** that `scripts/fl_origin.jl` derives
analytically (`Fl_c = −1/λ_min(B⁻¹H_act)/κ`, κ-independent when `ks ≫ κ`) — never yet tested
against simulation.

---

## 7. Measured performance (for planning)

| quantity | value |
|---|---|
| physics only, standard run | 0.76 s (5.05 µs/step) |
| marginal per-run in a sweep (incl. observables, traj, summary.png) | 1.90 s |
| per-run without summary.png, 4× length (30 s) | **3.3 s** |
| Julia startup + JIT | ~19 s per process |
| disk per run (no PNG) | 0.77 MB |
| throughput, 1 worker | 0.53 runs/s |
| throughput, **8 workers** | **1.75 runs/s** (3.3×) |
| throughput, 16 workers | 1.39 runs/s — **worse; do not oversubscribe** |

`nproc` reports 16 logical processors but the workload gains nothing from hyperthreading.
**8 concurrent workers is the operating point.** Scaling is sub-linear (3.3×, not 8×).

---

## 8. Current state of work

### Scripts added this session
- **`scripts/reduce_phase_features.jl`** — per-run trajectory → scalar features, cached one CSV
  per folder. Superset of what all classifiers need, so candidates score on identical input.
  All windows sliced from `times`.
- **`scripts/phase_reliability.jl`** — scores any classifier on reproducibility + coherence;
  writes `phase_reliability_<clf>.csv`, `phase_coherence_<clf>.csv`, and a 3-panel figure.
- **`scripts/run_targeted_seeds.jl`** — extended with `--cells`, `--out`, `--frames`,
  `--substeps`, `--seeds`, `--shard I/N`. Defaults reproduce the original campaign. Run-id
  allocation rewritten: shard-reserved id blocks + in-memory counter (the old `next_run_id`
  re-scanned the output dir per run — O(n²), and raced between concurrent workers).

### Campaign in progress
`Results_N17_phi_long` — **1079 cells (Φ > 0.5) × 10 seeds = 10,790 runs at 30 s** (1500 frames
× 400 substeps), 8 shards, ~8.3 GB, ETA ~2–3 h.

Region chosen as the "lower triangle" below the f₀ = κ diagonal in log-log, plus a half-decade
margin. It contains 100% of all known-unreliable cells under both classifiers. It also contains
the high-Φ corner already measured at 0.984, so **those cells double as the control group**.

All 1079 cells are being run (not just the 836 unseeded) because the existing 243 have their 10
seeds at 7.5 s; mixing run lengths would leave an inconsistency in the confidence field exactly
where the data is densest.

### Immediate next steps
1. **Add classify-at-arbitrary-cutoff to `reduce_phase_features.jl`.** A 30 s run contains its
   own 7.5 s run, so labelling the same trajectories at both cutoffs directly tests whether the
   settled states are true attractors or long-lived metastable states — i.e. whether 7.5 s was
   ever long enough. Currently `TAIL_SECONDS = 5.0` is fixed.
2. Re-score classifiers on the new 10-seed data → the **confidence field**.
3. Render confidence as a continuous channel rather than an "undecided" class (§6b).
4. Optionally resolve the Φ vs f₀ confound (§6c) with ~160 high-κ runs.

### Open threads not yet addressed
- Cross-seed reproducibility of the **kinked-IC** sweeps (basin-of-attraction / hysteresis
  question) — data exists, no analysis under the current model.
- `scripts/linear_stability.jl` predicts `f₀_c ≈ 75κ` (Φ_c ≈ 75), but observed structure sits
  near Φ ≈ 2–15. This discrepancy is **unexplained** and worth attention.
- Nothing since commit `afac2fa` is committed; ~40 scripts and 20 configs are untracked.

---

## 9. Reference — classifier thresholds currently in use

```
eq            = 360/N = 21.176°   equilibrium turning angle (N=17)
RG_CURLED_FRAC = 0.70             Rg < 0.70·(N/2π) = 1.894 ⇒ folded
B_WIN_SECONDS  = 0.25             window for LOCAL shape-fluctuation measure
B_SLITHER      = 0.04             median local std(b) above ⇒ writhing, not gliding
B_LOCKED       = 0.01             below ⇒ genuinely at rest
IN_ABS_DEG     = 5.0              Kconn below ⇒ sample "well inward"
OUT_OFF_DEG    = 20.0             Kconn above eq+this ⇒ "well outward"
INWARD_DEV     = -8.0             ⟨Kconn⟩−eq below ⇒ settled inward
SKIP_SECONDS   = 0.5              IC-relaxation transient dropped
TAIL_SECONDS   = 5.0              settle window (t ∈ [2.5, 7.5] for a 7.5 s run)
```

**Use LOCAL std(b) over 0.25 s windows, never global std.** A slow monotonic drift (the ring
elongating, or folding) inflates a global std even though nothing is fluctuating. The local
measure separates a slitherer from a slow drift by ~7×; the global one does not separate them
at all. Getting this wrong previously painted the circle→ellipse transition line as "chaotic".
