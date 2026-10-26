# Active Anti-Polar Ring — Complete Context for Building a LaTeX/Beamer Presentation

**Purpose of this file.** Everything needed to write a presentation about this project, with no access to the
simulation or the data. Every number here is measured, not estimated. Paths are relative to the project root
(the folder containing `Active Ring Final/`).

**Read §13 before writing any claim.** It lists what the data does *not* support. The results are striking and
easy to overstate.

Last updated 2026-10-06. Supersedes `AI_CONTEXT.md`, which predates the three-state result, the passive
control and the kinetics, and whose §5–§8 are stale.

---

## 1. The system

A closed ring of **N = 17** beads in **two dimensions**, overdamped, with thermal noise.

- Beads 1–8: `:active`, self-propel along **+t̂** (the local tangent, towards increasing index).
- Bead 9: `:passive`, no propulsion. This is **the junction**.
- Beads 10–17: `:active_rev`, self-propel along **−t̂**.

Both driven arcs therefore push **towards bead 9**, which is permanently under compression; the opposite point
of the ring (between beads 17 and 1) is under tension. N is odd so there is exactly one junction. This is the
"anti-polar" layout.

### 1.1 Equation of motion

Euler–Maruyama, overdamped, no inertia:

```
r_i(t+dt) = r_i(t) + dt · μ · F_i + sqrt(2 μ kBT dt) · ξ_i,     ξ ~ N(0,1) per component
F_i = F_bond + F_bend + F_WCA + F_active
```

### 1.2 Exact parameters (from `params.toml`, identical across every campaign)

| Symbol | Value | Meaning |
|---|---|---|
| `N` | 17 | beads |
| `l` | 1.0 | bond rest length |
| `ks` | 4444.0 | bond stiffness |
| `κ` (kappa) | **swept, 0.5 – 5** | bending stiffness |
| `f₀` | **swept, 0 – 12** | propulsion force per active bead |
| `kBT` | 0.1 | temperature (so 1 simulation energy unit = 10 kBT) |
| `μ` (mu) | 1.0 | mobility |
| `dt` | 5 × 10⁻⁵ | integration step |
| `σ` (sigma) | 1.0 | bead diameter |
| `ε` (epsilon) | 1.0 | repulsion strength |
| `topology` | ring | closed |
| `theta_max` | π | bending cap (never reached in practice) |
| `theta_buf` | 0.1 | buffer for the cap |
| `k_stop` | 0.0 | cap stiffness (inactive) |
| `L` | 50.0 | box; the ring is shifted rigidly by L when it fully leaves the half-box |

The box is **not** a periodic boundary for interactions — it only recentres the ring. Unwrap centre-of-mass
displacement by detecting jumps larger than L/2.

### 1.3 Potentials (and one important convention)

```
U_bond = Σ_i  ½ ks ( |r_{i+1} − r_i| − l )²                    over all 17 bonds
U_bend = (κ/l) Σ_i ( 1 − cos θ_i )                            θ_i = turning angle at bead i
U_wca  = (ε/6)[ (σ/r)¹² − (σ/r)⁶ ] + ε/24   for r < 2^(1/6) σ, non-bonded pairs only
```

**Convention note that matters:** the engine's repulsive force is `fac = ε/r² (2·sr12 − sr6)`, which is **24×
softer** than a textbook WCA at the same ε. The potential above is the one consistent with that force. This is
a property of the model, not an error, and the energy function has been verified against the engine's own
force routines (§6.5).

**The active force has no potential.** It is non-conservative — it does net work around a closed cycle. Its
absence from the energy is the reason a driven steady state can disagree with Boltzmann statistics.

---

## 2. Observables (computed every frame, inside the run loop)

| Name | Definition | Reference values |
|---|---|---|
| `Rg` | √(λ₁+λ₂), eigenvalues of the 2×2 gyration tensor of the 17 positions about their centroid | perfect circle **2.72** = 1/(2 sin(π/17)); folded ≈ **1.5** |
| `Kconn` | signed turning angle at bead 9: wrap(ψ₉ − ψ₈), where ψ is the bond direction | perfect circle **+21.18°** = 360/17 |
| `b` | (λ₁−λ₂)²/(λ₁+λ₂)² — asphericity | 0 = circular, 1 = a line; invariant under translation, rotation, scaling |
| `v_com`, `v_mean` | centre-of-mass and mean bead speed | used for translating/slithering states |

**Why Kconn matters.** The turning angles of a closed curve sum to +360°. A *positive* angle at the junction is
a convex bend (the junction sits at the tip of a hairpin, exposed). A *negative* angle is a concave dent (the
junction is pushed into the ring, which wraps around it). Rg cannot distinguish these — both are ≈ 1.5.

---

## 3. How a state is defined

One hysteretic tracker, used by **every** analysis, so all numbers are mutually comparable.

```
thr      = 0.70 · N/(2π) = 1.894          reference size
folded   : Rg < 0.9·thr = 1.705
open     : Rg > 1.1·thr = 2.083
between  : state unchanged (deadband — prevents chatter at the boundary)
minimum dwell : a new state must hold 0.5 s before the switch is counted
burn-in  : the first 5.0 s of every run is discarded (runs start as a perfect circle)

three-state refinement, applied to folded frames only:
fold-out : Kconn > +30°
fold-in  : Kconn < −30°
```

---

## 4. How each quantity is computed

**Occupancy** — time in a state ÷ total time. *Not* the state at the final frame. (The final-frame label was
an early error; it discards every switch and misreports exactly the cells of interest.)

**Escape rate** — censored-exponential maximum likelihood:
`k = (completed escapes from the state) / (total time spent in the state)`.
The first and last segment of each run are truncated by the recording window: they contribute to the exposure
time but are never counted as events.

**CV** — standard deviation ÷ mean of the *complete* dwell times. CV = 1 ⇔ exponential ⇔ memoryless escape,
which is what quoting a "rate" presumes. A mixture of two exponentials always gives CV > 1.

**⟨U | Rg⟩** — compute U_bond + U_bend + U_wca for every saved frame from the bead positions, then average
within bins of Rg. Divide by kBT = 0.1 to express in kBT.

**−ln P(Rg)** — histogram Rg, divide by the total, take −ln. **This is an occupancy map, not an energy.**

**Kramers–Moyal effective potential** — from the trajectory, per bin of Rg:
```
D₁(x) = ⟨ΔRg⟩ / τ                drift
D₂(x) = ⟨ΔRg²⟩ / (2τ)            diffusion
U_eff(x) = −∫ D₁/D₂ dx           trapezoid over well-sampled bins
```
τ = 0.05 s (the frame spacing). Makes no Boltzmann assumption, so it is valid in a driven steady state.
Self-consistency identity for a 1-D Langevin process with multiplicative noise:
`−ln P = U_eff + ln D₂ + const`.

**Parameter-free Kramers rate** (the test of the above):
```
k = (D₂(top) / 2π) · √( |U″(well)| · |U″(top)| ) · exp(−ΔU)
```
Curvatures from a least-squares parabola over ±2 bins. Nothing is fitted.

---

## 5. Campaign inventory

| Folder (`Active Ring Final/results/`) | Runs | Cells | Seeds | Length | Trajectories saved | Purpose |
|---|---|---|---|---|---|---|
| `Results_N17_phi_long` | 32,370 | 1,079 | 30 | 30 s | no | the phase diagram |
| `Results_N17_bistable_long` | 401 | 40 | ~10 | 300 s | **yes** | kinetics + energies |
| `Results_N17_wellcuts` | 451 | 45 | 10 | 300 s | no | six fixed-parameter lines |
| `Results_N17_passive` | 70 | 7 | 10 | 300 s | **yes** | the f₀ = 0 control |
| `Results_N17_collapse` + `_indep` | 2,701 ×2 | — | — | 30 s | no | Φ collapse test |
| `Results_N17_kbt_pilot` | 601 | — | — | 30 s | no | temperature dependence |
| `Results_N17_circle_indep` | 1,500 | — | — | 7.5 s | no | seed reproducibility |

Frame spacing is 0.05 s in the 300 s campaigns (6000 frames × 1000 substeps × dt).
`Results_N17_phi_long_sharedseed` exists and is **contaminated** — see §7.1. Do not use it.

---

## 6. Results — every measured number

### 6.1 The phase diagram (`Results_N17_phi_long`)

- 1,079 cells, 30 independent seeds each, 30 s per run.
- **Mean cross-seed agreement 0.883**; **65%** of cells reach ≥ 0.90.
- Chance baselines (essential context — never score against 1.0):
  - 10 seeds: 0.623 (two-way split), 0.493 (three-way), 0.712 (70/30 split)
  - 30 seeds: 0.572 (two-way), 0.424 (three-way)
- Regime counts at 30 s: **719 quiescent, 11 bistable, 349 absorbing** (majority rule);
  **60 of 1,079** cells have ≥ 20% of seeds switching both ways.
- Outputs: `phase_diagram_regimes.png`, `phase_diagram_confidence.png`, `phase_reliability_route.png`.

### 6.2 The control parameter is not Φ

The flexure number Φ = f₀/κ does **not** collapse the data. The transition location scales as
**Φ₅₀ ∝ κ^0.41 · kBT^−0.60**, so two cells at equal Φ are not equivalent. Later campaigns therefore sweep κ
and f₀ separately along fixed lines rather than sweeping Φ.

### 6.3 Three states, one-way cycle (`Results_N17_bistable_long`, 401 runs, 40 cells)

Occupancy, pooled: **fold-in 75%, open 23%, fold-out 2%**.

| State | Mean dwell | CV |
|---|---|---|
| fold-in (Kconn ≈ −65°) | 36.67 s | 1.54 |
| open (Kconn ≈ +90°) | 27.79 s | 1.06 |
| fold-out (Kconn ≈ +110°) | 3.50 s | 1.09 |

Transition counts (and rates where quoted):

| From → to | Count | Rate |
|---|---|---|
| open → fold-in | 626 | |
| open → fold-out | 219 | |
| fold-out → open | 684 | 0.285 s⁻¹ |
| fold-out → fold-in | 3 | |
| fold-in → fold-out | 455 | 0.005 s⁻¹ |
| **fold-in → open** | **0** | never observed |

The inward fold never opens directly: it must first evert to the outward fold, which then opens. This is a
one-way cycle and is direct evidence of broken detailed balance, visible in trajectories rather than inferred.

Two-state split for comparison: folds with Kconn > 0 live 4.10 s (CV 1.00); folds with Kconn < 0 live 50.35 s
(CV 1.22).

### 6.4 Why the two folds differ (`scripts/explain_fold_asymmetry.jl`)

Compared **within each cell** (removes composition confounds), fold-in vs fold-out, 30 cells:

| Quantity | Median difference | Cells where fold-in > fold-out |
|---|---|---|
| non-bonded contacts | +3.43 | **30 / 30** |
| U_wca | +21.7 kBT | **30 / 30** |
| U_bend | +15.2 kBT | **30 / 30** |
| dRg/dt from active forces | −0.089 | 23 / 30 (stronger closing push) |
| dRg/dt from elastic forces | +0.054 | 16 / 30 (no consistent difference) |

So fold-in costs **≈ 37 kBT more** than fold-out, in every cell, and is nonetheless the state the ring occupies
75% of the time. Pooled per-state values (all cells, simulation units):

| State | contacts | U_wca | U_bend | dRg/dt active | dRg/dt elastic |
|---|---|---|---|---|---|
| open | 0.40 | 0.015 | 4.70 | −0.028 | −0.036 |
| fold-out | 12.35 | 2.795 | 10.70 | −0.383 | +0.222 |
| fold-in | 16.34 | 6.090 | 13.43 | −0.534 | +0.386 |

Occupancy is kinetic: fold-in is entered ~3× more often from open (626 vs 219) and escapes ~50× more slowly.

### 6.5 Calibrations (the instrument checks)

- **Energy function vs engine forces:** finite-differencing −dU/dr against the force routines agrees to a
  worst relative mismatch of **3.05 × 10⁻⁹**. The energies are the simulation's own.
- **Thermostat:** with f₀ = 0 all internal forces cancel in the sum, so the centre of mass is a free Brownian
  particle with **D = μ·kBT/N = 0.005882 exactly** (Euler–Maruyama is exact on a free coordinate). Measured:

  | lag | D | deviation |
  |---|---|---|
  | 0.10 s | 0.005871 | −0.2% |
  | 0.25 s | 0.005858 | −0.4% |
  | 0.50 s | 0.005868 | −0.2% |
  | 1.00 s | 0.005855 | −0.5% |
  | 2.50 s | 0.005784 | −1.7% |
  | 5.00 s | 0.005751 | −2.2% |

  (The drift at long lags is falling sample count.) **Equipartition was deliberately not used:** ks·μ·dt = 0.22,
  so Euler–Maruyama inflates bond energy by a mode-dependent factor between 1.00 and 1.29.

### 6.6 The passive control (`Results_N17_passive`)

f₀ = 0, seven stiffnesses (κ = 0.5, 0.667, 0.889, 1.0, 1.186, 1.581, 2.108), 10 seeds, 300 s each.

- 70 runs, **20,647 s** of simulated time
- **0 folding events**
- closest approach ever: **Rg = 2.24**, against a folding threshold of 1.894

The folded state does not exist without the drive. This rules out thermal fluctuation, threshold artefacts and
integrator artefacts in one measurement, since none of those switch off with f₀.

### 6.7 The central non-equilibrium result

Across the **38 cells** where the ring visits both states:

- energy cost of folding: **62 – 232 kBT**, median **145**
- measured folded/open time ratio: median **6.50** (the ring prefers the expensive state)
- Boltzmann prediction at the same ΔU: median **4.6 × 10⁻⁶³**
- discrepancy: **≈ 64 orders of magnitude** (the figure annotates the median-ΔU cell at 62)

Single reference cell (κ = 1.0, f₀ = 2.214, 10 runs, 59,000 frames): folding costs **90 kBT**; the ring is
folded about half the time.

### 6.8 The three landscapes (reference cell κ = 1.0, f₀ = 2.2143)

| Curve | Barrier from folded | from open | Top at Rg |
|---|---|---|---|
| −ln P | 3.47 | 3.48 | 1.88 |
| U_eff (Kramers–Moyal) | 2.19 | 2.08 | 2.06 |

Wells at Rg 1.55 and 2.52, equally deep (difference 0.01). D₂ ranges 1.40×10⁻³ … 7.11×10⁻³, a factor **5.1**;
ln(5.1) = 1.63 accounts for the gap between the two estimates. The self-consistency check
(U_eff + ln D₂ overlaying −ln P) passes, with a visible ~0.4 discrepancy near the open well.

**Real potential energy** relative to the folded well: barrier position −43.9 kBT, open well **−75.6 kBT**
(bending −53.4, repulsion −21.7, stretching −0.6). **Monotonic, no hump.** There is no energy barrier.

Mean energies over the window (simulation units): U_bond 1.2, U_bend 5.16, U_wca 0.668.

### 6.9 Kinetics (`Results_N17_wellcuts` + `Results_N17_bistable_long`, 85 cells, 29 usable)

- Escape rates span **0.003 – 0.15 s⁻¹**.
- Rates respond oppositely to the two parameters and **cross at the transition line**. Example at f₀ = 2.85:
  κ 0.67 → 1.58 raises folded-escape from 0.0054 to 0.102 s⁻¹ (20×) while open-escape falls from 0.108 to
  0.0073 s⁻¹ (15×). The drive *tilts* the landscape rather than lifting one side.
- Memoryless test: escape from **open CV = 0.95** (exponential ✓), from **folded CV = 1.49** (✗ — a mixture,
  because "folded" is two states).
- **Parameter-free Kramers prediction vs measurement:**

  | Direction | n | Median measured/predicted | 10–90% |
  |---|---|---|---|
  | open → folded (passes the CV test) | 29 | **2.01** | 0.60 – 9.45 |
  | folded → open (fails it) | 29 | 4.61 | 2.56 – 47.31 |

- −ln P overstates the kinetic barrier: median ratio **1.21** (folded direction), **1.73** (open direction).
- The `ln k` vs barrier **slope test is not a clean discriminator**: −0.38 (U_eff, R² 0.66) and −0.69
  (−ln P, R² 0.72) where Kramers predicts −1. Confounded by a per-cell prefactor, by the two directions of one
  cell not being independent, and by the −ln P barrier difference being partly tied to occupancy by
  construction. Binning sensitivity (nbin, min pairs → usable cells, U_eff slope, −ln P slope):
  (40,80) → 24, −0.55, −0.64 · (36,60) → 34, −0.52, −0.66 · (28,40) → 58, −0.38, −0.69 ·
  (24,30) → 70, −0.36, −0.71 · (20,20) → 78, −0.31, −0.71. **Recommend omitting this panel.**

### 6.10 Markov check (31 cells) — the limits of the effective description

⟨ΔRg²⟩ ~ τ^p over τ = 0.05 – 0.40 s (p = 1 diffusive, p = 2 ballistic, p < 1 confined):

| Where | p | 10–90% |
|---|---|---|
| folded well | **0.57** | 0.46 – 0.65 |
| open well | **0.98** | 0.86 – 1.10 |
| barrier top | **1.69** | 1.38 – 1.90 |

So Rg is **not** a diffusion at our sampling lag, and the violation is worst exactly where the rate is decided.
Extrapolating on the three shortest lags, the true barrier is **10–16% higher** than measured at τ = 0.05 s.
(A line through all six lags wrongly reports +5%; the lag dependence is curved — always extrapolate on the
short end.)

Chapman–Kolmogorov: |P(2τ) − P(τ)²| = **2.93×** the odd-vs-even-runs noise floor (median CK distance 0.0295,
floor 0.0124). Real memory, but small in absolute terms — about 0.03 of probability mass.

**Rejected hypothesis, worth stating:** the memory is *not* the hidden fold-in/fold-out distinction — CK/floor
does not correlate with the folded-dwell CV (r = −0.02).

### 6.11 Well cuts (`Results_N17_wellcuts`)

45 cells on six lines (three at fixed f₀ sweeping κ, three at fixed κ sweeping f₀), 10 seeds × 300 s =
59,000 samples per cell. Folded well at Rg ≈ 1.5, open well at 2.45–2.7 (it moves right as κ increases,
towards the perfect circle at 2.72).

Sampling floor: with 59,000 samples the rarest visible event sits **9–10.8** units above the minimum, so any
−ln P hump at 7–9 is a lower bound, not a measurement. Smooth humps at 3–6 are genuine.

Barrier grows along the transition line (cells with roughly even occupancy):

| Cell (κ, f₀) | Time folded | −ln P hump |
|---|---|---|
| 0.89, 2.13 | 57% | 4.1 |
| 1.19, 2.85 | 60% | 5.0 |
| 1.58, 3.79 | 55% | 5.3 |
| 2.11, 5.06 | 50% | 5.9 |
| 2.81, 6.75 | 53% | 6.9 |

---

## 7. Corrections made during the project (all four changed published numbers)

**7.1 Shared random seeds.** One noise stream was reused across an entire sweep, so "independent seeds" were
correlated. Fixed to `run_seed = seed × 10⁶ + cell_index`, recorded per run as `seed_effective`. The affected
campaign was **rerun from scratch**, not patched; the contaminated data is kept under
`Results_N17_phi_long_sharedseed` and must not be used.

**7.2 Final-frame classification.** Replaced by occupancy (§4).

**7.3 Run length assumed rather than calibrated.** 7.5 s → 30 s → 300 s. About **30%** of cells labelled
permanently folded at 30 s do switch at 300 s.

**7.4 Φ assumed to be the control parameter.** Tested and rejected (§6.2).

A fifth, conceptual: **−ln P was labelled an "effective potential"** on earlier figures. It is an occupancy map.
In equilibrium that identification is exact and standard; with a drive it is unavailable.

---

## 8. Limitations — state these, don't wait to be asked

1. **The current phase diagram classifies on Rg alone**, so it cannot show the tristability. It answers
   "folded or open" reliably; it does not resolve the two folds.
2. **U_eff is approximate** by a measured 10–16% (§6.10), and barriers above ~6 noise units are unreliable
   because those cells cross too rarely to constrain the integral.
3. **The one-way cycle is measured on the bistable locus only**, not across the diagram.
4. **The ±30° cut** separating the two folds has not been robustness-scanned (the populations sit near −65°
   and +110°, so nothing should hinge on it, but this is argued rather than shown).
5. **The model**: two dimensions, a single ring, free-draining (no hydrodynamics), and excluded-volume
   repulsion 24× softer than a textbook WCA.
6. **No experimental comparison** has been attempted.

---

## 9. Figure inventory

All PNGs are rendered at `px_per_unit = 3` (≈ 4500 px wide), so they are safe at any Beamer size. Where a
`.pdf` exists, prefer it — it is vector and scales perfectly.

| File (under `Active Ring Final/results/`) | PDF? | Aspect | What it shows |
|---|---|---|---|
| `campaign/model_schematic.png` | yes | wide | the 17-bead ring, drive directions, the junction |
| `Results_N17_phi_long/phase_diagram_regimes.png` | yes | ~square | regime map over (κ, f₀) |
| `Results_N17_phi_long/phase_diagram_confidence.png` | yes | ~square | the same, shaded by confidence |
| `Results_N17_phi_long/phase_reliability_route.png` | yes | wide | seed agreement + neighbour coherence per cell |
| `campaign/traces_simple.png` | **no** | 2.4:1 | Rg(t) for one passive and one active run — the simplest evidence |
| `campaign/showcase_nonequilibrium.png` | yes | 2.9:1 | **the headline**: energy vs occupancy, the two traces, Boltzmann vs measured |
| `Results_N17_bistable_long/tristate.png` | yes | wide | 2-D occupancy map in (Rg, Kconn), transition matrix, survival curves |
| `Results_N17_bistable_long/fold_shapes.png` | **no** | 2.7:1 | real snapshots of open / fold-out / fold-in with active-force arrows |
| `Results_N17_bistable_long/bistability_evidence.png` | yes | wide | switching evidence across seeds |
| `Results_N17_bistable_long/dwell_analysis.png` | **no** | wide | dwell-time distributions and survival |
| `Results_N17_bistable_long/seed_ensemble.png` | **no** | wide | same parameters, different seeds, different outcomes |
| `campaign/energy_landscape.png` | yes | 2.9:1 | −ln P vs U_eff vs real energy, in three panels |
| `campaign/well_cuts.png` | yes | wide | effective double wells along the six cuts |
| `campaign/cut_map.png` | yes | ~square | where those six cuts sit on the diagram |
| `campaign/kramers_rates.png` | yes | 1.7:1 | escape rates, the Kramers test, the CV check |
| `campaign/markov_check.png` | yes | 1.7:1 | lag dependence of D₁, D₂, the barrier, and Chapman–Kolmogorov |
| `campaign/validation_map.png` | yes | ~square | which cells were rendered as movies, numbered |
| `campaign/before_after.png` | yes | wide | the diagram before and after the corrections |

### 9.1 Caption drafts

- **showcase_nonequilibrium** — "The ring's own potential energy (red, left axis) falls monotonically from the
  folded to the open size, while the time it spends there (black, right axis) has minima at both. Shaded: the
  entire range visited by a passive ring in 20,647 s. Centre: one run each at κ = 1.0, with and without
  propulsion. Right: for every cell visiting both states, the occupancy ratio equilibrium would predict from
  the measured energy gap (blue) against the measured ratio (red)."
- **tristate** — "Occupancy in the (Rg, Kconn) plane pooled over 401 runs, showing three basins; the
  transition matrix between them; and the survival curve of each state. The inward fold is never observed to
  open directly."
- **fold_shapes** — "Representative configurations, each the frame nearest its state's median (Rg, Kconn).
  Arrows show the active force on each bead. The junction (black) sits at the tip of a hairpin when the fold is
  outward and is buried within the ring when it is inward."
- **energy_landscape** — "Three quantities with the same double-well appearance and different meanings:
  −ln P(Rg), the occupancy; U_eff, reconstructed from drift and diffusion, whose barriers set the rates; and
  ⟨U|Rg⟩, the true potential energy, which has no barrier at all."
- **kramers_rates** — "Escape rates along fixed-parameter lines (A, B); −ln P against the kinetic barrier (C);
  the memoryless test (F); and a parameter-free Kramers rate against the measured one (E)."
- **markov_check** — "Drift and diffusion at sampling lags 0.05–0.40 s, the resulting barrier extrapolated to
  zero lag, and the Chapman–Kolmogorov residual against the noise floor obtained from independent halves of
  the data."

---

## 10. Suggested presentation structures

### 10.1 Twenty minutes, result-led (recommended for a general audience)

1. The system — `model_schematic`
2. The question: how many states, and where — one sentence
3. The phase diagram with its reliability numbers — `phase_diagram_confidence`
4. Not two states but three — `fold_shapes`, then `tristate`
5. Are they real states? — `traces_simple`, the passive control numbers
6. They cost energy and are occupied anyway — `showcase_nonequilibrium`
7. Kinetics: rates cross at the transition — panels A/B of `kramers_rates`
8. Limits and next steps

### 10.2 Thirty minutes, chronological (for a supervisor who wants the reasoning)

Add after (3): the four corrections (§7), each with its before/after — `before_after` fits here.
Add after (7): the effective potential and its test — `energy_landscape`, then panels E/F of `kramers_rates`,
then `markov_check` as the honesty slide.

### 10.3 Methods seminar

One frame per measurement: observable → state definition → occupancy → rate → energy → calibration → control.
Each frame states what goes in, what comes out, and the control. The two calibrations (§6.5) and the passive
control (§6.6) carry this version.

---

## 11. Notation for LaTeX

```latex
N = 17, \; \kappa, \; f_0, \; k_BT, \; \mu, \; \sigma, \; \epsilon, \; \Delta t
R_g                                  % radius of gyration
K_{\mathrm{conn}}                    % signed turning angle at the junction
\Phi = f_0/\kappa                    % flexure number (NOT the control parameter)
\langle U \,|\, R_g \rangle          % mean potential energy at a given size
-\ln P(R_g)                          % occupancy map
U_{\mathrm{eff}}                     % Kramers-Moyal reconstruction
D_1, D_2                             % drift, diffusion
\mathrm{CV}                          % coefficient of variation of dwell times
```

Useful macros:

```latex
\newcommand{\kbt}{k_{\mathrm B}T}
\newcommand{\Rg}{R_{\mathrm g}}
\newcommand{\Kconn}{K_{\mathrm{conn}}}
\newcommand{\foldin}{\textsc{fold-in}}
\newcommand{\foldout}{\textsc{fold-out}}
```

Beamer notes: the wide figures (2.4:1 to 2.9:1) fit `\includegraphics[width=\textwidth]` on a 16:9 frame with
room for a caption. `showcase_nonequilibrium` and `energy_landscape` are three-panel figures — if a frame is
tight, crop to the single panel being discussed rather than shrinking the whole thing. Use `\tiny` sparingly;
these figures already carry their own axis labels at a readable size.

---

## 12. Sentences that are safe to use verbatim

- "The ring spends a median of 6.5 times more time in the folded state than the open one, although folding
  costs between 62 and 232 kBT depending on the parameters."
- "Equilibrium statistics at the measured energy gap would give an occupancy ratio of order 10⁻⁶³; we measure
  of order 1."
- "With the propulsion switched off and every other parameter unchanged, the ring did not fold once in
  20,647 s of simulation, and never approached closer than Rg = 2.24 against a folding threshold of 1.894."
- "Escape from the open state is memoryless (CV = 0.95), and a Kramers rate built from the reconstructed
  potential, the measured diffusion and the well curvatures predicts the measured rate within a median factor
  of two across 29 cells, with no fitted parameter."
- "In 455 observed exits from the inward fold, the ring never once passed directly to the open state."
- "The energy function used here was verified against the engine's own force routines to a relative accuracy
  of 3 × 10⁻⁹."

---

## 13. What the data does NOT support — read before writing any claim

1. **Do not call −ln P a free energy, an energy, or a potential of mean force.** It is an occupancy map. This
   identification requires equilibrium, which this system does not have.
2. **Do not say we measured a free energy difference.** We measured a *potential energy* difference (§6.7) and,
   separately, an occupancy. The entropy term is not measured. The argument closes because the passive control
   is a genuine equilibrium system whose occupancy already contains every entropic contribution — and it gives
   zero. Make that the argument, not an entropy estimate.
3. **Do not state the barrier is "N kBT".** U_eff is in units of the effective noise, not kBT, and it is
   10–16% low (§6.10). Quote barriers as dimensionless or avoid them.
4. **Do not present the slope test** (`ln k` vs barrier) as evidence for or against anything (§6.9).
5. **Do not claim the mechanism of the fold asymmetry.** We know fold-in is entered more often and escapes far
   more slowly, and that it costs ~37 kBT more. Why the drive favours entering it is *not* established.
6. **Do not claim tristability across the whole diagram.** It is measured on the bistable locus (40 cells).
7. **Do not describe the states as "phases".** They are metastable states of a finite system; there is no
   thermodynamic limit here.
8. **Do not quote the phase diagram as a three-state map.** It classifies on Rg (§8.1).
9. **Do not claim experimental relevance, a prediction for an experiment, or agreement with published
   measurements.** None has been attempted.
10. **Do not round 20,647 s to "6 hours"** or similar — quote measured numbers as measured.
11. **Do not invent error bars.** Where a spread is given above it is a 10–90% range across cells, not a
    standard error; say so.
12. **Do not attribute the Chapman–Kolmogorov memory to the hidden fold states.** That was tested and rejected
    (§6.10).

---

## 14. Scripts, for reference

| Script (`Active Ring Final/scripts/`) | Produces |
|---|---|
| `reduce_phase_features.jl` | per-run features → `phase_features.csv` |
| `phase_reliability.jl` | reproducibility and coherence maps |
| `phase_diagram_regimes.jl`, `plot_regimes.jl` | regime map |
| `phase_diagram_confidence.jl` | confidence-shaded diagram |
| `analyze_dwell.jl` | dwell times, survival curves |
| `analyze_tristate.jl` | the three-state analysis |
| `explain_fold_asymmetry.jl` | per-state energies, contacts, force projections, snapshots |
| `ring_energy.jl` | the energy module (validated against engine forces) |
| `analyze_energy_landscape.jl` | −ln P vs U_eff vs ⟨U\|Rg⟩ |
| `analyze_kramers_rates.jl` | escape rates, the Kramers test |
| `markov_check.jl` | lag dependence, Chapman–Kolmogorov |
| `showcase_nonequilibrium.jl` | the headline figure and its three checks |
| `plot_traces_simple.jl` | the single-panel trace figure |
| `plot_well_cuts.jl`, `plot_cut_map.jl` | the six cuts and their map |
| `run_targeted_seeds.jl` | the campaign runner (note: `--out` is relative to the project root) |
