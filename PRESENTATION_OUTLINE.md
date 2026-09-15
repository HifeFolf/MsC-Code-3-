# Supervisor Presentation — Slide-by-Slide Outline

**The spine (say this out loud on slide 1, and return to it at the end):**

> *We set out to map the phases of the active ring. We discovered we could not tell
> whether our map was correct. So we built a way to measure correctness — and that
> changed both the method and the physics.*

Everything below hangs off that turn. Don't present as a chronological list of runs;
present as a problem that forced a change of approach. Roughly 24 slides, ~35 min.

All figure paths are relative to `Active Ring Final/results/`.

---

## Slide 1 — Where this ends (open with the destination)

**Figure:** `Results_N17_bistable_long/bistability_evidence.png`

Say: "This is the result — the ring has two states and hops between them, and we can
measure the rates. The rest of this talk is how we got here and why you should believe it."

Audiences follow a long walkthrough far better when they know the destination.

---

## ACT 1 — The question (3 slides)

### Slide 2 — The model
**Figure:** `campaign/model_schematic.png`

- N = 17 beads. Half drive counter-clockwise, half clockwise, meeting at **one passive junction**.
- Net torque is **zero by construction** — the drive cannot be relieved by rotation, so it
  has to go into **shape**. This is a deliberately frustrated system.
- The junction is where frustration localises ⇒ `Kconn` (turning angle there) is the order parameter.
- Panel B: the two states, taken from a real trajectory. Open `Rg ≈ N/2π = 2.71`,
  folded `Rg ≈ N/4π = 1.35`.

### Slide 3 — The hypothesis
- Overdamped Brownian dynamics, stiff bonds (`ks = 4444 ≫ κ`), `kBT = 0.1`.
- Expectation from theory (`fl_origin.jl`): the **flexure number `Fl = f₀/κ`** is the single
  control parameter, and `Fl_c` is κ-independent in the stiff-bond limit.
- `linear_stability.jl` predicted `Fl_c ≈ 75`.
- **Grid:** 30 κ (1→50) × 50 f₀ (1→120), log-spaced.

### Slide 4 — First phase diagrams
**Figures:** `Results_N17_circle_indep/phase_diagram_kconn_states.png`, `phase_diagram_motion.png`

Observed structure sat at Φ ≈ 2–15, **not 75**. A ~20× discrepancy, unexplained at this point.
Flag it — it gets resolved in Act 5.

---

## ACT 2 — The crisis (2 slides)

### Slide 5 — Five classifiers, five answers
**Figures side by side:** `Results_N17_circle_indep/phase_diagram_kconn_std.png`
and `phase_diagram_final_state.png`

Sequence tried: amplitude → equipartition → Kconn-persistence → std(Kconn) → route.
Each looked correct until neighbouring cells were inspected.

Why each failed (one line each):
- **amplitude** — fluctuation size is a featureless continuum, no valley to threshold on
- **equipartition** — the junction is constraint-stiffened, so `Var/(kBT/κ)` sits below 1 and Otsu grabs the wrong valley
- **persistence / std** — measured shape *wobble*, not motion; split near-identical neighbours

### Slide 6 — The actual problem
**Same data, different answers, and no way to choose.** The bottleneck was that correctness
was being judged by eye. That is the turn in the talk.

---

## ACT 3 — The instrument (3 slides)

### Slide 7 — Make reliability a number
Two measurements:
- **Reproducibility** — same (κ, f₀), different noise, same label? Needs repeated seeds.
- **Coherence** — does a cell agree with its 8 neighbours? Needs independent noise per cell.

### Slide 8 — Reading them together
| | low coherence | high coherence |
|---|---|---|
| **low reproducibility** | genuine stochasticity (physics) | unstable classifier (our problem) |
| **high reproducibility** | real fine boundary structure | reliable |

**Figure:** `Results_N17_phi_long/phase_reliability_route.png`

### Slide 9 — Score against chance, not against 1.0
Random labels at n = 10 give modal fraction **0.623** (two-way), 0.493 (three-way).
An agreement of 0.71 is barely above chance.

**Control that validates the instrument:** the already-retired std model scored *worst* on
both axes. If the scoring were meaningless it would not have known that.

---

## ACT 4 — What the instrument caught (3 slides)

### Slide 10 — Two seeding bugs
**Figure:** `Results_N17_phi_long_sharedseed/phase_diagram_confidence.png` vs
`Results_N17_phi_long/phase_diagram_confidence.png`

Both scripts seeded the RNG with a *constant* per sweep, so every parameter point replayed
the **same noise stream** — neighbouring cells were not independent samples.

Damage, measured: the identical physical point (κ=1, kBT=0.1) appeared in three campaigns
and gave Φ₅₀ = **2.29 / 2.85 / 3.51** — a 1.53× spread purely from which streams were drawn,
*as large as the effects being measured*.

Fixed: `run_seed = seed·10⁶ + cell_index`, recorded as `seed_effective`. Everything re-run.

### Slide 11 — The unreliable band is NOT where we assumed
Reproducibility vs Φ (243 cells × 10 seeds):

| Φ | <2 | **2–4** | **4–8** | 8–16 | 16–32 | **>32** |
|---|---|---|---|---|---|---|
| agreement | 0.808 | **0.610** | **0.606** | 0.713 | 0.876 | **0.984** |

The **high-Φ corner is the most reproducible region**, not the least — the opposite of the
prior working assumption. The irreducible band is Φ ≈ 2–4, sitting *at or below* the 0.623
chance null.

### Slide 12 — Undecidedness is invisible in a single run
At unreliable cells, `bstd_fin = 0.0113` vs 0.0075 at reliable ones — both far below the
0.04 threshold. **The runs ARE settled.** They each picked a state and locked in; they just
picked different ones. No threshold separates them (at 0.010: flags 47.8% of weak runs but
also 22.0% of strong).

**Consequence:** cell-level reliability *must* be measured with replicate seeds. And
"undecided" should not be a class with a threshold — render confidence as a colour channel.

---

## ACT 5 — The physics (3 slides)

### Slide 13 — The flexure collapse fails
Binomial-ML logistic fit of P(fold) vs log Φ, per κ row:

| κ window | d(log Φ₅₀)/d(log κ) |
|---|---|
| 1.0–2.0 | +0.236 |
| 2.0–4.0 | +0.742 |
| 4.0–6.7 | +1.213 |

A pure `Fl` collapse needs **0**. And it is **not even a power law** — the slope steepens.
`f₀_c` grows faster than linearly in κ.

*Method note worth stating:* an earlier version used "first Φ with P ≥ 0.5" and reported the
midpoint as pinned. That estimator is biased low and outlier-sensitive. Always fit the logistic.

### Slide 14 — Temperature matters, non-monotonically
| kBT | 0.003 | 0.01 | 0.03 | **0.10** | 0.30 | 1.00 |
|---|---|---|---|---|---|---|
| Φ₅₀ | unbracketed | 5.96 | 3.31 | **2.29** | 3.55 | 7.16 |

U-shaped, **minimum at kBT = 0.1** — which is exactly the value all production data used.
Spread 3.1×.

⚠️ Do NOT summarise with a log-log slope: the two arms cancel to ≈ +0.04 and read as
"temperature independent". The high-kBT arm is also confounded by unfinished kinetics
(median collapse 15.96 s at kBT = 1.0).

### Slide 15 — And `kBT/κ` is not the answer either
Held τ = kBT/κ fixed, varied κ over 4×. Predictions: τ-collapse **0.000**, κ-only **+0.385**.
**Measured −0.19** — past zero, so `kBT/κ` *over*-corrects.

Combining directions (b = −0.596 measured directly): **Φ₅₀ ∝ κ^0.41 · kBT^−0.60**.

*Possible resolution of the Fl_c ≈ 75 puzzle from slide 4:* extrapolating the temperature
dependence toward kBT → 0 reaches Φ₅₀ ≈ 75 around kBT ≈ 1.5e-4. So the deterministic
prediction and the observed Φ ≈ 3 may be the same curve at different temperatures.
**Three-decade extrapolation — present as a hypothesis, not a result.**

---

## ACT 6 — The result (4 slides)

### Slide 16 — Two-way switching is real
**Figure:** `Results_N17_bistable_long/bistability_evidence.png` (Panel A)

Of runs that folded, **13.2% later unfolded**; 95 runs showed ≥3 sustained transitions.
Earlier 7.5 s runs were simply too short to see escape — the "absorbing" conclusion was
correct for high Φ but wrong as a general statement.

### Slide 17 — Localised, and with a control
Mean fraction of seeds switching both ways:

| Φ | 0.5–2 | **2–4** | 4–8 | 8–16 | 16–32 | **>32** |
|---|---|---|---|---|---|---|
| f_switch | 0.030 | **0.099** | 0.018 | 0.011 | 0.010 | **0.000** |

**Zero switching above Φ = 32** is the control — the switching at Φ 2–4 is not a threshold
artifact, because a region exists where the same measurement returns exactly nothing.

### Slide 18 — Dwell-time kinetics (the quantitative core)
**Figure:** `Results_N17_bistable_long/dwell_analysis.png`

1210 complete dwells, 1533 transitions, from 300 s runs sampled at 0.05 s.

| Φ | open mean | open **CV** | folded mean | folded CV |
|---|---|---|---|---|
| 2.0–2.6 | 32.58 s | **1.00** | 20.36 s | 1.80 |
| 2.6–3.2 | 21.15 s | **1.13** | 38.75 s | 1.75 |
| 3.2–4.0 | 15.56 s | **0.81** | 42.47 s | 1.45 |
| 4.0–5.1 | 15.55 s | **0.89** | 74.34 s | 1.09 |

- **Open state escapes exponentially (CV ≈ 1)** ⇒ memoryless, Kramers. This is what licenses
  "two-state switching" over "chaotic wandering across a threshold".
- Rates move monotonically in **opposite** directions — textbook two-state kinetics.
- **Mean dwell times cross at Φ ≈ 2.5** ⇒ a **threshold-free** transition midpoint. Given how
  much trouble arbitrary thresholds caused in this project, this is the number for the abstract.

### Slide 19 — The ensemble view (what an experiment sees)
**Figure:** `Results_N17_bistable_long/seed_ensemble.png`

Ten seeds, identical parameters. At t = 30 s they split 5/5 and look like basin selection —
but every trace visits both states, and several are in the *opposite* state at 300 s.
Occupancy 21–63%, none near 0 or 1.

**The ensemble spread is a snapshot of ergodic hopping.** Consequence: the right order
parameter is **occupancy**, not final state.

---

## ACT 7 — Validation (2 slides)

### Slide 20 — Before / after
**Figure:** `campaign/before_after.png`

Same physics. Left: 1 seed, 7.5 s, forced label — speckled, and a coin-flip cell is
indistinguishable from a certain one. Right: 30 seeds, 30 s, regime from transition
*direction*, saturation carrying agreement.

*(Note the left panel covers the full grid; the right covers Φ > 0.5, hence the white wedge.)*

### Slide 21 — The map makes falsifiable predictions, and we checked
**Figure:** `campaign/validation_map.png` — 22 numbered runs keyed to movie files.

Play two live: a **SWITCH** and an **UPPER**. Final dataset: **32,370 runs, 30 seeds/cell**,
all with independent noise streams.

---

## ACT 8 — Open questions (1 slide)

### Slide 22
1. **Resolved — the folded state is two states.** Splitting folded dwells by the sign of `Kconn`
gives an outward fold (4.1 s, CV 1.00) and an inward fold (50.4 s, CV 1.22) with the same Rg;
averaging them produced the CV 1.83 and the two-slope survival curve. They form a one-way cycle
(open → fold-in → fold-out → open, with fold-in → open never observed) — broken detailed balance.
**Still open:** the inward fold alone is not fully exponential, and the cycle has only been measured
on the bistable locus. See the two new Act 6 slides and `tristate.png`.
2. **The absorbing/bistable boundary is window-limited.** In κ 1–2, Φ 2–5, the 30 s map calls
   30% of cells absorbing that 300 s data shows switching. Fix: map a *rate*, which is
   window-independent, rather than a fraction observed.
3. **Shape and motion are independent axes** and every current figure collapses them. Two
   cells both labelled "never folds" differ 90× in speed; two both "folded" differ 25×.
   `v_com` is already stored — this is a plotting change, not new simulation.
4. High-κ half of the grid: the transition never completes within f₀ ≤ 120, and f₀ = 120 is
   already unphysical. Lowering κ is the way to reach high Φ.

---

## Three pieces of delivery advice

**Put the mistakes in.** Two seeding bugs; a "midpoint is pinned" claim that was an estimator
artifact; a regime classifier that got *stricter* as data improved (13 → 11 cells while
detection went 165 → 233). Each one changed a result. They are the strongest evidence of
rigour you have, and a supervisor will respect them far more than a clean story.

**One number per slide.** Not "reproducibility improved" but "0.828 → 0.886, weak cells
256 → 149".

**Say why, not just what.** Occupancy over final state *because* the final frame is a coin
flip. Logistic fit over threshold crossing *because* crossings are biased low. Those reasons
are the intellectual content of the work.

---

## Asset index

| slide | file (under `Active Ring Final/results/`) |
|---|---|
| 1, 16 | `Results_N17_bistable_long/bistability_evidence.png` / `.pdf` |
| 2 | `campaign/model_schematic.png` / `.pdf` |
| 4 | `Results_N17_circle_indep/phase_diagram_kconn_states.png`, `phase_diagram_motion.png` |
| 5 | `Results_N17_circle_indep/phase_diagram_kconn_std.png`, `phase_diagram_final_state.png` |
| 8 | `Results_N17_phi_long/phase_reliability_route.png` / `.pdf` |
| 10 | `Results_N17_phi_long_sharedseed/phase_diagram_confidence.png` vs `Results_N17_phi_long/...` |
| 18 | `Results_N17_bistable_long/dwell_analysis.png` |
| 19 | `Results_N17_bistable_long/seed_ensemble.png` |
| Act 6 (new) | `Results_N17_bistable_long/tristate.png` / `.pdf` |
| 20 | `campaign/before_after.png` / `.pdf` |
| 21 | `campaign/validation_map.png` / `.pdf` + `campaign/movies_validation/` (22 files) |
| spare | `Results_N17_phi_long/phase_diagram_regimes.png`, `phase_diagram_confidence.png` |

All figures export at `px_per_unit = 3`; PDFs are vector where listed.
