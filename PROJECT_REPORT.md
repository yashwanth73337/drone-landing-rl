# MTP Progress Report — DRL for Autonomous Multirotor Landing

**Yashwanth · IIT Ropar · Semester 1**

---

## 1. Aim

Learn a control policy that lands a quadrotor on a platform — simulation in
semester 1, the lab's ArduPilot hardware in semester 2.

The methodological aim, which has shaped the experimental design throughout:
**binary success rate is a saturated and misleading metric for autonomous
landing.** A policy can report high success while touching down with unsafe
velocity, because the success criterion never asks *how* the drone arrived.

---

## 2. Two strands

### Strand A — Vision-based landing (original work)

`LandingAviary` on `gym-pybullet-drones`: rendered downward camera, real
`cv2.aruco` detection, `solvePnP` pose recovery, UWB fallback. Three sensing
modes as the ablation axis — `privileged`, `abstract`, `camera`.

| Finding | Evidence |
|---|---|
| Abstract sensing models are optimistic | 96.7% (abstract) vs 45% (camera + ArUco) — a 51-point gap |
| Success rate hides safety failures | The 96.7%-success run had **34.9% lateral-velocity violations** |
| Seed variance concentrates in safety | Camera seeds: 0% / 65% / 65% vz violations, success stable |
| Descent capping trades safety for consistency | vz violations 43.5% → 4.1%, success variance up |

### Strand B — Lander.AI reimplementation (this report)

Faithful rebuild of Peter et al., *"Lander.AI: DRL-based Autonomous Drone
Landing on Moving 3D Surface in the Presence of Aerodynamic Disturbances"*,
ICUAS 2024. Purpose: a published-method baseline on the **control** side with
perception removed, so control-side and perception-side failures can be
attributed separately.

---

## 3. Implementation

**Observation** (paper Eq. 3, 15-dim, clipped and normalised to [−1,1]):
attitude · linear velocity · angular velocity · relative pad position ·
relative pad velocity. Privileged ground truth — no camera, no noise.

**Action** (paper Eq. 5): policy outputs `c_t ∈ [−1,1]³`; position target is
`Δp_t = 0.1·c_t`, fed to `DSLPIDControl`.

> `gym-pybullet-drones`' built-in `ActionType.PID` does **not** implement
> this — it treats the action as an *absolute destination* and moves up to
> 1 m/step. `_preprocessAction` is overridden to match the paper exactly.

With a PID controller mediating the action the policy learns only *where to
go*, not *how to stay airborne*. Across 600 evaluation episodes there were
**zero tilt-bound failures** — no crashes, no flips. A concrete validation of
the paper's architectural choice, in sharp contrast to earlier raw-RPM
policies which crashed constantly.

**Algorithm:** TD3 with the paper's hyperparameters (LR 1e-4, buffer 1e6,
batch 100, `learning_starts=100`, ReLU, Adam, actor FC512×2 → FC256 → FC128).

---

## 4. Main technical finding — a discount-evadable stall optimum

Training on the full 11–30 cm spawn range **collapsed**: 1/20 success,
episodes pinned at exactly 300 steps, `actor_loss ≈ +3.8` (Q ≈ −3.8) with a
small, *confidently converged* critic loss. The policy had learned to hover
until the deadline.

Value arithmetic at γ = 0.99:

| Policy | Undiscounted | Discounted |
|---|---|---|
| Hover 300 steps → timeout (−1) | −1 | 0.99³⁰⁰ ≈ 0.05 → **≈ −0.05** |
| Approach, crash at step 50 (−1) | −1 | 0.99⁵⁰ ≈ 0.61 → **≈ −0.61** |
| Approach, land at step 100 (+1 + shaping) | ≈ +2 | 0.99¹⁰⁰ ≈ 0.37 → **≈ +0.74** |

A **terminal-only** timeout penalty falls 300 steps in the future, where
discounting shrinks it to ~5% of face value — effectively free. An immediate
crash costs full price. Break-even success probability for *attempting*
rather than hovering: **≈ 45%**.

That single number explains the whole pattern. At 11–14 cm, exploration
reaches contact often enough to clear 45%, so training escapes. At 25–30 cm
it does not, so the critic correctly converges on hovering. Same code, same
reward — different discovery probability.

**Fix, two coupled changes:**

1. **Per-step time cost, −0.02.** Cannot be evaded by discounting: 300 steps
   at γ=0.99 sums to ≈ −1.9. Success bonus 1 → 20, failure penalty −1 → −5.
   New break-even: **≈ 5%**.
2. **Single-run curriculum.** Difficulty widens *inside one uninterrupted TD3
   run*, so replay buffer, optimizer moments and target networks stay
   continuous. (Earlier save → reload → fine-tune attempts degraded every
   good checkpoint they touched — exactly this discontinuity.)

**Confirmation:** Q flipped from −3.8 to +12.4 after the fix, and every
subsequent successful run sat in the +10 to +12 range.

---

## 5. Results

Curriculum: 12 levels. Levels 1–7 widen the spawn annulus 0.14 → 0.30 m on a
stationary pad; levels 8–12 hold the widest annulus and ramp platform speed
0.05 → 0.25 m/s. Promotion at ≥90% on 20 fixed-seed episodes.

100 evaluation episodes per seed, spawn radius 11–30 cm, `final_model`:

### Stationary platform

| Seed | Success | Precision (cm) |
|---|---|---|
| 1 | 100/100 | 2.73 ± 1.06 |
| 2 | 100/100 | 2.96 ± 0.94 |
| 3 | 100/100 | 4.48 ± 1.18 |
| **Mean** | **100.0 ± 0.0%** | **3.39 ± 0.78** |

### Moving platform (0.25 m/s, horizontal linear)

| Seed | Success | Precision (cm) |
|---|---|---|
| 1 | 96/100 | 4.56 ± 2.48 |
| 2 | 100/100 | 4.05 ± 2.21 |
| 3 | 100/100 | 5.84 ± 1.70 |
| **Mean** | **98.7 ± 1.9%** | **4.82 ± 0.76** |

All three moving-platform seeds reached level 12/12 with Q ≈ +10.3 to +10.6.

**Reliability.** Of five stationary runs, four succeeded and one collapsed
into the stall regime. The reward fix removed the stall optimum but did not
make training bulletproof. The failure is sharply detectable — a stuck run
never promotes past level 1 and its `actor_loss` stays positive — so it can
be caught within ~50k steps rather than wasting a full budget.

**Comparison to the paper.** Lander.AI reports 93.33% success at
6.50 ± 2.14 cm for its LMPL scenario. The comparison is *same scenario type,
lower difficulty*: their platform range is ±0.46 m/s against 0.25 here, and
their episodes are 20 s against 10 s. Not a like-for-like claim.

---

## 6. A worked example of the metric problem

The same three moving-platform training runs read as either

- **46 ± 38%** (100, 17, 21) using `best_success_model.zip`, or
- **98.7 ± 1.9%** (96, 100, 100) using `final_model.zip`

Same policies, same task, same evaluation code. The only difference is which
file was loaded.

**Cause.** The checkpoint selector saved only on strict improvement in
success rate. Once it recorded 100% at an easy early level, nothing could
ever beat it — 100% is the ceiling. Every later 100% at a harder level merely
*tied*, and the distance tie-breaker favoured the easy level, whose
approaches are shorter. Seed 2 saved **twice in 600k steps**, freezing its
"best" checkpoint at a low-speed level.

The project's own `success_evaluations.csv` recorded the truth throughout:

```
590000, 20/20, 100.0%, 3.333 cm
595000, 20/20, 100.0%, 3.256 cm
600000, 20/20, 100.0%, 3.982 cm
```

200k steps of sustained 100% at level 12, with no save. The logs knew; the
selector could not act on it.

**Fix.** Selection now compares `(curriculum_level, success_rate, −distance)`
lexicographically, so a harder level always wins. Replaying seed 2's history
against both rules: 1 save under the old rule, 12 under the new — one per
level reached.

This is the thesis argument reproduced from the inside. A reported landing
success rate depended entirely on a measurement choice invisible in the
headline figure — and the failure was silent, self-consistent, and only
caught because two artefacts of the same run happened to disagree.

---

## 7. Scope and limitations

1. **Sensing is privileged** — exact relative state, no camera, no noise, no
   dropout. Perception is deliberately excluded.
2. **Training budget** is 600k steps against the paper's 5M initial (up to
   35M extended).
3. **Platform motion is horizontal linear only.** Vertical drift is disabled
   (`vertical_speed_factor = 0.0`). With the previous 0.3 factor the pad
   moved ±0.75 m vertically over a 10 s episode from a start of z = 0.25 —
   downward it buried the pad under the floor, upward it put the pad's top
   surface at 1.25 m, above the drone's 1.0 m spawn, making
   `height_above_pad` negative and triggering an unavoidable
   `below_platform` failure. Roughly half of all episodes would have been
   unwinnable. 3D surface motion needs a *bounded* trajectory, not unbounded
   linear drift.
4. **Four documented deviations from the published reward (Eq. 6):**
   - *Mid-range branch.* The paper defines `R` as "the current distance to
     target", identical to `d_target`, making `tanh(α(d_target − R))`
     identically zero. Reimplemented as a progress term on the previous
     step's distance.
   - *"Otherwise" branch.* The first three branches already partition the
     positive reals; the fourth is unreachable and is dropped.
   - *Far-field branch.* The paper's flat `tanh(γ)` carries no gradient;
     replaced with a distance-graded term plus progress.
   - *Terminal rescaling and the per-step time cost* (§4).
5. **No aerodynamic disturbance model** (paper Eqs. 1–2), no domain
   randomisation.

---

## 8. Future work

**Perception integration.** Merge Strand B's stable PID-mediated control with
Strand A's camera + ArUco pipeline. Strand A showed real camera sensing
produces an ~88% blind fraction near touchdown — the marker leaves the 80°
FOV below ~0.15–0.20 m — far worse than abstract dropout models assume.
Whether PID-mediated actions stay stable through that blind window is the
natural next experiment.

**Reference:** Shin et al., *"Vision-Based Autonomous Drone Landing on Moving
Platforms With Uncertain Motion via Deep Reinforcement Learning"*, IEEE RA-L
2026. Its **active-perception reward** (Sec III-C) penalises actions whose
consequence is worse state-estimation accuracy, rather than penalising loss
of FOV directly — the paper explicitly reports that a binary "target in view"
reward *degrades* performance under aggressive platform motion, because
visibility-chasing corrections fight stable descent. Full adoption also needs
an LSTM state estimator and an asymmetric critic, which are architecture-level
changes beyond stock Stable-Baselines3.

**Higher difficulty.** Platform speed to the paper's ±0.46 m/s, 20 s episodes,
and bounded 3D surface motion — the configuration directly comparable to the
paper's LMPL number.

**Semester 2 — hardware.** ArduPilot quadrotors/hexarotors, UWB ranging,
ArUco markers, ROS 1 Noetic. PyBullet was chosen partly for its ground-effect
model, identified early as a likely sim-to-real collapse factor.

---

## 9. Bugs found and fixed

Most results in this project came from finding bugs, not from tuning.

| Bug | Effect | How it was found |
|---|---|---|
| Reward/termination ordering | `BaseAviary.step()` calls `_computeReward()` before `_computeTerminated()`, so touchdown bonuses were never paid — across eight reward variants | Scripted descent with fixed motor commands, printing reward at each height |
| Double-call | `_checkTouchdown()` ran from both reward and termination each step, halving the 3 s approach window to 1.5 s | Per-step cache; eval collapsed to 1/100 under the real window |
| Ratchet vs abort | `RATCHET_SLACK` 0.10 m < `MIN_CLEARANCE_AFTER_ABORT` 0.15 m truncated episodes mid-recovery-climb | Instrumented `truncation_reason` counters |
| ArUco corner scale | `MARKER_OBJ_PTS` used the inner 4×4 payload, scaling every distance by 4/6 | Detection error 0.238 m → 0.020 m after fix |
| Contact geometry | CF2X collision cylinder is 0.025 m, centred on the origin → true contact height 0.0125 m, not the 0.04 m guessed | Read the URDF directly |
| Evaluation radius | Checkpoint callbacks evaluated the default 0.11–0.14 m annulus, not the current curriculum level | Shared `CurriculumState` |
| Curriculum silent at final level | Early return skipped reporting once level 12 was reached | Evaluate before the level check |
| Unbounded vertical drift | Pad left the world within one episode (§7.3) | Trajectory simulation before enabling motion |
| Checkpoint selection ceiling | `best_success_model` frozen at an easy level (§6) | Two artefacts of the same run disagreed |

All results are appended to CSV rather than left in terminal scrollback, and
every major pivot is archived before changes land.

---

## Appendix — reproducing

```bash
cd src/scripts
python train_lander.py --steps 600000 --seed 1

python -c "
import sys; sys.path.append('..')
import train_lander as t
t.ENV_KWARGS['spawn_radius_min'] = 0.11
t.ENV_KWARGS['spawn_radius_max'] = 0.30
t.ENV_KWARGS['platform_speed_range'] = 0.25
t.evaluate(t._latest_model(), 100, 'moving 0.25 m/s, 11-30cm')"
```

Checkpoints: `moving_025_seed{1,2,3}_final.zip`,
`stationary_seed{1,2,3}_final.zip`. Note that `final_model` is the trusted
artefact for runs predating the §6 selector fix.
