# Landing on a Moving Platform — Experiment Notes

**Project:** RL-based autonomous multirotor landing on a moving platform
**Simulator:** gym-pybullet-drones (PyBullet physics)
**Algorithm:** PPO (stable-baselines3), MLP policy
**Drone:** Crazyflie 2.X (CF2X) — 27 g, thrust-to-weight 2.25, ground effect modelled

---

## 1. Headline result

A PPO policy lands a quadrotor on a platform whose motion is **randomised every
episode** — amplitude 0.2–1.0 m, angular frequency 0.2–1.5 rad/s, arbitrary
phase offset.

**85% success over 100 evaluation episodes.**
Touchdown at 0.041 m/s vertical, 0.075 m/s lateral, 2.49° tilt, 4.5 cm from pad
centre. Zero violations of the proposed safety thresholds.

Of the 15 failures, **none** were landings that missed the pad. All 85
touchdowns were on target; the remaining 15 episodes never committed to a
descent and were truncated.

---

## 2. What was built

Two custom environment classes were written for this project. Neither exists in
gym-pybullet-drones. Both inherit from the library's `BaseRLAviary`, which
supplies the physics — rotor thrust, drag, ground effect, PyBullet stepping.

**`TrackingAviary`** — the drone follows a moving target point. An intermediate
task: landing without the descent.

**`LandingAviary`** — the core environment.

### Custom components in `LandingAviary`

| Component | Description |
|---|---|
| Moving platform | Kinematic box (mass 0) teleported along a sine trajectory each step. Has a collision shape so the drone can rest on it, but physics cannot push it. |
| Randomised motion | Amplitude, frequency, and phase resampled at every episode reset. See §5. |
| Relative observation | Parent observation extended by 6 values: platform position and velocity **relative to the drone**. |
| Reward function | Five terms — alignment, velocity matching, gated descent progress, tilt penalty, terminal touchdown bonus. |
| Touchdown detection | Defined as a height condition (within 4 cm of the pad surface) rather than relying on the physics engine's contact reporting, which depends on collision-mesh detail and is not reproducible. |
| Contact metrics | Vertical speed, lateral relative speed, tilt, and offset recorded at the instant of contact. |
| Descent ratchet | The drone may not climb more than a fixed slack above the lowest altitude already reached. Removes indefinite hovering as an option. |

---

## 3. Observation space — current status and known limitation

The policy currently receives **privileged state**: exact relative position and
velocity of the platform, read directly from the simulator, noiseless and always
available.

```python
rel_pos = self._getPlatformPos() - state[0:3]
rel_vel = self._getPlatformVel() - state[10:13]
```

This is deliberate. It separates the **control** problem from the **perception**
problem. If the drone cannot land even with perfect information, adding sensor
realism only obscures the cause of failure.

A real drone cannot measure this. The planned replacement models the lab's
actual sensing stack:

- **UWB** — noisy range, 10–50 Hz, decimetre accuracy, occasional NLOS
  outliers. Available at all ranges and attitudes.
- **ArUco marker** — centimetre accuracy, but only while the marker is inside
  the downward camera's field of view. Geometry: a camera at height *h* sees a
  footprint of roughly 2*h*, so a marker of side *s* leaves the frame at about
  *h ≈ s*. For a 20–50 cm marker, the drone is **blind for the final 20–60 cm
  of every landing**.

Handling that terminal blind window is the intended contribution of this work.

---

## 4. Reward function

```
reward = alignment + velocity_matching + progress + tilt_penalty + touchdown_bonus
```

| Term | Form | Purpose |
|---|---|---|
| Alignment | `exp(-3 · horizontal_distance)` | Get over the pad. Horizontal only — vertical distance is what we want the drone to close. |
| Velocity matching | `0.5 · exp(-2 · \|v_drone − v_platform\|)` | Move *with* the platform. Right place at the wrong speed means a sideways impact. |
| Progress | `30 · (previous_height − current_height)`, gated | Rewards *closing the gap*, not *being close*. |
| Tilt penalty | `−0.3 · tilt` | A drone touching down tilted catches a leg and flips. |
| Touchdown bonus | `300 + 800 · softness · centred · level` | Terminal reward scaled by landing quality. |

### Two design decisions worth defending

**Progress rather than proximity.** Earlier versions rewarded *being near* the
pad. Any such reward creates a place to park: if hovering at some altitude pays
well per step, the policy hovers there forever. A progress term is a function of
*change* — positive when descending, negative when climbing, exactly zero when
hovering. It cannot be farmed by standing still.

**The descent gate.** Progress only pays while the drone is horizontally within
1.5 pad-widths. Outside that radius, descending earns nothing. This encodes the
*commit-timing* decision, which is what makes landing harder than tracking: the
drone must position itself over the pad **first**, then descend. Without the
gate the policy dives immediately from wherever it happens to be and lands on
open floor.

---

## 5. Two bugs found

### Bug 1 — the reward/termination ordering bug

Eight consecutive reward designs produced 0/100 successful landings. All
converged on the same behaviour: the drone tracked the platform correctly, then
hovered above it until the episode timed out.

| Version | Change | Outcome |
|---|---|---|
| v1 | bonus 100, time penalty −0.5 | Hovered the full episode. |
| v2 | bonus 600, descent reward ×3 | Hovered, at a different altitude. |
| v3 | time penalty −2.0 | Self-terminated within 4 steps — living cost more than dying. |
| v4 | progress term, bonus 200 | Hovered at 0.26 m, centred to 5 mm, perfectly level. |
| v4b | bonus 800, alignment weight cut 70% | Still hovered; tracking *degraded* to 0.22 m offset. |
| v5 | descent ratchet, slack 0.15 | Oscillated between 0.17 m and 0.32 m. |
| v5b | slack 0.05, flat +300 bonus | Still hovered. |
| **v6** | **ordering bug fixed** | **100/100 successful landings.** |

`BaseAviary.step()` evaluates in this order:

```python
obs        = self._computeObs()
reward     = self._computeReward()        # ← first
terminated = self._computeTerminated()    # ← calls _checkTouchdown()
```

`_checkTouchdown()` sets `touchdown_recorded = True`, but it was only called
from `_computeTerminated()` — which runs **after** `_computeReward()`.

So on the step where the drone reached the pad, the reward function still saw
`touchdown_recorded == False`, skipped the bonus block, and returned an ordinary
shaping reward. The episode then terminated, leaving no subsequent step in which
the bonus could be paid.

**The landing bonus was never awarded, at any value.** The agent had no evidence
that landing was worth anything and correctly learned to hover.

Fix: `_computeReward()` now calls `_checkTouchdown()` itself before evaluating
the bonus. The method is idempotent, so calling it from both places is safe.

**How it was found:** by instrumenting a *scripted descent* — flying the drone
down with fixed motor commands, no neural network involved, printing the reward
at each height. This separated *"can the environment produce a landing at all?"*
from *"is the reward encouraging one?"*. Eight rounds of reward tuning had been
spent on a problem that was never in the reward.

### Bug 2 — phase memorisation

After the ordering fix, a four-stage curriculum reached **100% success on a
moving platform**. That number turned out to be an artefact.

Two sweeps of the trained policy over platform motion parameters gave
contradictory results at the same peak speed:

| Peak speed (m/s) | Frequency sweep (A=0.5, ω varies) | Amplitude sweep (ω=0.5, A varies) |
|---|---|---|
| 0.250 | 100% | 100% |
| 0.375 | **26%** | **100%** |
| 0.500 | **0%** | **92%** |
| 0.750 | 0% | 0% |

Same speed, opposite outcomes — so peak speed was not the limiting quantity.

A follow-up test decoupled speed from acceleration by solving for the *A* and
*ω* that produce any desired pair (peak speed = *Aω*, peak acceleration =
*Aω²*). It rejected the acceleration hypothesis too, because success was
**non-monotonic** in both:

| ω | peak accel | success |
|---|---|---|
| 0.200 | 0.050 | 0% ← *gentler* than trained, still fails |
| 0.500 | 0.125 | 100% ← trained value |
| 0.800 | 0.200 | 0% |
| 1.200 | 0.300 | 0% |
| 1.800 | 0.450 | 100% ← near a harmonic of the trained rhythm |
| 2.800 | 0.700 | 0% |

A physical limit produces a clean threshold. Success that appears and disappears
with frequency is a **timing artefact**.

**Diagnosis.** With fixed *A* and *ω*, and every episode starting at *t* = 0,
the platform is at a fully predictable position at every timestep. Nothing
forced the policy to use its relative-position observation — memorising a timed
routine ("at step 60, move here") scored equally well and was easier to learn.
The reported 100% was measuring memorisation, not tracking.

**Fix.** Platform amplitude, frequency, and phase are now resampled at every
episode reset. Phase matters most: without it every episode began with the
platform at *x* = 0 moving in +*x*, itself a memorisable cue. With randomisation,
no timed schedule can succeed and the six relative-observation values are the
only way to locate the platform.

Shin et al. (RA-L 2026) randomise platform motion during training for exactly
this reason.

---

## 6. Results

### Fixed-motion curriculum (superseded — see §5, Bug 2)

Retained for the record. These numbers reflect memorisation, not tracking.

| Step | Start height | Pad half-width | Amplitude | Success | vz (m/s) | lat (m/s) | tilt (°) | offset (m) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.4 m | 0.50 m | 0.0 | 100% | 0.184 | 0.088 | 5.22 | 0.102 |
| 0\* | 0.4 m | 0.50 m | 0.0 | 100% | 0.184 | 0.020 | 0.85 | 0.073 |
| 1 | 0.8 m | 0.50 m | 0.0 | 100% | 0.187 | 0.034 | 0.09 | 0.065 |
| 2 | 0.8 m | 0.20 m | 0.0 | 100% | 0.187 | 0.029 | 0.12 | 0.095 |
| 3 | 0.8 m | 0.20 m | 0.5 | 100% | 0.091 | 0.076 | 0.51 | 0.039 |

\* Same configuration and code as step 0, different random seed, second machine.

### Randomised motion (current, honest baseline)

Amplitude ∈ [0.2, 1.0] m, ω ∈ [0.2, 1.5] rad/s, phase ∈ [0, 2π), all resampled
per episode. Pad half-width 0.20 m, start height 0.8 m.

**Training budget comparison** (single seed):

| Training steps | Success | Touchdowns | vz (m/s) | lat (m/s) | tilt (°) | offset (m) | lat > 0.3 |
|---|---|---|---|---|---|---|---|
| 500k | 42% | 80/100 | 0.162 ± 0.060 | 0.206 ± 0.126 | 2.32 ± 1.86 | 0.107 ± 0.054 | 21.4% |
| 1.5M | 85% | 85/100 | 0.041 ± 0.027 | 0.075 ± 0.041 | 2.49 ± 1.83 | 0.045 ± 0.023 | 0% |

**Three independent training seeds at 1.5M steps**, 100 evaluation episodes each.
Uncertainties within a row are across episodes; the final row is across seeds.

| Seed | Success | vz (m/s) | lat (m/s) | tilt (°) | offset (m) |
|---|---|---|---|---|---|
| 1 | 85% | 0.041 ± 0.027 | 0.075 ± 0.041 | 2.49 ± 1.83 | 0.045 ± 0.023 |
| 2 | 86% | 0.056 ± 0.043 | 0.075 ± 0.045 | 2.57 ± 1.67 | 0.054 ± 0.035 |
| 3 | 79% | 0.045 ± 0.025 | 0.053 ± 0.032 | 2.36 ± 1.89 | 0.032 ± 0.018 |
| **Across seeds** | **83.3 ± 3.1%** | **0.047 ± 0.006** | **0.068 ± 0.010** | **2.47 ± 0.09** | **0.044 ± 0.009** |

Constraint violations were **0% in all three seeds** for all three thresholds
(vz > 0.5 m/s, lateral > 0.3 m/s, tilt > 10°).

In every seed, the number of touchdowns equalled the number of successes — the
policy never landed off the pad. All failures were episodes in which it did not
commit to a descent and was truncated.

---

## 7. Analysis

**The failure mode changed qualitatively with training budget.** At 500k steps,
80 episodes reached the ground but only 42 landed on the pad — 38 descents onto
open floor, i.e. committing to a descent while misaligned. At 1.5M steps,
**85 touchdowns and 85 successes: zero misses.** The 15 failures are episodes
where the policy never committed at all and was truncated.

That is a much healthier failure mode. It is not descending in the wrong place;
it occasionally fails to find a moment to commit.

**42% was a training-budget limit, not an architecture limit.** Tripling the
steps took success from 42% to 85% with no code change. This is a
methodological point worth carrying forward: before concluding that any future
modification did not help, confirm the budget is sufficient. Stopping at 500k
would have supported the false conclusion that an MLP cannot handle randomised
motion — and might have led to crediting a later architectural change with an
improvement that was really just more steps.

**Success rate alone would not have revealed either bug.** The fixed-motion
policy read 100% while having learned a timed routine rather than a controller.
The 500k randomised policy read 42% with no indication of *why*. The
touchdown-kinematics columns supplied the diagnosis in both cases: at 500k,
lateral speed was 0.206 m/s with a 21.4% violation rate, showing the drone was
arriving at the pad still sliding relative to it.

This is the gap identified in the literature survey. Goldschmid & Ahmad (2024),
TornadoDrone (2024), and Shin et al. (2026) all report binary success rates and
**none reports how hard the drone hit**. Their reported simulation success rates
sit between 97% and 100%, so the metric is saturated and cannot discriminate
between methods — nor detect a policy that has memorised its training
trajectory.

**Randomised motion appears to REDUCE seed variance.** On the fixed-motion task,
two seeds of identical code gave tilt-violation rates of 13% and 0% — a spread
large enough that either run alone would support a different claim. On the
randomised task, three seeds gave 85%, 86%, and 79% success with touchdown
metrics agreeing closely (tilt 2.47 ± 0.09°, lateral 0.068 ± 0.010 m/s).

A plausible explanation: a policy that memorises a schedule can memorise a
*different* schedule in each seed, so outcomes scatter. A policy forced to use
feedback converges toward a similar controller regardless of initialisation.

This does not weaken the case for multi-seed reporting — seed 3 was 6 points
below seed 2, which is well outside episode-level noise — but it does suggest
that variance itself is diagnostic. Unusually high seed variance may indicate a
task in which memorisation is a viable strategy.

---

## 8. Next steps

1. **Multi-seed protocol.** Three training seeds at 1.5M steps; report mean ±
   std across seeds, not only across episodes.
2. **Realistic sensing.** Replace privileged state with the modelled UWB +
   ArUco pipeline including terminal-phase marker dropout. This is the primary
   contribution, and 85% is the baseline it is measured against.
3. **Recurrent policy.** Add a GRU so the network can estimate relative state
   through the blind window; compare against an attention-based variant.
4. **Classical baseline.** Implement and properly tune a cascaded PI controller.
   Goldschmid & Ahmad's PI baseline scored 100% in every simulation scenario, so
   this is not a straw man.
5. **Difficulty breakdown.** Platform parameters are now logged per episode in
   `_computeInfo`, so the 15 failures can be attributed to specific motion
   regimes rather than treated as uniform noise.

---

## 9. Reproducibility

| | Laptop | Lab desktop |
|---|---|---|
| OS | Ubuntu 24.04 | Ubuntu 22.04 |
| Python | 3.12 | 3.12 (deadsnakes PPA) |
| GPU | RTX 5050, 8 GB (sm_120) | NVIDIA T400, 4 GB (sm_75) |
| PyTorch | 2.13.0+cu130 | 2.13.0+cu130 |

Both machines reproduce the fixed-motion step-0 result at 100/100.

Code and full history: `github.com/yashwanth73337/drone-landing-rl` (private).

Training is CPU-bound — the policy is a small MLP and the bottleneck is PyBullet
physics, so GPU capability has little effect on throughput. The second machine
is useful for running experiments in parallel, not for running them faster.
