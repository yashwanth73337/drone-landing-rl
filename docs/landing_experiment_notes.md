# Landing on a Moving Platform — Experiment Notes

**Project:** RL-based autonomous multirotor landing on a moving platform
**Simulator:** gym-pybullet-drones (PyBullet physics)
**Algorithm:** PPO (stable-baselines3) and RecurrentPPO (sb3-contrib)
**Drone:** Crazyflie 2.X (CF2X) — 27 g, thrust-to-weight 2.25, ground effect modelled

---

## 1. The central result

Three experimental conditions, each evaluated over 100 episodes on randomised
platform motion:

| Condition | Success | Lateral speed at touchdown | **lat > 0.3 m/s violations** |
|---|---|---|---|
| Privileged state (perfect sensing), MLP | 83.3 ± 3.1% | 0.068 m/s | **0.0%** |
| Realistic sensing, MLP | **96.7 ± 0.9%** | 0.270 m/s | **34.9%** |
| Realistic sensing, LSTM | 79.0% | 0.209 m/s | **24.1%** |

Two findings, and neither is visible in success rate alone.

**Degrading the sensing raised success by 13 points while introducing a 35%
safety-violation rate that did not previously exist.** Under the metric every
comparable paper reports, giving the drone worse sensors *improved* the system.
Under touchdown-kinematics metrics, one landing in three now arrives sliding
sideways fast enough to tumble a real airframe.

**Adding memory cut those violations by 31% relative** (34.9% → 24.1%), at a
cost of 18 percentage points of success rate. Under success rate alone the
recurrent policy simply looks worse, with no indication that it addressed the
safety problem it was introduced to solve.

---

## 2. What was built

Two custom environment classes. Neither exists in gym-pybullet-drones; both
inherit from the library's `BaseRLAviary`, which supplies the physics — rotor
thrust, drag, ground effect, PyBullet stepping.

**`TrackingAviary`** — the drone follows a moving target point. An intermediate
task: landing without the descent.

**`LandingAviary`** — the core environment.

| Component | Description |
|---|---|
| Moving platform | Kinematic box (mass 0) teleported along a sine trajectory each step. Collision shape so the drone can rest on it; physics cannot push it. |
| Randomised motion | Amplitude, frequency, and phase resampled every episode. See §4. |
| Sensing model | UWB + ArUco with realistic noise and terminal dropout. See §5. |
| Reward | Five terms — alignment, velocity matching, gated descent progress, tilt penalty, terminal bonus. See §3. |
| Touchdown detection | Height condition (within 4 cm of pad surface) rather than physics-engine contact reporting, which depends on collision-mesh detail and is not reproducible. |
| Contact metrics | Vertical speed, lateral relative speed, tilt, offset — recorded at the instant of contact. |
| Descent ratchet | The drone may not climb more than a fixed slack above the lowest altitude already reached. Removes indefinite hovering as an option. |

---

## 3. Reward function

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

**Progress rather than proximity.** Earlier versions rewarded *being near* the
pad. Any such reward creates a place to park: if hovering at some altitude pays
well per step, the policy hovers there forever. A progress term is a function of
*change* — positive descending, negative climbing, exactly zero hovering. It
cannot be farmed by standing still.

**The descent gate.** Progress only pays while horizontally within 1.5
pad-widths. This encodes the *commit-timing* decision, which is what makes
landing harder than tracking: the drone must position over the pad **first**,
then descend. Without the gate the policy dives immediately and lands on open
floor.

**The reward uses ground truth, not sensor readings.** This is deliberate: the
reward defines the *task*, and the task is to land on the pad, not to believe
you have. Rewarding a measured landing would let the policy score points for
being fooled by sensor noise.

---

## 4. Two bugs found

### Bug 1 — reward/termination ordering

Eight consecutive reward designs produced 0/100 successful landings. All
converged on the same behaviour: the drone tracked the platform correctly, then
hovered above it until the episode timed out.

| Version | Change | Outcome |
|---|---|---|
| v1 | bonus 100, time penalty −0.5 | Hovered the full episode. |
| v2 | bonus 600, descent reward ×3 | Hovered, at a different altitude. |
| v3 | time penalty −2.0 | Self-terminated within 4 steps. |
| v4 | progress term, bonus 200 | Hovered at 0.26 m, centred to 5 mm. |
| v4b | bonus 800, alignment weight cut | Still hovered; tracking degraded. |
| v5 | descent ratchet, slack 0.15 | Oscillated between 0.17 m and 0.32 m. |
| v5b | slack 0.05, flat +300 bonus | Still hovered. |
| **v6** | **ordering bug fixed** | **100/100 successful landings.** |

`BaseAviary.step()` evaluates in this order:

```python
obs        = self._computeObs()
reward     = self._computeReward()        # ← first
terminated = self._computeTerminated()    # ← calls _checkTouchdown()
```

`_checkTouchdown()` sets `touchdown_recorded = True`, but was only called from
`_computeTerminated()` — which runs **after** `_computeReward()`. So on the step
where the drone reached the pad, the reward function still saw
`touchdown_recorded == False`, skipped the bonus block, and returned an ordinary
shaping reward. The episode then terminated, leaving no later step in which the
bonus could be paid.

**The landing bonus was never awarded, at any value.** The agent had no evidence
that landing was worth anything and correctly learned to hover.

**How it was found:** by instrumenting a *scripted descent* — flying the drone
down with fixed motor commands, no neural network involved, printing the reward
at each height. This separated *"can the environment produce a landing?"* from
*"is the reward encouraging one?"*. Eight rounds of reward tuning had been spent
on a problem that was never in the reward.

### Bug 2 — phase memorisation

After the ordering fix, a four-stage curriculum reached **100% success**. That
number was an artefact.

Sweeping the trained policy over platform parameters gave contradictory results
at the same peak speed:

| Peak speed (m/s) | Frequency sweep (A=0.5) | Amplitude sweep (ω=0.5) |
|---|---|---|
| 0.250 | 100% | 100% |
| 0.375 | **26%** | **100%** |
| 0.500 | **0%** | **92%** |
| 0.750 | 0% | 0% |

A follow-up test decoupled speed from acceleration by solving for the *A* and
*ω* producing any desired pair (peak speed = *Aω*, peak acceleration = *Aω²*).
It rejected an acceleration limit too, because success was **non-monotonic** in
both quantities:

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
the platform's position was fully predictable from the step counter. Nothing
forced the policy to use its observation — memorising a timed routine ("at step
60, move here") scored equally well and was easier to learn. The reported 100%
was measuring memorisation, not tracking.

**Fix.** Amplitude, frequency, and phase are resampled every episode. Phase
matters most: without it every episode began with the platform at *x* = 0 moving
in +*x*, itself a memorisable cue.

Shin et al. (RA-L 2026) randomise platform motion during training for exactly
this reason.

---

## 5. The sensing model

Every version up to this point handed the policy **privileged state** — exact
relative position and velocity, read straight from the simulator. A real drone
cannot measure that.

The model reflects the lab's actual stack:

**UWB (ultra-wideband ranging)** — works at any range and attitude, but noisy
(σ ≈ 0.15 m per axis), updates at ~20 Hz, and throws NLOS outliers at 2%.

**ArUco marker + downward camera** — accurate to ~2 cm, but only while the
marker is in frame.

The critical detail is geometric. A downward camera at height *h* sees a ground
footprint of roughly 2*h*, so a marker of side *s* leaves the frame at about
*h ≈ s*. **The drone is blind for the final ~30 cm of every landing** — exactly
when precision matters most. The marker is also lost above 0.5 rad of tilt, and
the detector drops 10% of frames.

The observation grows from 6 numbers to 8: measured relative position (3),
measured relative velocity (3), and two validity flags. The flags matter —
without them the policy cannot distinguish "the platform is right here" from
"I cannot see the platform", since a stale held measurement looks identical to
a fresh one.

**No Kalman filter, deliberately.** Shin et al. showed an EKF *diverges* during
marker dropout because it falls back on a constant-velocity model, whereas a
learned temporal estimator degrades gracefully. Handing the policy raw
measurements plus flags leaves the estimation problem to the network — which is
what the recurrent policy is meant to solve.

---

## 6. Results

### 6.1 Fixed-motion curriculum (superseded — see §4, Bug 2)

Retained for the record. These numbers reflect memorisation, not tracking.

| Step | Start height | Pad half-width | Amplitude | Success | vz | lat | tilt | offset |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.4 m | 0.50 m | 0.0 | 100% | 0.184 | 0.088 | 5.22° | 0.102 |
| 1 | 0.8 m | 0.50 m | 0.0 | 100% | 0.187 | 0.034 | 0.09° | 0.065 |
| 2 | 0.8 m | 0.20 m | 0.0 | 100% | 0.187 | 0.029 | 0.12° | 0.095 |
| 3 | 0.8 m | 0.20 m | 0.5 | 100% | 0.091 | 0.076 | 0.51° | 0.039 |

### 6.2 Randomised motion, privileged state, MLP

Amplitude ∈ [0.2, 1.0] m, ω ∈ [0.2, 1.5] rad/s, phase ∈ [0, 2π), resampled per
episode. Three independent training seeds, 1.5M steps each.

| Seed | Success | vz (m/s) | lat (m/s) | tilt (°) | offset (m) |
|---|---|---|---|---|---|
| 1 | 85% | 0.041 | 0.075 | 2.49 | 0.045 |
| 2 | 86% | 0.056 | 0.075 | 2.57 | 0.054 |
| 3 | 79% | 0.045 | 0.053 | 2.36 | 0.032 |
| **Across seeds** | **83.3 ± 3.1%** | **0.047 ± 0.006** | **0.068 ± 0.010** | **2.47 ± 0.09** | **0.044 ± 0.009** |

Zero constraint violations in all three seeds. In every seed the number of
touchdowns equalled the number of successes — the policy never landed off the
pad. All failures were episodes where it did not commit and was truncated.

A training-budget comparison at seed 1: 500k steps gave 42% success with 21.4%
lateral violations; 1.5M gave 85% with 0%. **The 42% was a budget limit, not an
architecture limit.**

### 6.3 Randomised motion, realistic sensing, MLP

Three seeds, 1.5M steps each.

| Seed | Success | vz (m/s) | lat (m/s) | tilt (°) | lat > 0.3 |
|---|---|---|---|---|---|
| 1 | 96% | 0.194 | 0.298 | 3.05 | 39.6% |
| 2 | 96% | 0.128 | 0.238 | 4.13 | 32.3% |
| 3 | 98% | 0.165 | 0.273 | 3.75 | 32.7% |
| **Across seeds** | **96.7 ± 0.9%** | **0.162 ± 0.027** | **0.270 ± 0.025** | **3.64 ± 0.44** | **34.9 ± 3.3%** |

### 6.4 Randomised motion, realistic sensing, LSTM

`RecurrentPPO` with `MlpLstmPolicy`, 256 hidden units, one layer. Same
environment, reward, sensor model, and evaluation protocol as §6.3.

| Run | Config | Steps | Success | vz | lat | lat > 0.3 | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | defaults | 1.5M | — | — | — | — | Never learned. See below. |
| 2 | n_steps=1024 | 1.5M | 68% | 0.342 | 0.194 | 26.5% | Learned, did not converge. |
| 3 | n_steps=1024 | 3M | — | — | — | — | Collapsed. See below. |
| **4** | **stabilised** | **2M** | **79%** | **0.302** | **0.209** | **24.1%** | **Stable.** |

**Run 1 failed completely.** `ep_len_mean` was pinned at exactly 36.0 across all
75 evaluations — the drone free-fell every episode. Cause: `RecurrentPPO`
defaults to `n_steps=128` against PPO's 2048. With 4 environments that is 512
steps per update, which rarely contains a complete landing, so the terminal
bonus almost never appeared in a rollout and there was no learning signal.

**Run 3 collapsed.** Reward fell from 304 to 4.75 and episodes to 6.6 steps.
Diagnostics: `approx_kl` 1.047 (healthy ~0.01–0.03), `clip_fraction` 0.777
(healthy ~0.1–0.3), action `std` 0.077 (down from 0.482). Each update was
violently rewriting the policy, and exploration had stopped entirely. This is
policy collapse, not under-training — run 2 was likely already mid-decline
rather than plateaued.

**Run 4 stabilised it.** `learning_rate` 3e-4 → 1e-4, `target_kl` = 0.02 (aborts
any update moving the policy too far), `ent_coef` = 0.005 (prevents the action
distribution collapsing). Final diagnostics: `approx_kl` 0.014,
`clip_fraction` 0.082, `std` 0.84 — all healthy.

---

## 7. Analysis

**Worse sensing improved the success rate.** 83.3% → 96.7%. With perfect
information the policy could hover, refine its alignment, and descend when
conditions were ideal. With noisy information that cuts out near the pad,
waiting does not help — more time means more noise and more chance of losing
the marker — so committing early genuinely maximises reward. The reward pays for
*landing*; it does not sufficiently punish landing *badly*. The policy found
that out.

**This is the gap in the literature, demonstrated in our own data.**
Goldschmid & Ahmad (2024), TornadoDrone (2024), and Shin et al. (2026) all
report binary success rates and none reports how hard the drone hit. Their
simulation success rates sit between 97% and 100%, so the metric is saturated.
Under that metric, degrading our sensing made the system better by 13 points.
Under touchdown metrics, it introduced a 35% safety-violation rate.

**Memory addressed the specific failure it was introduced for.** Lateral
violations fell 34.9% → 24.1%, a 31% relative reduction. `ep_len_mean` was 86.5
for the LSTM against ~49 for the MLP: the recurrent policy spends far longer
positioning before committing, which is what better horizontal precision looks
like.

**But it cost 18 points of success and nearly doubled vertical touchdown
speed.** This is a trade-off, not a strict loss — better horizontal precision,
worse commitment and worse vertical control. It is only visible because
touchdown metrics are reported; under success rate alone the LSTM simply looks
worse.

**Recurrent policies are markedly harder to train.** Three of four runs failed
or collapsed. Default hyperparameters did not learn at all; the corrected
configuration collapsed when given more steps; only explicit KL constraint and
a reduced learning rate produced a stable run. This is consistent with the known
instability of recurrent architectures in RL — the reason GTrXL exists for
transformers — and is worth stating as a cost of the approach.

**Training budget must be ruled out before blaming architecture.** The MLP went
from 42% to 85% purely from tripling its budget, with no code change. Stopping
at 500k would have supported the false conclusion that an MLP cannot handle
randomised motion.

**Randomised motion reduced seed variance.** On the fixed-motion task, two
identical-code seeds gave tilt-violation rates of 13% and 0%. On the randomised
task, three seeds agreed to 2.47 ± 0.09° on tilt. A policy that memorises a
schedule can memorise a *different* schedule per seed, so outcomes scatter; a
policy forced to use feedback converges toward a similar controller. Unusually
high seed variance may itself be diagnostic of a memorisable task.

---

## 8. Naming correction

Earlier planning notes referred to a GRU. The implementation is an **LSTM** —
`sb3-contrib`'s `RecurrentPPO` provides `MlpLstmPolicy` only, with no GRU
option. Both are gated recurrent networks serving the same purpose here, but
reports and slides should say LSTM.

---

## 9. Next steps

1. **Two more LSTM seeds** at the stabilised configuration, for error bars
   matching the MLP conditions.
2. **Classical baseline.** A properly tuned cascaded PI controller.
   Goldschmid & Ahmad's PI baseline scored 100% in every simulation scenario,
   so this is not a straw man.
3. **Constrained RL.** The trade-off in §7 is exactly what PPO-Lagrangian is
   for: maximise success *subject to* a hard touchdown-velocity constraint,
   rather than hand-balancing reward weights. Nine reward iterations in this
   project were spent doing that balancing manually.
4. **Difficulty breakdown.** Platform parameters are logged per episode, so
   failures can be attributed to motion regimes. Failed episodes average
   ~0.7–0.9 m/s peak speed against a distribution mean near 0.5, so failures
   cluster at the fast end rather than scattering.

---

## 10. Reproducibility

| | Laptop | Lab desktop |
|---|---|---|
| OS | Ubuntu 24.04 | Ubuntu 22.04 |
| Python | 3.12 | 3.12 (deadsnakes PPA) |
| GPU | RTX 5050, 8 GB (sm_120) | NVIDIA T400, 4 GB (sm_75) |
| PyTorch | 2.13.0+cu130 | 2.13.0+cu130 |

Both machines reproduce the fixed-motion baseline at 100/100. Evaluation is
deterministic: re-running the three MLP seeds reproduced 85 / 86 / 79 exactly.

Every evaluation is appended to `results_log.csv`. Code and full history:
`github.com/yashwanth73337/drone-landing-rl` (private).

Training is CPU-bound — the policy is a small MLP and the bottleneck is PyBullet
physics, so GPU capability has little effect on throughput. MLP training runs at
~1100 fps; the LSTM at ~180 fps, roughly 6× slower, because sequences cannot be
shuffled.
