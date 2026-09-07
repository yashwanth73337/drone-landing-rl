# Landing on a Moving Platform — Experiment Notes

**Project:** RL-based autonomous multirotor landing on a moving platform
**Simulator:** gym-pybullet-drones (PyBullet physics)
**Algorithms:** PPO (stable-baselines3), RecurrentPPO (sb3-contrib)
**Drone:** Crazyflie 2.X (CF2X) — 27 g, thrust-to-weight 2.25, ground effect modelled

---

## 1. Headline result

Four experimental conditions, each on randomised platform motion, 1.5M
training steps, 100 evaluation episodes.

| Condition | Seeds | Success | vz (m/s) | lateral (m/s) | **vz > 0.5** | **lat > 0.3** |
|---|---|---|---|---|---|---|
| Privileged state, MLP | 3 | 83.3 ± 3.1% | 0.047 | 0.068 | 0.0% | **0.0%** |
| Abstract sensing, MLP | 3 | **96.7 ± 0.9%** | 0.162 | 0.270 | 0.0% | **34.9 ± 3.3%** |
| Abstract sensing, LSTM | 1 | 79.0% | 0.302 | 0.209 | 0.0% | **24.1%** |
| Camera + ArUco, MLP | 3 | **45.7 ± 0.5%** | 0.461 | 0.290 | **43.5 ± 30.7%** | **40.7 ± 14.0%** |

### Three findings, none visible in success rate

**1. Degrading sensing RAISED success while creating a safety failure.**
83.3% → 96.7% success, and a 34.9% lateral-violation rate appeared where there
had been none. Under the metric every comparable paper reports, giving the
drone worse sensors made the system better.

**2. An abstract sensing model overstated success by 51 points.**
Hand-set parameters gave 96.7%. A rendered camera with real OpenCV ArUco
detection — same policy, reward, motion, and budget — gave 45.7%.

**3. Success rate is stable across seeds while safety is not.**
The camera condition gave 45%, 46%, 46% success — ±0.5%, near-perfect
reproducibility. The same three seeds gave vertical-violation rates of 0%,
65.2%, and 65.2% — ±30.7%. Same code, same reward, same budget; only the
random seed differs.

> Success rate reports near-perfect agreement across policies that differ by
> 65 percentage points in whether they would damage the airframe.

---

## 2. What was built

Two custom environment classes. Neither exists in gym-pybullet-drones; both
inherit from `BaseRLAviary`, which supplies rotor thrust, drag, ground effect,
and PyBullet stepping.

**`TrackingAviary`** — follow a moving target point. Landing without the
descent.

**`LandingAviary`** — the core environment.

| Component | Description |
|---|---|
| Moving platform | Kinematic box (mass 0) teleported along a sine trajectory. Collision shape so the drone can rest on it; physics cannot push it. |
| Randomised motion | Amplitude, frequency, and phase resampled every episode. See §4.2. |
| Sensing model | Three selectable modes: privileged, abstract, camera. See §5. |
| Reward | Five terms — alignment, velocity matching, gated descent progress, tilt penalty, terminal bonus. See §3. |
| Touchdown detection | Height condition (within 4 cm of pad surface) rather than physics-engine contact reporting, which depends on collision-mesh detail and is not reproducible. |
| Contact metrics | Vertical speed, lateral relative speed, tilt, offset — at the instant of contact. |
| Descent ratchet | The drone may not climb more than 0.10 m above the lowest altitude already reached. Removes hovering as an option. |

---

## 3. Reward function

```
reward = alignment + velocity_matching + progress + tilt_penalty + touchdown_bonus
```

| Term | Form | Purpose |
|---|---|---|
| Alignment | `exp(-3 · horizontal_distance)` | Get over the pad. Horizontal only — vertical distance is what we want closed. |
| Velocity matching | `0.5 · exp(-2 · \|v_drone − v_plat\|)` | Move *with* the platform. Right place at the wrong speed means a sideways impact. |
| Progress | `30 · (prev_height − height)`, gated | Reward *closing the gap*, not *being close*. |
| Tilt penalty | `−0.3 · tilt` | A drone touching down tilted catches a leg and flips. |
| Touchdown bonus | `300 + 800 · softness · centred · level` | Terminal reward, scaled by landing quality. |

**Progress rather than proximity.** Earlier versions rewarded *being near* the
pad. Any such reward creates a place to park: if hovering at some altitude pays
well per step, the policy hovers there forever. A progress term is a function
of *change* — positive descending, negative climbing, exactly zero hovering. It
cannot be farmed by standing still.

**The descent gate.** Progress only pays within 1.5 pad-widths horizontally.
This encodes the *commit-timing* decision, which is what makes landing harder
than tracking: get over the pad first, then descend. Without the gate the
policy dives immediately and lands on open floor.

**The reward uses ground truth, not sensor readings.** The reward defines the
*task*, and the task is to actually land on the pad — not to believe you have.
Rewarding a measured landing would let the policy score points for being
fooled by sensor noise.

---

## 4. Four bugs found

### 4.1 Reward/termination ordering

Eight consecutive reward designs gave 0/100 successful landings. All converged
on the same behaviour: the drone tracked the platform correctly, then hovered
above it until timeout.

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
reward     = self._computeReward()        # ← FIRST
terminated = self._computeTerminated()    # ← calls _checkTouchdown()
```

`_checkTouchdown()` sets `touchdown_recorded = True`, but it lived only inside
`_computeTerminated()` — which runs **after** `_computeReward()`. So on the
step where the drone reached the pad, the reward function still saw the flag as
`False`, skipped the bonus, and returned an ordinary shaping reward. The
episode then terminated, leaving no later step in which the bonus could be
paid.

**The landing bonus was never awarded, at any value.** The agent had no
evidence that landing was worth anything and correctly learned to hover.

**How it was found:** by instrumenting a *scripted descent* — fixed motor
commands, no neural network, printing the reward at each height. This separated
"can the environment produce a landing?" from "is the reward encouraging one?".

### 4.2 Phase memorisation

After the ordering fix, a curriculum reached 100% success. The number was
distrusted and tested.

A frequency sweep gave **non-monotonic** results:

| ω | peak accel | success |
|---|---|---|
| 0.200 | 0.050 | 0% ← *gentler* than trained, still fails |
| 0.500 | 0.125 | 100% ← the trained value |
| 0.800 | 0.200 | 0% |
| 1.200 | 0.300 | 0% |
| 1.800 | 0.450 | 100% ← near a harmonic of the trained rhythm |
| 2.800 | 0.700 | 0% |

A physical limit produces a clean threshold. Success that appears and
disappears with frequency is a **timing artefact**.

**Cause.** With fixed A and ω, and every episode starting at *t* = 0, the
platform's position was fully predictable from the step counter. Memorising a
timed routine scored as well as tracking and was easier to learn. The reported
100% was measuring memorisation.

**Fix.** Amplitude, frequency, and phase resampled every episode. Phase matters
most: without it every episode began with the platform at *x* = 0 moving in
+*x*, itself a memorisable cue. Honest result after the fix: **83.3 ± 3.1%.**

### 4.3 Marker rendering

Three attempts to place an ArUco marker failed, all because PyBullet repeats
textures rather than mapping them once:

1. **Texture on the pad box** — `GEOM_BOX` has no useful UVs; the marker
   stretched across all six faces. The image showed one blown-up black cell.
2. **Texture on a thin box plate** — same problem, still a box.
3. **Texture on `plane.obj`** — that mesh has UVs, but set to repeat, so the
   marker tiled 5×4. Sixteen detections, all id 0, with no way to know which
   tile was the pad centre. Shrinking the mesh packed in more tiles (14×14,
   none decodable) because the UV repeat scales with the mesh.

**Fix:** build the marker from 36 black boxes, one per cell of the 6×6 grid.
No texture, no UV mapping, exact scale.

**A second bug underneath.** `cv2.aruco` reports the corners of the marker's
outer black square; `MARKER_OBJ_PTS` had been set to the inner 4×4 payload,
scaling every recovered distance by 4/6. Detection error was 0.238 m; after the
fix, **0.020 m**.

### 4.4 LSTM training collapse

Three of four recurrent runs failed.

| Run | Config | Outcome |
|---|---|---|
| 1 | defaults | Never learned. `ep_len_mean` pinned at exactly 36.0 across all 75 evaluations. |
| 2 | `n_steps=1024` | Learned but did not converge. Reward peaked 328, drifted to 304. |
| 3 | 3M steps | **Collapsed.** Reward 4.75, episodes 6.6 steps. |
| 4 | stabilised | Stable. 79% success. |

**Run 1 cause.** `RecurrentPPO` defaults to `n_steps=128` against PPO's 2048.
With 4 environments that is 512 steps per update, rarely containing a complete
landing, so the terminal bonus almost never appeared in a rollout.

**Run 3 diagnostics.** `approx_kl` 1.047 (healthy 0.01–0.03), `clip_fraction`
0.777 (healthy 0.1–0.3), action `std` 0.077 (down from 0.482). Each update was
violently rewriting the policy and exploration had stopped.

**Fix.** `learning_rate` 3e-4 → 1e-4, `target_kl` = 0.02, `ent_coef` = 0.005.
Final diagnostics: `approx_kl` 0.014, `clip_fraction` 0.082, `std` 0.84.

---

## 5. The sensing model

Three modes, selected by one constructor argument, which makes them a clean
ablation.

### 5.1 Privileged state

Exact relative position and velocity from the simulator. Noiseless, always
available, not physically realisable. Used to separate the control problem from
the perception problem: if the drone cannot land with perfect information,
adding sensor realism only obscures the cause.

**Observation: +6 values.**

### 5.2 Abstract sensing

Models the *output* of a detection pipeline without rendering.

- **UWB** — σ = 0.15 m per axis, 20 Hz updates, 2% NLOS outliers of ~1 m
- **ArUco surrogate** — σ = 0.02 m, available when height > 0.30 m, tilt < 0.50
  rad, and a 10% frame-drop check passes

**Observation: +8 values** — measured position (3), measured velocity (3), and
two validity flags. The flags matter: without them the policy cannot
distinguish "the platform is right here" from "I cannot see the platform,"
since a stale held measurement looks identical to a fresh one.

### 5.3 Camera + ArUco

A real 128×96 grayscale downward camera, rendered every 2 control steps, with
an 80° vertical FOV. A `DICT_4X4_50` marker, 0.30 m across, built from 36 black
boxes. Detection via `cv2.aruco.detectMarkers`, pose via `cv2.solvePnP`,
rotated from camera coordinates into world axes. UWB fallback when detection
fails.

**Verified before training:**

| Check | Result |
|---|---|
| Detection error vs ground truth | **0.020 m** |
| Sign and tracking of measured vs true x | Correct |
| Dropout height, measured | 0.195 m → 0.123 m |
| Dropout height, predicted `s/(2·tan(FOV/2))` | **0.18 m** |
| Throughput | 197 steps/sec (34× slower than state-based) |

The dropout height is **derived, then verified** — not assumed. A reviewer can
check the arithmetic.

**No Kalman filter, deliberately.** Shin et al. (RA-L 2026) showed an EKF
*diverges* during marker dropout because it falls back on a constant-velocity
model. A learned temporal estimator degrades gracefully. Passing raw
measurements plus flags leaves the estimation to the network.

**Velocity is differentiated** from successive position measurements, since
neither a camera nor a UWB anchor reports velocity. That amplifies noise, which
is realistic and part of what makes the task hard. A `nan_to_num` guard and a
±10 m/s clip were required after an early run crashed with NaN in the
observation when two measurements arrived in the same step.

---

## 6. Results in detail

### 6.1 Privileged state, MLP — 3 seeds

| Seed | Success | vz | lat | tilt | offset |
|---|---|---|---|---|---|
| 1 | 85% | 0.041 | 0.075 | 2.49° | 0.045 |
| 2 | 86% | 0.056 | 0.075 | 2.57° | 0.054 |
| 3 | 79% | 0.045 | 0.053 | 2.36° | 0.032 |
| **Mean** | **83.3 ± 3.1%** | **0.047 ± 0.006** | **0.068 ± 0.010** | **2.47 ± 0.09°** | **0.044 ± 0.009** |

Zero violations in all three seeds. In every seed the number of touchdowns
equalled the number of successes — the policy never landed off the pad. All
failures were episodes where it did not commit and was truncated.

Budget note: at seed 1, 500k steps gave 42% success with 21.4% lateral
violations; 1.5M gave 85% with 0%. **The 42% was a budget limit, not an
architecture limit.**

### 6.2 Abstract sensing, MLP — 3 seeds

| Seed | Success | vz | lat | tilt | lat > 0.3 |
|---|---|---|---|---|---|
| 1 | 96% | 0.194 | 0.298 | 3.05° | 39.6% |
| 2 | 96% | 0.128 | 0.238 | 4.13° | 32.3% |
| 3 | 98% | 0.165 | 0.273 | 3.75° | 32.7% |
| **Mean** | **96.7 ± 0.9%** | **0.162 ± 0.027** | **0.270 ± 0.025** | **3.64 ± 0.44°** | **34.9 ± 3.3%** |

### 6.3 Abstract sensing, LSTM — 1 seed

79% success, vz 0.302, lateral 0.209, tilt 4.57°, offset 0.122.
Violations: vz 0%, lateral 24.1%, tilt 2.5%.

### 6.4 Camera + ArUco, MLP — 3 seeds

| Seed | Success | vz | lat | tilt | **vz > 0.5** | **lat > 0.3** |
|---|---|---|---|---|---|---|
| 1 | 45% | 0.299 | 0.230 | 1.93° | **0.0%** | 24.4% |
| 2 | 46% | 0.531 | 0.386 | 4.04° | **65.2%** | 58.7% |
| 3 | 46% | 0.552 | 0.255 | 2.66° | **65.2%** | 39.1% |
| **Mean** | **45.7 ± 0.5%** | **0.461 ± 0.114** | **0.290 ± 0.066** | **2.88 ± 0.87°** | **43.5 ± 30.7%** | **40.7 ± 14.0%** |

---

## 7. Analysis

### 7.1 Worse sensing improved the success rate

83.3% → 96.7%. With perfect information the policy could hover, refine
alignment, and descend when conditions were ideal. With noisy information that
cuts out near the pad, waiting does not help — more time means more noise and
more chance of losing the marker — so committing early genuinely maximises
reward. It lands more often and lands harder.

**The reward pays for *landing*. It does not sufficiently punish landing
*badly*.** The policy found that out.

### 7.2 The abstract sensing model was optimistic by 51 points

96.7% → 45.7% on replacing modelled sensing with a rendered camera.

The mechanism is measurable. The abstract model assumed a ~30% blind fraction
from a vertical-only FOV threshold. Measured blind fraction with a real camera
was **~88%**, because with a moving platform the drone drifts up to 0.7 m
laterally while descending, and the marker leaves the frame *sideways* long
before the height threshold matters. The model accounted for only one of two
dropout mechanisms.

This is a claim about methodology, not only about this drone: abstract sensor
models should be validated against a rendered pipeline before conclusions are
drawn from them. None of the three comparison papers does this — Shin et al.
render images, Goldschmid and TornadoDrone use motion capture, and none
compares an abstract model against a rendered one on the same task.

### 7.3 Success rate is stable across seeds; safety is not

The sharpest result in the project.

| Camera seed | Success | vz > 0.5 m/s |
|---|---|---|
| 1 | 45% | 0.0% |
| 2 | 46% | 65.2% |
| 3 | 46% | 65.2% |
| **Spread** | **±0.5%** | **±30.7%** |

Same code, same reward, same training budget. Only the random seed differs.

A reviewer looking at success rate would call this highly reproducible and move
on. One of these policies never exceeds the touchdown-velocity limit; two
exceed it in roughly two landings out of three.

This also sharpens the multi-seed argument beyond the usual "results vary."
**Seed variance is not uniform noise — it is concentrated entirely in the
dimension nobody measures.**

### 7.4 Memory addresses the specific failure it was introduced for

The LSTM cut lateral violations 34.9% → 24.1%, a 31% relative reduction, at a
cost of 18 points of success. `ep_len_mean` was 86.5 against the MLP's ~49: the
recurrent policy spends far longer positioning before committing, which is what
better horizontal precision looks like.

A trade-off, not a strict loss — and only visible because touchdown metrics are
reported. Under success rate alone the LSTM simply looks worse.

### 7.5 Recurrent policies are markedly harder to train

Three of four runs failed or collapsed. Default hyperparameters did not learn
at all; the corrected configuration collapsed when given more steps; only
explicit KL constraint and a reduced learning rate produced a stable run.
Consistent with the known instability of recurrent architectures in RL — the
reason GTrXL exists for transformers — and worth stating as a cost of the
approach.

### 7.6 Randomised motion reduced seed variance in the state-based conditions

On the fixed-motion task, two identical-code seeds gave tilt-violation rates of
13% and 0%. On the randomised task with privileged state, three seeds agreed to
2.47 ± 0.09° on tilt. A policy that memorises a schedule can memorise a
*different* one per seed; a policy forced to use feedback converges toward a
similar controller.

Note this does **not** hold for the camera condition, where safety variance is
enormous — a different mechanism, driven by intermittent perception rather than
memorisation.

### 7.7 Failures cluster at high platform speed

Failed episodes averaged 0.73–0.90 m/s peak platform speed against a
distribution mean near 0.5, consistently across all seeds and conditions.
Failures are a capability limit, not noise.

---

## 8. Naming correction

Early planning notes referred to a GRU. The implementation is an **LSTM** —
`sb3-contrib`'s `RecurrentPPO` provides `MlpLstmPolicy` only, with no GRU
option. Both are gated recurrent networks serving the same purpose, but reports
and slides should say LSTM.

---

## 9. Next steps

1. **Two more LSTM seeds** at the stabilised configuration, for error bars
   matching the other conditions.

2. **Reward ablation.** The most important remaining experiment. A reviewer
   will say: *"success went up with worse sensors because your reward is badly
   shaped."* Varying the touchdown-bonus weights across three settings and
   showing the divergence persists turns an anecdote into a finding. If it does
   not persist, that is worth knowing first.

3. **Classical baseline.** A properly tuned cascaded PI controller.
   Goldschmid & Ahmad's PI scored 100% in every simulation scenario, so a weak
   baseline would be spotted immediately.

4. **Constrained RL (PPO-Lagrangian).** The natural conclusion of this work.
   Instead of hand-balancing reward weights, state the requirement directly:
   *maximise success subject to touchdown velocity ≤ 0.5 m/s.* A Lagrange
   multiplier tunes itself. Nine reward iterations here were spent doing that
   balancing manually, which is the evidence for why it is needed. §7.3
   strengthens the case further: a constraint would have bound seeds 2 and 3
   where a fixed weight did not.

5. **ArduPilot SITL validation.** Zero-shot transfer from PyBullet to Gazebo
   running production firmware. Both installed and verified.

---

## 10. Reproducibility

| | Laptop | Lab desktop |
|---|---|---|
| OS | Ubuntu 24.04 | Ubuntu 22.04 |
| Python | 3.12 | 3.12 (deadsnakes PPA) |
| GPU | RTX 5050, 8 GB (sm_120) | NVIDIA T400, 4 GB (sm_75) |
| PyTorch | 2.13.0+cu130 | 2.13.0+cu130 |
| OpenCV | opencv-contrib-python (needed for `cv2.aruco`) | same |

Evaluation is deterministic: re-running the three privileged-state seeds
reproduced 85 / 86 / 79 exactly.

Every evaluation appends a row to `results_log.csv` — timestamp, model path,
note, all metrics, and the platform motion of the failed episodes.

Training is CPU-bound. The policy is a small MLP and the bottleneck is PyBullet
physics, so GPU capability has little effect. Throughput: MLP state-based
~6700 steps/sec; MLP with camera ~197; LSTM state-based ~180.

Code and full history: `github.com/yashwanth73337/drone-landing-rl` (private).
