# Landing on a Moving Platform — Experiment Notes

**Project:** RL-based autonomous multirotor landing on a moving platform
**Simulator:** gym-pybullet-drones (PyBullet physics)
**Algorithms:** PPO (stable-baselines3), RecurrentPPO (sb3-contrib)
**Drone:** Crazyflie 2.X (CF2X) — 27 g, thrust-to-weight 2.25, ground effect modelled

---

## 1. Headline result

Five experimental conditions, each on randomised platform motion, 1.5M
training steps, 100 evaluation episodes.

| Condition | Seeds | Success | vz (m/s) | lateral (m/s) | **vz > 0.5** | **lat > 0.3** |
|---|---|---|---|---|---|---|
| Privileged state, MLP | 3 | 83.3 ± 3.1% | 0.047 | 0.068 | 0.0% | **0.0%** |
| Abstract sensing, MLP | 3 | **96.7 ± 0.9%** | 0.162 | 0.270 | 0.0% | **34.9 ± 3.3%** |
| Abstract sensing, LSTM | 1 | 79.0% | 0.302 | 0.209 | 0.0% | **24.1%** |
| Camera + ArUco, MLP | 3 | 45.7 ± 0.5% | 0.461 | 0.290 | **43.5 ± 30.7%** | **40.7 ± 14.0%** |
| Camera + ArUco, MLP, **descent-capped** | 3 | 43.0 ± 9.1% | **0.301 ± 0.145** | 0.247 | **4.1 ± 4.0%** | 24.9 ± 3.7% |

### Four findings, none visible in success rate

**1. Degrading sensing RAISED success while creating a safety failure.**
83.3% → 96.7% success, and a 34.9% lateral-violation rate appeared where there
had been none.

**2. An abstract sensing model overstated success by 51 points.**
Hand-set parameters gave 96.7%. A rendered camera with real OpenCV ArUco
detection gave 45.7%.

**3. Success rate is stable across seeds while safety is not.**
The camera condition gave 45%, 46%, 46% success (±0.5%). The same three seeds
gave vertical-violation rates of 0%, 65.2%, and 65.2% (±30.7%).

**4. A hand-tuned fix for one failure mode redistributes the risk rather than
removing it.**
Capping descent speed cut vertical violations from 43.5% to 4.1% and shrank
their seed variance from ±30.7 to ±4.0 — a real, reproducible improvement. But
it left success-rate variance *worse* (±0.5% → ±9.1%) and lateral violations
only partially addressed (40.7% → 24.9%). See §7.4.

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
| Reward | Five terms plus a descent-rate cap. See §3. |
| Touchdown detection | Height condition (within 4 cm of pad surface) rather than physics-engine contact reporting, which depends on collision-mesh detail and is not reproducible. |
| Contact metrics | Vertical speed, lateral relative speed, tilt, offset — at the instant of contact. |
| Descent ratchet | The drone may not climb more than 0.10 m above the lowest altitude already reached. Removes hovering as an option. |

---

## 3. Reward function

```
reward = alignment + velocity_matching + progress + descent_penalty + tilt_penalty + touchdown_bonus
```

| Term | Form | Purpose |
|---|---|---|
| Alignment | `exp(-3 · horizontal_distance)` | Get over the pad. Horizontal only — vertical distance is what we want closed. |
| Velocity matching | `0.5 · exp(-2 · \|v_drone − v_plat\|)` | Move *with* the platform. |
| Progress | `30 · min(v_desc, 0.35) · dt`, gated | Reward closing the gap, capped in speed. See §3.2. |
| Descent penalty | `−4.0 · max(0, v_desc − 0.35) · dt` | Cost for exceeding the cap. |
| Tilt penalty | `−0.3 · tilt` | A drone touching down tilted catches a leg and flips. |
| Touchdown bonus | `300 + 800 · softness · centred · level` | Terminal reward, scaled by landing quality. |

### 3.1 Progress rather than proximity

Earlier versions rewarded *being near* the pad. Any such reward creates a
place to park: if hovering at some altitude pays well per step, the policy
hovers there forever. A progress term is a function of *change* — positive
descending, negative climbing, exactly zero hovering. It cannot be farmed by
standing still.

### 3.2 The descent-rate cap

**Why it was added.** The camera-sensing condition showed a 43.5% rate of
touchdown-velocity violations. The original progress term paid a flat rate per
metre closed, `30 · (prev_height − height)`, regardless of the speed at which
that metre was closed. Descending twice as fast earned the same total reward,
but collected it sooner — and sooner is worth more under discounting
(γ = 0.99). The reward therefore preferred a dive to a controlled descent. The
touchdown bonus's softness factor was one terminal payment competing against
this per-step incentive, and the per-step term won.

**The fix.** Only descent up to 0.35 m/s earns progress reward; descent beyond
that is explicitly penalised. The cap is set below the 0.5 m/s safety
threshold because the final centimetres are ballistic once thrust is reduced,
so touchdown speed lands somewhat above the commanded descent rate.

**This is a hand-tuned patch, not the principled fix.** See §7.4 and §9.

### 3.3 The descent gate

Progress only pays within 1.5 pad-widths horizontally. This encodes the
*commit-timing* decision: get over the pad first, then descend. Without the
gate the policy dives immediately from wherever it happens to be.

### 3.4 The reward uses ground truth, not sensor readings

The reward defines the *task*, and the task is to actually land on the pad —
not to believe you have. Rewarding a measured landing would let the policy
score points for being fooled by sensor noise.

---

## 4. Four bugs found

### 4.1 Reward/termination ordering

Eight consecutive reward designs gave 0/100 successful landings. All converged
on hovering above the platform until timeout.

`BaseAviary.step()` calls `_computeReward()` **before** `_computeTerminated()`,
where `_checkTouchdown()` — which sets `touchdown_recorded` — originally
lived. On the touchdown step, the reward function saw the flag as `False`,
skipped the bonus, and the episode then ended. **The landing bonus was never
awarded, at any value.**

Found by instrumenting a *scripted descent* — fixed motor commands, no
network, printing reward against height. Fixed by having `_computeReward()`
call `_checkTouchdown()` itself.

### 4.2 Phase memorisation

A curriculum reached 100% success. A frequency sweep gave **non-monotonic**
results — 0% at ω=0.2, 100% at 0.5 (the trained value), 0% at 0.8 and 1.2, and
100% again at 1.8 (near a harmonic of the trained rhythm). A physical limit
produces a clean threshold; this was a timing artefact. With fixed motion
starting at t=0, the platform's position was predictable from the step counter
alone, and the policy had memorised a schedule.

Fixed by resampling amplitude, frequency, and phase every episode. Honest
result: **83.3 ± 3.1%.**

### 4.3 Marker rendering

Three attempts to place an ArUco marker failed because PyBullet repeats
textures rather than mapping them once: a textured box stretched the marker
across all six faces; a textured plate had the same problem; `plane.obj` has
UVs but they repeat, tiling the marker 5×4.

Fixed by building the marker from 36 black boxes, one per cell of the 6×6
grid — geometry has no UV mapping, so nothing can tile.

### 4.4 Marker corner scale

`cv2.aruco` reports the corners of the marker's outer black square;
`MARKER_OBJ_PTS` had been set to the inner 4×4 payload, scaling every
recovered distance by 4/6. Detection error was 0.238 m; after the fix,
**0.020 m**. Caught by `verify_camera.py` before any training run.

### 4.5 LSTM training collapse

Three of four recurrent runs failed. Defaults (`n_steps=128`) never learned —
`ep_len_mean` pinned at exactly 36.0 across 75 evaluations, since with 4
environments that gave 512 steps per update, rarely containing a complete
landing. A 3M-step run collapsed: `approx_kl` reached 1.047 (healthy
0.01–0.03), `std` fell to 0.077. Fixed with `learning_rate=1e-4`,
`target_kl=0.02`, `ent_coef=0.005`. Final: 79% success.

---

## 5. The sensing model

Three modes, selected by one constructor argument.

### 5.1 Privileged state

Exact relative position and velocity from the simulator. Not physically
realisable; isolates the control problem from perception. **+6 observation
values.**

### 5.2 Abstract sensing

Models sensing *output* without rendering: UWB (σ=0.15 m, 20 Hz, 2% NLOS
outliers) plus an ArUco surrogate (σ=0.02 m, available above 0.30 m height,
below 0.50 rad tilt, 10% random dropout). **+8 values** — measured position,
measured velocity, two validity flags. The flags matter: without them the
policy cannot distinguish "the platform is right here" from "I cannot see the
platform."

### 5.3 Camera + ArUco

A real 128×96 grayscale downward camera, 80° FOV, rendered every 2 control
steps. A `DICT_4X4_50` marker, 0.30 m, built from 36 black boxes. Detection via
`cv2.aruco.detectMarkers`, pose via `cv2.solvePnP`, rotated into world axes.
UWB fallback on detection failure.

**Verified before training:**

| Check | Result |
|---|---|
| Detection error vs ground truth | 0.020 m |
| Dropout height, measured | 0.195 m → 0.123 m |
| Dropout height, predicted `s/(2·tan(FOV/2))` | 0.18 m |
| Throughput | 197 steps/sec (34× slower than state-based) |

**No Kalman filter, deliberately.** Shin et al. (RA-L 2026) showed an EKF
diverges during marker dropout, falling back on a constant-velocity model. A
learned temporal estimator degrades gracefully instead.

---

## 6. Results in detail

### 6.1 Privileged state, MLP — 3 seeds

83.3 ± 3.1% success. vz 0.047 ± 0.006, lateral 0.068 ± 0.010, tilt
2.47 ± 0.09°. Zero violations in all three seeds.

### 6.2 Abstract sensing, MLP — 3 seeds

96.7 ± 0.9% success. vz 0.162 ± 0.027, lateral 0.270 ± 0.025, tilt
3.64 ± 0.44°. Lateral violations 34.9 ± 3.3%.

### 6.3 Abstract sensing, LSTM — 1 seed

79.0% success. vz 0.302, lateral 0.209, tilt 4.57°. Lateral violations 24.1%.

### 6.4 Camera + ArUco, MLP — 3 seeds

| Seed | Success | vz | lat | vz > 0.5 | lat > 0.3 |
|---|---|---|---|---|---|
| 1 | 45% | 0.299 | 0.230 | 0.0% | 24.4% |
| 2 | 46% | 0.531 | 0.386 | 65.2% | 58.7% |
| 3 | 46% | 0.552 | 0.255 | 65.2% | 39.1% |
| **Mean** | **45.7 ± 0.5%** | **0.461 ± 0.114** | **0.290 ± 0.066** | **43.5 ± 30.7%** | **40.7 ± 14.0%** |

### 6.5 Camera + ArUco, MLP, descent-capped — 3 seeds

Same environment and camera pipeline; reward's progress term capped at
0.35 m/s with an explicit penalty above it (§3.2).

| Seed | Touchdowns | Success | vz | lat | vz > 0.5 | lat > 0.3 |
|---|---|---|---|---|---|---|
| 1 | 94 | 51% | 0.395 | 0.215 | 7.8% | 21.6% |
| 2 | 89 | 33% | 0.128 | 0.292 | 0.0% | 24.2% |
| 3 | 99 | 45% | 0.379 | 0.234 | 4.4% | 28.9% |
| **Mean** | — | **43.0 ± 9.1%** | **0.301 ± 0.145** | **0.247 ± 0.088** | **4.1 ± 4.0%** | **24.9 ± 3.7%** |

---

## 7. Analysis

### 7.1 Worse sensing improved the success rate

83.3% → 96.7%. With perfect information the policy can hover and refine
before descending. With noisy information that cuts out near the pad, waiting
does not help — more time means more noise and more chance of losing the
marker — so committing early genuinely maximises reward. **The reward pays for
landing; it does not sufficiently punish landing badly.**

### 7.2 The abstract sensing model was optimistic by 51 points

96.7% → 45.7%. The abstract model assumed a ~30% blind fraction from a
vertical-only FOV threshold. Measured blind fraction with a real camera was
~88%, because with a moving platform the drone drifts up to 0.7 m laterally
while descending, and the marker leaves the frame *sideways* long before
height matters. The model captured only one of two dropout mechanisms.
Abstract sensor models should be validated against a rendered pipeline before
conclusions are drawn from them.

### 7.3 Success rate is stable across seeds; safety is not

Camera condition, before the descent cap: success 45/46/46% (±0.5%);
vz-violations 0%/65.2%/65.2% (±30.7%). Same code, same reward, same budget —
only the seed differs. **Seed variance is concentrated entirely in the
dimension nobody measures.**

### 7.4 A hand-tuned fix for one failure mode redistributes the risk

Capping descent speed is a real, reproducible improvement on the axis it
targets: vz-violations fell from 43.5 ± 30.7% to 4.1 ± 4.0%, and — as
important as the mean — **the seed variance on that metric collapsed** from
±30.7 to ±4.0. All three capped seeds now agree the policy is safe on vertical
speed; before the cap, seed alone decided between "perfectly safe" and
"unsafe two-thirds of the time."

But two costs appeared:

**Success-rate variance increased.** 45.7 ± 0.5% → 43.0 ± 9.1%. Touchdown
counts (94, 89, 99) show the drone still reaches the ground reliably; the
scatter is in whether it lands *on* the pad. Seed 2 traded accuracy for
caution more heavily than seeds 1 and 3.

**Lateral violations were only partially addressed.** 40.7% → 24.9%. The cap
constrains vertical speed specifically; lateral speed is affected only as a
side effect of the slower, more deliberate descent it produces.

**This is the general problem with hand-tuned reward shaping**, demonstrated
directly: fixing one failure mode by adding a term requires rebalancing
against every other term, by trial and error, and the previous nine reward
iterations in this project show how unreliable that process is. A fixed
penalty coefficient cannot express "stay under 0.5 m/s" as a hard requirement
— it can only be tuned until violations happen to be rare on the seeds you
tested, which is precisely what produced the high variance above. This
motivates §9.

### 7.5 Memory addresses the specific failure it was introduced for

The LSTM (on abstract sensing) cut lateral violations 34.9% → 24.1%, a 31%
relative reduction, at a cost of 18 points of success. `ep_len_mean` was 86.5
against the MLP's ~49 — the recurrent policy spends longer positioning before
committing.

### 7.6 Recurrent policies are markedly harder to train

Three of four LSTM runs failed or collapsed. Only explicit KL constraint and a
reduced learning rate produced a stable run — consistent with the known
instability of recurrent architectures in RL, the reason GTrXL exists for
transformers.

### 7.7 Training budget must be ruled out before blaming architecture

The MLP went 42% → 85% (privileged state) purely from tripling its training
budget, no code change. Stopping at 500k would have wrongly suggested an MLP
cannot handle randomised motion.

### 7.8 Failures cluster at high platform speed

Failed episodes averaged 0.73–0.90 m/s peak platform speed against a
distribution mean near 0.5, consistently across conditions and seeds.
Failures are a capability limit, not noise.

---

## 8. Naming correction

Early planning notes referred to a GRU. The implementation is an **LSTM** —
`sb3-contrib`'s `RecurrentPPO` provides `MlpLstmPolicy` only, with no GRU
option.

---

## 9. Next steps

1. **PPO-Lagrangian.** The direct conclusion of §7.4. Instead of a fixed
   descent-speed cap and penalty coefficient chosen by hand, state the
   requirement as a constraint — maximise success subject to touchdown
   velocity ≤ 0.5 m/s — with a Lagrange multiplier that tunes itself. The
   descent-cap result is the baseline this must beat: it should achieve
   comparable or better vz-violation rates *without* the success-rate
   variance the hand-tuned cap introduced, and *without* requiring the cap
   value (0.35) and penalty weight (4.0) to be chosen by trial and error.

2. **Reward ablation.** Vary the touchdown-bonus and descent-cap parameters
   across several settings and confirm finding 7.1 (worse sensing → higher
   success) persists. Needed before the claim can be called general rather
   than an artefact of one reward design.

3. **Classical baseline.** A properly tuned cascaded PI controller.
   Goldschmid & Ahmad's PI baseline scored 100% in every simulation scenario,
   so this is not a straw man.

4. **Two more LSTM seeds**, for error bars matching the other conditions.

5. **ArduPilot SITL validation** (semester 2 preparation). Zero-shot transfer
   from PyBullet to Gazebo running production firmware. Both installed and
   verified.

---

## 10. Reproducibility

| | Laptop | Lab desktop |
|---|---|---|
| OS | Ubuntu 24.04 | Ubuntu 22.04 |
| Python | 3.12 | 3.12 (deadsnakes PPA) |
| GPU | RTX 5050, 8 GB (sm_120) | NVIDIA T400, 4 GB (sm_75) |
| PyTorch | 2.13.0+cu130 | 2.13.0+cu130 |
| OpenCV | opencv-contrib-python | same |

Evaluation is deterministic: re-running seeds reproduces success rates
exactly. Every evaluation appends a row to `results_log.csv`.

Training is CPU-bound. Throughput: MLP state-based ~6700 steps/sec; MLP with
camera ~197; LSTM state-based ~180.

Code and full history: `github.com/yashwanth73337/drone-landing-rl` (private).
