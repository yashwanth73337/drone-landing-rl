# Project Notes — Vision-Based Drone Landing

**Plain-language reference for the whole project.**
As of 23 Sep 2026.

Items marked **[verify]** are inferred from behaviour or from other files
rather than read directly out of the source. Check them before quoting them in
a thesis.

---

## 1. What the project is trying to do

Land a quadrotor on a moving platform, using only what a camera on the drone
can see — no GPS, no motion capture, no cheating with simulator ground truth.

The long-term goal is a real drone doing this. Everything so far is in
simulation, built so that the simulation results transfer honestly.

### The research claim being built

> **Binary success rate is a misleading metric for this task.**

Two policies can have the same success rate and fail in completely different
ways — one aborts safely, one flies into the ground. Success rate cannot see
the difference. The project builds evidence for this stage by stage; §9 lists
the instances so far.

---

## 2. The one rule everything follows

**Exactly one thing changes per stage, and each stage builds on a frozen,
verified previous stage.**

This sounds bureaucratic. It has paid for itself repeatedly: every serious bug
in §10 was found only because one variable had moved and the effect could be
attributed to it.

Consequences enforced throughout:

- No result is compared against a baseline measured on a different task.
- Every evaluation states the task it was measured on.
- Verdicts are reported per bin, never pooled.
- Deviations from the reproduced papers are deliberate and written down.
- Negative and ambiguous results are reported as they are.

Every report has a **"what was established"** and a **"what was NOT
established"** section, in pairs.

---

## 3. The two strands

Early on the work split into two parallel lines. They are not sequential
stages; they are two different starting points.

### Strand A — exploratory PPO *(superseded, kept for reference)*

The first attempts. PPO on a landing environment, before the methodology
tightened up.

**Why it matters:** it produced the finding that shaped everything afterwards.
A policy hit 100% success on a sinusoidally moving platform — and it was
**memorising the trajectory phase**, not tracking the platform. A frequency
sweep exposed it: the policy failed at ω=0.2, succeeded at 0.5, failed at 0.8,
succeeded again at 1.8. A policy that genuinely tracked would degrade smoothly
with frequency. Non-monotonic success is the signature of memorisation.

That is where "100% success can mean nothing" entered the project.

### Strand B — faithful Lander.AI rebuild *(validated, frozen)*

A complete rebuild of Peter et al., *"Lander.AI"* (ICUAS 2024), done properly:

- **TD3** (off-policy actor-critic)
- The paper's **15-dimensional privileged observation** (their Eq. 3)
- A **PID-mediated action**: the policy outputs a small position offset, and a
  PID controller flies the drone to it
- A **12-level curriculum** that widens the spawn area first, then speeds up
  the platform

**Validated results, three controlled seeds:**

| Condition | Success | Touchdown precision |
|---|---|---|
| Stationary platform | **100.0 ± 0.0 %** | 3.39 ± 0.78 cm |
| Moving platform, 0.25 m/s | **98.7 ± 1.9 %** | 4.82 ± 0.76 cm |

Six checkpoints are frozen, tracked in git, and have not been touched since.
**These are the reference point for everything.**

Strand B is "privileged" — the policy is handed the exact relative position
and velocity of the platform by the simulator. That is not realistic. Closing
that gap is what the V-series is for.

---

## 4. The V-series — replacing ground truth with vision

Each V-stage takes the previous frozen thing and changes one part.

```
LanderAIAviary        Strand B, privileged, frozen
      |
VisionLanderAviary    V1 — camera attached, observes, never feeds the policy
      |
ArucoLanderAviary     V2/V3a — camera FEEDS the policy (ArUco + dead reckoning)
      |
LSTMLanderAviary      V3b — dead reckoning replaced by a learned LSTM
      |
ShinLanderAviary      V4 — estimator trained jointly with the policy (PPO)
```

### V1 — attach the camera and measure what it can see *(no training)*

Put a downward-facing camera on the drone and an ArUco marker on the pad, and
measure how much of a real landing the camera can actually see. **The policy,
physics and reward are untouched** — the camera observes and nothing else.

**Findings:**

- Physics proved **bit-identical across 9,000+ steps** with and without the
  camera. So the camera provably costs nothing.
- The marker has **no collision shape**, so it cannot contaminate contact
  detection.
- The camera sees the marker for roughly **half** of a landing trajectory. The
  blind stretches cluster in the final approach, exactly when you need it.

**Blind by height vs blind by time:** the closed-form geometry (centred approach)
says the marker is lost for 25% of the descent *height*; real trained-policy
flights are blind **48.5 ± 1.9%** of the *time*, because the drone spends
disproportionate time low, near the pad. The dominant cause is `too_close`
(marker overflows the frame, 55% avg), not lateral drift. (An earlier version of
these notes said a model "overstated performance by 51 points"; that figure is
not in the V1 report and was wrong.) Lesson: render the camera, don't model it.

### V2 — measure what vision costs *(no training)*

Take Strand B's **frozen** checkpoints and make them fly on an ArUco-derived
estimate instead of ground truth. Only the observation source changes —
policy, physics, reward, contact logic and episode seeds all identical.

| Dropout handling | Success |
|---|---|
| Privileged ground truth (baseline) | 98.0 ± 1.6 % |
| Constant-velocity prediction | **53.3 ± 5.7 %** |
| Zero-order hold | ~12 % |

**Vision costs about 45 percentage points.** Dominant failure: descending onto
a *stale* platform position — the drone commits to a landing based on where
the pad was several seconds ago.

Two ways of handling blindness:

- **zero-order hold** — freeze the last estimate. The platform keeps moving,
  the estimate doesn't, error grows linearly.
- **constant-velocity prediction** — keep extrapolating at the last observed
  velocity. Better, but it is *not* a Kalman filter: no covariance, no
  measurement update, just dead reckoning.

That 45-point gap mixes two causes — the estimator being bad, and the policy
seeing a distribution it never trained on. V3a separates them.

### V3a — retrain on the vision observation

Train TD3 from scratch on exactly the ArUco + constant-velocity observation it
will be tested on. No warm start.

**It did not answer its original question.** The vision-trained policy never
reached the task the baselines were measured on. It got to curriculum level
4/12 in 600k steps, level 9/12 at 1.8M, then **regressed** — 40.2% down to
21.2% — with the decline driven entirely by rising timeouts. It converged on
hovering.

What it produced instead was more useful: a clean characterisation of
**hovering as a rational response to an input the policy cannot verify**. The
15-D observation has no validity bit. The actor cannot tell a fresh detection
from a 181-step-old extrapolation. If you cannot tell, not committing is
correct.

### V3b — replace dead reckoning with a learned estimator

Swap the hand-coded constant-velocity predictor for an **LSTM trained offline
by supervised regression, then frozen**. The LSTM lives in the observation
pipeline, exactly where the predictor sat. TD3 is untouched.

Two data collection rounds were needed:

- **Round 1** passed offline validation decisively but failed a closed-loop
  cross-check — it lost in four of eight motion bins. The pooled verdict said
  "better" because one cell was dominated by near-stationary samples where the
  LSTM trivially wins.
- **Round 2** deliberately targeted the hard regime: episodes kept only if
  they contained a blind stretch ≥60 steps during which the platform was
  actually moving (mean |Δv| ≥ 0.12 m/s). 152 of 270 episodes kept.

**Results at matched task** (radius ≤0.23 m, **stationary**, 100 episodes,
seed base 9000):

| | V3a | V3b seed 1 | V3b seed 2 |
|---|---|---|---|
| success | 71.0% | 85.0% | 67.0% |
| failures | 29 | 15 | 33 |
| — timeout | 23 | 15 | 33 |
| — **descend-onto-stale-position** | **6** | **0** | **0** |
| position est. error ¹ | 0.01104 m | 0.0125 m | 0.0110 m |
| velocity est. error ¹ | 0.18371 m/s | 0.218 m/s | 0.188 m/s |

*Which file "V3b seed 1" is:* `best_success_model.zip` at **560k** steps in
`results_lstm_v3b/run_20260914_195837`. Re-evaluated 23 Sep and reproduced to
every digit. It is **not** `best_model.zip` (430k) — see below.

¹ *What the error rows measure (corrected 23 Sep).* They come from
`evaluate_policy_full`, which averages `info['est_err_pos_norm']` /
`info['est_err_vel_norm']` **only on frames where the marker was detected**.
That key is the parent class's **constant-velocity** estimate —
`LSTMLanderAviary._computeInfo` relabels it `cv_err_pos_norm`. So for V3b
these rows are **not the LSTM's error**, and for every run they contain **no
blind frames at all**. They are essentially detection accuracy. **[verify the
key's definition in `ArucoLanderAviary._computeInfo`.]** The same applies to
`pos_err_m` / `vel_err_ms` in every `perception_evaluations.csv`. For the
LSTM's own error, use `diag_v3b_moving.py` (blind-age table).

**The headline is not the success rate.** The two V3b seeds differ by 18
points, which is larger than either seed's gap to V3a — so success rate cannot
resolve the comparison at two seeds. But both seeds produce **zero**
stale-descent failures against V3a's six, and they agree exactly despite that
18-point disagreement. Probability of zero in 48 failures if the rate were
unchanged: about 1.5 × 10⁻⁵. **This holds on a stationary pad only** — on a
moving pad V3b seed 1 crashes 5/100 at 0.05 m/s and 27/100 at 0.10 m/s (next
section).

**The mechanism** is the tail of the error distribution, not its centre. At
blind ages of 16–60 steps the constant-velocity predictor exceeds one metre of
position error on 5.6% of frames, worst case 2.76 m. The LSTM, on the same
frames, never exceeds one metre and peaks at 0.27 m. Against a 0.1 m success
radius, a metre-scale error during descent is a commanded descent into empty
space. *(Scope: those frames came from the offline/passive comparison. Under
V3b's own closed-loop flight the LSTM does exceed 1 m beyond 120 steps blind,
at every speed tested — next section.)*

### V3b on a moving pad, and its continuation *(22–23 Sep)*

**Checkpoint identity — three different policies in one run directory**
(`run_20260914_195837`):

| File | Timestep | Written by | Note |
|---|---|---|---|
| `best_model.zip` | 430k | SB3 `EvalCallback` (mean reward, 5 episodes) | trained at level ≤ 4; 65% on its level-4 promotion eval |
| `best_success_model.zip` | 560k | curriculum callback (level-dominant) | **the reported V3b seed 1**; 85% on its level-5 promotion eval |
| `final_model.zip` | 600k | end of run | 25% on its last level-5 promotion eval |

The 22 Sep "zero-shot moving" numbers (64% at 0.05 m/s, 36% at 0.10 m/s) were
measured on **`best_model.zip` (430k)** and do **not** describe the reported
policy. Level-5 promotion evals over 550–600k read 50, 15, 85, 50, 60, 80, 80,
65, 65, 60, 25% — the policy swings by more than sampling noise between evals
5k steps apart, so any single checkpoint is partly arbitrary.

**Zero-shot on a moving pad, 560k policy** (radius 0.11–0.23 m, 100 episodes,
seed 9000). Two actor inputs, same policy, same seeds — only
`actor_obs_source` differs (`diag_v3b_moving.py`):

| speed | LSTM estimate: success / timeout / below_platform | true state: success / timeout / below_platform |
|---|---|---|
| 0.00 m/s | 85 / 15 / **0** | 89 / 11 / **0** |
| 0.05 m/s | 73 / 22 / **5** | 76 / 24 / **0** |
| 0.10 m/s | 44 / 29 / **27** | 40 / 55 / **5** |

- **The policy limits success.** Handing it the true state adds at most 4
  points at any speed; with perfect information it still falls 89 → 76 → 40%.
- **The estimator causes the crashes.** All 5 at 0.05 m/s and 22 of 27 at
  0.10 m/s disappear with the true state. Over the last 30 steps before a
  `below_platform`, the LSTM placed the pad 0.11–0.14 m away while it was
  really 0.34–0.48 m away.
- **Blindness is partly a policy effect.** With the true state the drone is
  *more* blind at 0.10 m/s (69.6% vs 65.4%): it falls behind the pad and
  loses the marker. Its timeouts end ~0.30 m from the pad.
- *Caveat:* the true state is an input this policy never trained on. Read that
  column as an upper bound on what better estimation could buy.

**LSTM error by blind age, closed loop** (actor on LSTM, 3-D position error):

| blind age | mean at 0.00 / 0.05 / 0.10 m/s | p95 | frames |
|---|---|---|---|
| visible | 0.023 / 0.023 / 0.023 m | 0.05 m | ~4,850 each |
| 1–15 steps | 0.022 / 0.023 / 0.027 m | 0.04–0.06 m | ~1,640 each |
| 16–60 | 0.074 / 0.072 / 0.091 m | 0.19 / 0.18 / 0.23 m | 3066 / 3174 / 3765 |
| 61–120 | 0.227 / 0.189 / 0.226 m | 0.33 / 0.37 / 0.62 m | 1187 / 1784 / 3186 |
| >120 | 0.241 / 0.206 / 0.280 m | 0.47 / 0.77 / 1.12 m | 2089 / 3473 / 4837 |

- **Mean error per blind age is roughly speed-invariant** — so "the LSTM does
  not degrade on a moving pad" is right in that narrow sense.
- **But it was never small.** After 2 s blind it is already 0.2+ m on a
  *stationary* pad. The stationary policy converts that into timeouts; on a
  moving pad the drone spends 2.4× as many frames more than 2 s blind
  (3,276 → 8,023), the tail widens (p95 0.47 → 1.12 m), and the policy
  commits on it — crashes.
- **LSTM still beats constant velocity on every blind bin** (>120 steps at
  0.10 m/s: mean 0.28 vs 0.79 m, max 1.76 vs 11.26 m). CV is better on visible
  frames (0.012 vs 0.023 m).

**Continuation into the curriculum** (`continue_v3b.py`,
`results_lstm_v3b_cont/cont_20260923_011516`): resumed the 560k checkpoint
with its paired replay buffer, estimator frozen, start level 6 (radius 0.26 m,
stationary), 600k more steps → 1.16M total, 12.06 h. Report as **"560k fresh +
600k continued"**; RNG state is not restored.

| steps | mean success (20-ep evals) | best eval | timeout / eval | below_platform / eval | near-miss / eval |
|---|---|---|---|---|---|
| 560–760k | ~44% | 75% | ~9 | ~2 | ~10 |
| 760–960k | 10–24% | 40–55% | 13–17 | 1–3 | 15–18 |
| 960k–1.16M | 3–15% | 20–60% | 17–19 | ~0 | ~2 |

- **Negative result.** Never promoted past level 6 (best of 120 evals: 75%,
  threshold 90%), so it never reached the moving levels.
- **Collapsed into hovering** from ~960k. Perception checks flip between two
  modes: 34–68% blind (landing) and **< 1% blind** (hovering high, marker
  always in view). The last 200k steps are mostly hovering.
- **`best_L06.zip` on the original task** (r 0.11–0.23 m, stationary, 100 ep,
  seed 9000): **76%**, 24 timeouts, 0 crashes, 56.2% blind — against 85% for
  the starting checkpoint. No improvement.
- *Hypothesis, untested:* at radius 0.26 m blind stretches get much longer
  (longest 150–250 steps vs 78 at level 5); early crashes (−10) teach the
  critic that descending under uncertainty is costly; timeouts then take
  over. The V3a pattern again.
- **The collapsed policy hovers even with the true state.** Observation swap
  on `final_model.zip` (1.16M), r 0.11–0.23 m, stationary, 100 ep, seed 9000:
  **0% success, 100 timeouts on LSTM input and 100 timeouts on true-state
  input**; blind 1.0% / 0.3%; near-miss 1 / 0. Over its last second it sits
  only 0.05–0.08 m laterally from the pad centre, marker in view, and does
  not descend. So the hovering lives in the learned policy/values, not in the
  estimate — the V4 result (§9 #5) reproduced under TD3.
  *What this does NOT show:* whether a validity signal present *during
  training* would have prevented the collapse from forming. It shows only
  that, once formed, the collapse does not depend on the input.

### V4 — the full Shin et al. method *(built, does not land)*

The primary reference paper: Shin et al., *"Vision-Based Autonomous Drone
Landing on Moving Platforms With Uncertain Motion via DRL"* (IEEE RA-L, May
2026).

V4 moves **four variables at once**, deliberately, because they are
inseparable:

1. The estimator is trained **jointly** with the policy, not offline and frozen
2. The algorithm becomes **recurrent PPO**, not TD3
3. The critic becomes **asymmetric and privileged** — it sees ground truth, the
   actor does not, and it is discarded at deployment
4. An **active-perception reward** penalises actions that make the next step's
   estimate worse

Why inseparable: a recurrent estimator inside the policy cannot be trained by
TD3 without sequence-based replay, which is what forces PPO. An auxiliary loss
has nowhere to attach unless the estimator is inside the policy. The
active-perception reward is *defined* in terms of that auxiliary loss.

**Where V4 differs from the paper** (documented deviations):

| | Shin et al. | V4 |
|---|---|---|
| perception front-end | learned keypoint encoder, hexagonal pad | ArUco marker |
| camera | pitched 60° forward | nadir (straight down) |
| action | [vx, vy, vz, yaw-rate] | 3-D position offset, no yaw |
| output frame | body | world **[verify: `--frame body` exists but untested]** |
| platform motion | random accelerations | one constant velocity per episode |

**Result of run 1** (600k steps, 8.2 h): the **estimator worked** — 0.0126 m
position, 0.00464 m/s velocity. The **policy never landed**. Zero descent
attempts in 600k steps; it hovered at 0.43 m.

*(Caveat: that 0.0126 m was measured under a hovering policy at 0.1% blind
fraction — a trivially easy state distribution. It is NOT comparable to V3b's
0.0125 m at 53% blind — which, see ¹ above, is itself a visible-frame CV
figure, not the LSTM's.)*

**Result of run 2** (gSDE exploration, 150k steps): at 5,000 steps the drone
descended in **20 of 20 episodes**. It missed the pad every time, collected the
crash penalty, and had **unlearned descending by 35k steps**.

It missed because it was blind: the position estimate was 0.42–1.67 m while
the lateral offset it needed to correct was only 0.11–0.14 m.

**The diagnosis — a chicken-and-egg failure:**

> The policy needs a good estimate to land. The estimator only ever sees the
> states the policy visits. An early, bad policy generates states where
> estimation is hopeless. Neither can bootstrap the other.

This is the mirror image of an effect V3b already documented — that a policy
flying on *good* estimates visits states where estimation is *easier*. And it
is an argument **for** V3b's offline-then-freeze approach over the paper's
joint training, in this setting: V3b's estimator was trained on a curated
dataset that deliberately covered the hard regime, where V4's only sees its own
policy's garbage.

---

## 5. The environment, in detail

### Simulator and drone

- **PyBullet** physics, via **gym-pybullet-drones**
- Drone: **Crazyflie 2.X (CF2X)** — a small quadrotor, mass 0.027 kg, arm
  length 0.0397 m
- Control loop: **300 steps per episode over 10 simulated seconds**, so 30 Hz
  **[verify: `episode_len_sec = 10.0` is explicit; 30 Hz is inferred from
  300 steps]**

### How the drone is actually flown

The policy does **not** output motor thrusts. It outputs a 3-D command `c_t`,
which becomes a small position offset:

```
Δp = 0.1 · c_t
```

That offset is handed to `DSLPIDControl`, a built-in PID controller, which
works out the motor commands to get there.

**This matters more than it looks.** The offset is taken from the drone's
*current* position every step — it does not accumulate. So the action is
effectively a **velocity command**. Random uncorrelated actions therefore
produce almost no net movement: measured, a random policy descended 0.1 m in
300 steps where a true random walk would predict ~0.9 m. This is why PPO's
white-noise exploration failed in V4 and why TD3's noise-around-a-mean did not.

There is **no yaw control**, which is why the camera points straight down
rather than forward — a forward-pitched camera is useless if you cannot turn.

### How an episode starts

1. The platform is created and given **one constant velocity for the whole
   episode**, sampled by `_samplePlatformMotion()`. It does not accelerate.
   *(Important consequence: the constant-velocity predictor has the **correct
   model class** for this environment. Any claim that the LSTM handles
   "non-constant dynamics" cannot be made from this work.)*
2. Vertical platform drift is **switched off** (`vertical_speed_factor = 0.0`)
   — see the bug in §10.
3. The drone spawns at **0.5 m altitude**, at a random horizontal offset inside
   an annulus whose outer radius is set by the current curriculum level. The
   inner radius is always 0.11 m.
4. The estimator state is cleared completely. **No ground truth is used to
   initialise it** — not even once, not as a seed. Before the first detection,
   the estimate is exactly zero, which is *obviously* wrong rather than
   plausibly wrong, so a policy relying on it fails visibly.

### How an episode ends

| Ending | Meaning |
|---|---|
| **success** | contact with the pad's upper surface, within tolerance |
| **below_platform** | the drone went below the pad without landing on it — descended onto a stale or wrong position |
| **timeout** | 300 steps elapsed. In practice: hovering |
| **tilt_bound** | tipped past the attitude limit |
| **position_bound** | drifted out of the workspace |

Two more terms used in reports:

- **touchdown height = 0.0125 m** — derived from the Crazyflie's URDF, not
  guessed. The collision geometry is a cylinder 0.025 m long centred on the
  tracked origin, so contact happens at half that.
- **near-miss** — an episode that spent ≥10 steps below 0.03 m without
  succeeding. It got close and didn't finish.

### The camera and marker

- Grayscale, **128 × 128**, **90° field of view**, pointing straight down
- **ArUco marker, 0.25 m**, on the pad, with **no collision shape**
- Exactly **one render and one detection per control step**, cached and shared
  between the observation and the diagnostics. Rendering twice would halve
  throughput and, worse, could give the observation and the log different
  detections.

The marker's position is recovered with `cv2.solvePnP`. The maths is arranged
so the drone's own world position **cancels out completely** — the estimate
depends only on the drone's attitude (which a real IMU measures), two fixed
calibration constants, and the solver output. So the estimate is provably
non-privileged, rather than just asserted to be.

---

## 6. The observation, stage by stage

Strand B's observation is 15 numbers (the paper's Eq. 3):

```
[ θ (3)      attitude          PROPRIOCEPTIVE — the drone's own body
  v (3)      velocity          PROPRIOCEPTIVE
  ω (3)      angular velocity  PROPRIOCEPTIVE
  d (3)      relative position TARGET — this is what vision must supply
  Δv (3)     relative velocity TARGET — this too ]
```

The first nine describe the drone itself and a real airframe measures them
onboard, so keeping them truthful is not cheating. The last six describe the
*platform*, and those are what a camera has to provide.

| Stage | Where `d` and `Δv` come from |
|---|---|
| Strand B | simulator ground truth |
| V1 | ground truth (camera observes but doesn't feed) |
| V2 / V3a | ArUco detection + constant-velocity dead reckoning |
| V3b | frozen LSTM estimator |
| V4 | **the actor no longer receives an estimate at all** |

**V4 is structurally different.** The LSTM moved out of the environment and
into the policy, so the actor gets the *raw measurement* and does its own
temporal integration:

```
actor  (13-D)  d_meas (3), valid (1), velocity (3), rpy (3), angular vel (3)
critic (15-D)  proprioception (9) + TRUE relative state (6)   [PRIVILEGED]
target ( 6-D)  TRUE relative state — a LABEL for the auxiliary loss only
```

During blindness `d_meas` is zeros and `valid` is 0. The previous measurement
is *not* held at the input — carrying information over time is the recurrent
state's job, and that is exactly the capability being tested.

**No stage up to and including V3b gives the actor a validity bit.** It cannot
tell a fresh detection from a stale extrapolation. That is deliberate — it
isolates estimate *quality* from uncertainty signalling — and testing a
validity flag is a planned separate experiment.

`LSTMLanderAviary` can also feed the actor the true state
(`actor_obs_source='privileged'`) or the constant-velocity estimate
(`'predict'`), with both estimators still running passively. That is what the
observation-swap diagnostic uses.

---

## 7. The curriculum

Twelve levels. Spawn radius widens first, then the platform starts moving.

| Level | Spawn radius (m) | Platform speed (m/s) |
|---|---|---|
| 1 | 0.11 – 0.14 | 0.00 |
| 2 | 0.11 – 0.16 | 0.00 |
| 3 | 0.11 – 0.18 | 0.00 |
| 4 | 0.11 – 0.20 | 0.00 |
| 5 | 0.11 – 0.23 | 0.00 |
| 6 | 0.11 – 0.26 | 0.00 |
| 7 | 0.11 – 0.30 | 0.00 |
| 8 | 0.11 – 0.30 | 0.05 |
| 9 | 0.11 – 0.30 | 0.10 |
| 10 | 0.11 – 0.30 | 0.15 |
| 11 | 0.11 – 0.30 | 0.20 |
| 12 | 0.11 – 0.30 | 0.25 |

**Levels 1–7 are all stationary.** V3a (1.8M steps, one seed) entered the moving
phase — levels 8 and 9 (0.05 and 0.10 m/s) — then regressed at level 9 to ~21%
(best checkpoint 25% on its own level-9 task). V3b and V4 never left the
stationary levels: V3b seed 1's continuation (600k further steps from level 6)
never promoted past level 6. No vision stage has been validated on the
0.25 m/s task.

**Promotion rule:** ≥90% success on one 20-episode evaluation, run every 5,000
steps.

**That rule is noisy and it matters.** One 20-episode sample at near-threshold
capability clears 90% about one time in five if the true rate is 80%, and about
one in eleven at 75%. V3b seed 2 hit 80% four separate times at level 1 without
promoting, with a 5% evaluation interleaved. So "level reached" is substantially
a matter of luck, which is why cross-run comparisons are always done at a
**pinned, matched task** instead.

---

## 8. The reward

Roughly:

| Component | Value |
|---|---|
| successful landing | **+20** |
| crash / excessive drift | **−10** |
| every step | **−0.02** (time cost) |
| shaping | rewards progress toward the pad, penalises fast descent |

**[verify: the exact shaping weights live in `LanderAIAviary.py`.]**

**Four deviations from the published Lander.AI reward are documented:**

1. The mid-range term is self-contradictory as literally written — `R` is
   defined identically to `d_target`, which makes the term always exactly zero.
2. The "otherwise" branch is unreachable.
3. The far-field branch carries no gradient.
4. The terminal rescaling (to +20) and the per-step time cost were added by
   this project to fix the stall optimum in §10.

### What the reward landscape actually looks like

Measured by flying a perfect controller down and recording reward per step:

| Height band | reward/step, descending | reward/step, hovering |
|---|---|---|
| 0.45–0.60 m | −0.059 | −0.023 |
| 0.40–0.45 m | −0.065 | — |
| 0.30–0.40 m | −0.083 | — |
| 0.20–0.30 m | **−0.129** | — |
| 0.10–0.20 m | −0.118 | — |
| 0.05–0.10 m | −0.075 | — |
| 0.00–0.05 m | **+1.65** (the +20 lands here) | — |

**Descending is a valley.** A fast descent accumulates about −5.2 of penalty
before collecting +19.98 at touchdown. Total is still strongly positive
(+14.77 against hovering's −7.00) — but every individual step of the way there
is worse than doing nothing.

**A slow descent escapes the valley.** At a gentle rate the far-field reward is
*better* than hovering (+0.0102/step) and it still lands within the budget. So
the reward is navigable and is not itself the blocker — it is doing exactly
what "penalise fast descent" is supposed to do.

An off-policy learner (TD3) gets through the valley because one lucky landing
enters the replay buffer and is replayed thousands of times. An on-policy
learner (PPO) sees a rare success once and throws it away. **That difference is
the clearest single explanation for why V3a/V3b learn and V4 does not.**

---

## 9. The metric findings — the thesis argument

Instances of a summary statistic hiding or inverting the effect that actually
determines outcomes.

From V3a/V3b (stationary):

1. **Success rate.** At matched task, V3b's two seeds score 85.0% and 67.0%
   against V3a's 71.0%. The between-seed spread (18 points) is larger than
   either seed's gap to V3a, so success rate points in opposite directions
   depending on which seed you drew. Stale-descent count resolves it
   immediately and both seeds agree exactly: 6 for V3a, 0 for both.

2. **Mean position error.** V3a has the *best* reported mean error of the
   three runs (0.01104 m) and produces all six crashes. *Corrected 23 Sep:*
   the reported figure is the constant-velocity estimate on **visible frames
   only**, in every run (§4, footnote ¹). The earlier explanation — a penalty
   on visible frames cancelling an advantage on blind ones — was wrong: the
   metric contains **no blind frames at all**, which is where every crash
   originates. The point stands in a stronger form: the metric reported as
   "estimation error" cannot see the regime that decides safety. (The worst
   excursions, 11.24 m vs 2.19 m, come from the separate blind-age comparison
   and are unaffected.)

3. **Mean velocity error.** Same scope problem.

4. **Touchdown precision.** A discarded checkpoint had *better* precision
   (4.94 cm) than the one four times more likely to land at all (7.15 cm) —
   because it only succeeded on the easiest episodes.

From V4:

5. **A near-perfect estimate does not fix hovering.** V3b could only
   hypothesise that hovering was a rational response to an unverifiable
   estimate. V4 hands the policy a 0.0126 m estimate and it hovers completely.
   Estimate quality is definitively not the cause. *Reproduced 23 Sep under
   TD3:* the collapsed V3b continuation policy, handed the exact true state
   while centred 5–8 cm from the pad with the marker in view, times out in
   100 of 100 episodes.

6. **The paper's active-perception reward deactivates itself.** Its threshold
   τ=0.01 is tuned to the paper's error scale. V4's estimator drops below it by
   50k steps, after which the reward term is exactly zero — inert for 92% of
   training.

From V3b on a moving pad (22–23 Sep):

7. **Same success, different failures.** At 0.10 m/s the same policy scores
   44% on the LSTM estimate and 40% on the true state — success rate says the
   estimator does not matter. Crashes are **27 vs 5**.

8. **Zero crashes can mean zero attempts.** In the V3b continuation,
   `below_platform` fell to ~0 over the last 200k steps while success fell to
   3–15%. The policy had stopped descending.

---

## 10. Bugs found, and why they matter

Each of these was found only because one variable had moved. Each would have
silently corrupted results.

| Bug | What it would have done |
|---|---|
| `_computeReward()` runs before `_computeTerminated()` in `BaseAviary` | The touchdown bonus was never paid — across eight reward variants |
| Discount-evadable stall optimum: a terminal-only timeout penalty discounts to ~5% of face value | Hovering became the value-maximising policy |
| Checkpoint-selection ceiling bug | The same three runs read as 46 ± 38% or 98.7 ± 1.9% depending which file you loaded |
| Checkpoint-selection, mirror image (V3b seed 2) | A level-7 policy at 20% overwrote a level-6 policy at 90%. Same task, the discarded one scored 67% against the saved one's 17% |
| Unbounded vertical platform drift | The pad moved ±0.75 m vertically over an episode — roughly half of all episodes were physically unwinnable |
| `solvePnP` returning `ok=True` with NaN under near-collinear corners | NaN into the PID controller, SVD failure |
| Marker corner scale used the inner payload, not the outer black square | 0.238 m detection error, fixed to 0.020 m |
| Touchdown height assumed rather than derived | Corrected to 0.0125 m from the URDF collision cylinder |
| Phase memorisation mistaken for tracking | 100% success that was not a real capability |
| Pooled offline validation verdict | The estimator passed a pooled check while losing in four of eight motion bins |
| Verdict rule that could not express the effect it tested | Printed "hypothesis NOT supported" while its own tables showed the opposite |
| Unclipped actions in the V4 evaluator | The policy trained under action clipping and was evaluated without it, at up to twice the commanded step size |
| Evaluation attributed to the wrong checkpoint (22 Sep) | Moving-pad numbers measured on `best_model.zip` (430k) were read as the reported 560k policy. 64% vs 73% at 0.05 m/s, same seeds |
| "Estimation error" metric scope | Reported as estimator accuracy for V3b; it is the constant-velocity estimate on visible frames only. The LSTM's error was never in it |
| SB3 stores the transition *after* the callback | A buffer saved from inside the callback holds `num_timesteps − n_envs`. A strict model/buffer pairing check would reject every `best_success` pair. Caught in code review before any run |

**The recurring lessons:** *curriculum level is a state variable, not a
performance measure*; *a summary statistic that cannot express an effect will
confidently report its absence*; and *a number without its file, timestep and
task attached cannot be trusted*.

---

## 11. File map

### Environments — `src/envs/`

| File | Stage | Purpose |
|---|---|---|
| `LanderAIAviary.py` | Strand B | Privileged 15-D observation, the reward, contact logic, platform motion. **FROZEN** |
| `VisionLanderAviary.py` | V1 | Adds the downward camera and the collision-free ArUco marker. Observation unchanged |
| `ArucoLanderAviary.py` | V2/V3a | Overrides `_computeObs()` only — replaces `d` and `Δv` with ArUco estimates. Holds the detection cache and the dead-reckoning estimator |
| `LSTMLanderAviary.py` | V3b | Adds the frozen LSTM. `estimator_mode` and `actor_obs_source` deliberately decoupled |
| `ShinLanderAviary.py` | V4 | Dict observation {actor, critic, target}. The estimator has **left** the environment |
| `DiagnosticArucoAviary.py` | debugging | Strict finite-value assertions at every boundary; dumps full context at the first NaN |

### Policies — `src/policies/`

| File | Purpose |
|---|---|
| `asymmetric_recurrent.py` | V4's network. Actor reads `obs['actor']`, critic reads `obs['critic']`, estimation head hangs off the actor's LSTM. The privilege boundary is one function, `_split_features` |

### Training and evaluation — `src/scripts/`

| File | Purpose |
|---|---|
| `train_lander.py` | Strand B: TD3 train / eval / GUI playback. Holds `CurriculumState` |
| `train_landing.py` | Strand A PPO. Superseded, kept for reference |
| `train_aruco_v3a.py` | V3a trainer **and the shared curriculum machinery** — `CURRICULUM_LEVELS`, `CurriculumAndCheckpointCallback`, `evaluate_policy_full`. V3b imports from here so nothing can drift |
| `train_lstm_v3b.py` | V3b trainer. `--estimator` required |
| `continue_v3b.py` | V3b continuation. Paired model + buffer only (`--resume-from final\|best_success`), `--start-level` required, per-level checkpoints, writes `continuation.json` |
| `diag_v3b_moving.py` | No training. Same policy on LSTM vs true-state input; belief vs truth before each ending; LSTM and CV error by blind age |
| `train_shin_v4.py` | V4 trainer. Auxiliary loss inside the PPO objective, active-perception reward injected before advantages. `--selftest` |
| `train_estimator.py` | Trains the offline LSTM by supervised regression |
| `collect_estimator_data.py` | Round-1 dataset |
| `collect_hard_regime.py` | Round-2 dataset — the targeted hard-regime collection |
| `compare_error_distributions.py` | Runs both estimators on identical frames, per blind-age bin |
| `crosscheck_estimator_2x2.py` | Passive policy × level cross-check |
| `check_estimator_closed_loop.py` | Closed-loop estimator check |
| `eval_aruco_v2.py` | V2 evaluation of frozen checkpoints |
| `verify_camera.py` | Standalone ArUco detection check |
| `fov_diagnostic_v1.py` | V1 field-of-view geometry |
| `test_env.py` | Environment sanity check |
| `bench_v3a.py` | Throughput benchmark; asserts one render per control step |
| `repro_v3a_nan.py` | Reproduces the solvePnP NaN |
| `check_behind_cause.py` | Failure-cause analysis |
| `continue_v3a.py` | V3a continuation runs |
| `watch_landing.py` | GUI playback |
| `diag_v4_behaviour.py` | V4 trained vs random, minimum height reached |
| `diag_v4_reachability.py` | Trained / white random / held random / ground-truth oracle |
| `diag_v4_reward_profile.py` | Maps reward against height. `--gain` sweeps descent rate |

### Models — `src/models/`

| File | Purpose |
|---|---|
| `temporal_estimator.py` | The LSTM class: LSTM(13 → 128, 1 layer) → MLP(128 → 128 → 128 → 6) |
| `temporal_estimator_v3b_r2.pt` | The frozen round-2 estimator. **Every V3b result depends on this file** |

### Results

All training scripts run from `src/scripts/` and write there; the repo-root
`results/` directory is empty.

| Path (under `src/scripts/`) | Contents |
|---|---|
| `results_lander/*_seed{1,2,3}_final.zip` | The six trusted Strand B checkpoints |
| `results_aruco_v3a/`, `results_aruco_v3a_cont/` | V3a runs and continuations |
| `results_lstm_v3b/run_*/` | V3b runs, one directory each |
| `results_lstm_v3b_cont/cont_*/` | V3b continuations (`continuation.json`, `best_LNN.zip`, `best_per_level.csv`) |
| `results_shin_v4/`, `results_shin_v4_sde/` | V4 runs |
| `eval_logs/` | Tee'd outputs of pinned evaluations and diagnostics (`diag_moving_*/` holds per-episode CSVs) |
| `.../success_evaluations.csv` | **Per-run curriculum history — the raw evidence for every reported number** |
| `.../perception_evaluations.csv` | Visible-frame CV error over training (see §4 ¹ — not the LSTM) |
| `.../best_per_level.csv` | Best checkpoint per curriculum level (added Sep 2026) |
| `results_lander_log.csv` | Every Strand B evaluation, appended, never overwritten |

### Reports

| File | Status |
|---|---|
| `V3b_REPORT.md` | In the repo; two seeds, stationary. Needs the §4 ¹ metric-scope correction and the moving-pad results |
| `V1/V2/V3a_REPORT` | **PDF only** — no markdown source in the repo |
| `PROJECT_REPORT.md`, `STATUS_REPORT` | Earlier overviews |

---

## 12. Conventions you must not break

**Seed bases** are fixed so numbers from different runs are comparable:

| Base | Used for |
|---|---|
| 9000 | Final evaluations — **the cross-run comparison seed** |
| 10000 | Curriculum promotion evaluations during training |
| 20000 | Perception diagnostics |

**Every evaluation must pin its task explicitly:**

```bash
--radius-min 0.11 --radius-max 0.23 --speed 0.0
```

`--speed` is **not optional**. The scripts default to 0.25 m/s. Omitting it
silently evaluates the policy on a task it has probably never seen — this
already happened once and produced an 11% figure that meant nothing.

Without `--radius-min/--radius-max`, each run is scored on whatever level it
happened to reach, which is not a comparison at all.

**Report every number with the task attached.** A success rate without its
radius and speed is meaningless in this project.

**Name the checkpoint file and its timestep with every evaluation.**
`best_model.zip` (SB3 EvalCallback, mean reward), `best_success_model.zip`
(curriculum callback, level-dominant) and `best_LNN.zip` (per level) are
different policies. Read a zip's timestep without loading it:

```bash
python -c "import zipfile,json,sys; print(json.loads(zipfile.ZipFile(sys.argv[1]).read('data'))['num_timesteps'])" path/to/model.zip
```

**"pos estimation error" is not the LSTM's error.** For V3b estimator
questions use `diag_v3b_moving.py`'s blind-age table.

---

## 13. Where things stand

| Item | State |
|---|---|
| Strand B baseline | Complete, validated, 3 seeds, frozen |
| V1 camera instrumentation | Complete |
| V2 vision-cost measurement | Complete |
| V3a constant-velocity training | Complete, reported honestly as a non-answer |
| V3b LSTM estimator | Complete, 2 seeds, stationary. Moving-pad zero-shot and observation-swap diagnostic done (§4) |
| V3b seed 1 continuation | Done, **negative**: stuck at level 6, collapsed to hovering; best checkpoint 76% vs 85% start; final policy hovers 100/100 even on true state |
| V4 full Shin method | Built; estimator works, policy does not land; cause understood |
| Moving-platform task | V3a reached level 9 (0.10 m/s) at ~25%; V3b never trained on it; nothing validated at 0.25 m/s |

**You do have a working vision-based lander, on a stationary pad.** V3b seed 1
lands 85% at level 5 (r ≤ 0.23 m). Zero-shot on a moving pad: 73% at 0.05 m/s,
44% at 0.10 m/s, with crashes appearing (5 and 27 per 100).

**The open gap:** no vision-trained policy has been trained through the moving
levels without regressing to hovering. V3a (constant velocity) and V3b's
continuation (LSTM) both collapsed into hovering; V4 never descended at all.

**Planned next experiments:**

- **Faithful Shin et al. rebuild (V5)** — to be scoped in a new chat, spec
  sheet first, every deviation decided before code.
- **Validity-flag ablation — now a weaker bet.** The collapsed V3b policy
  hovers even with perfect input, so once hovering forms it is in the learned
  values, not the input. A flag could still matter *during* training (before
  the collapse forms), but the evidence now points at least as much toward
  training-side causes: reward valley, exploration, buffer composition.
  Needs a 16-D observation and a fresh run — a supervisor decision.
- **V4 with the estimator frozen** — a one-variable test of joint vs frozen
  inside V4's own PPO setup. Informative whichever way it comes out
- **Per-level checkpointing** is in place and already paid off: `best_L06.zip`
  survived the continuation's collapse

**Semester 2:** sim-to-real onto the lab's ArduPilot quadrotors and hexarotors,
with real UWB ranging and ArUco markers under ROS.
