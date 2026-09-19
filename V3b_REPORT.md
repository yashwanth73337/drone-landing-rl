# Vision-Based Landing — Stage V3b Report

**As of:** Sep 19, 2026 · Semester 1, MTP
**Scope:** Replacing the hand-coded constant-velocity predictor with a frozen,
offline-trained LSTM temporal estimator. TD3 unchanged. Seeds 1 and 2, 600k
transitions each, matched to V3a block 1. No PPO, no asymmetric critic, no
active-perception reward, no validity flag, no reward changes, no warm start.
`LanderAIAviary.py`, `VisionLanderAviary.py`, `ArucoLanderAviary.py` and the six
trusted checkpoints unchanged.

---

## 1. The question

V3a established that a memoryless TD3 policy trained on ArUco + constant-velocity-
predicted state reaches curriculum level 4/12 in 600k transitions, level 9/12 at
1.8M, and then **regresses** at level 9 (40.2% → 21.2%) with the decline driven
entirely by rising timeouts. The diagnosis was that the actor cannot distinguish a
fresh estimate from a 181-step-old extrapolation, so it learns not to commit to
touchdown.

V3b tests one thing:

> Does replacing the hand-coded predictor with a learned temporal estimator let the
> same memoryless TD3 policy get further?

**The actor interface is deliberately unchanged** — still 15-D, still no validity
bit, no estimate age, no hidden state. This isolates estimate QUALITY rather than
smuggling in an uncertainty signal. Testing a validity flag remains a separate later
ablation.

---

## 2. This is an adaptation of Shin et al., not a reproduction

| | Shin et al. (RA-L 2026) | V3b |
|---|---|---|
| perception front-end | learned keypoint/visual embedding | ArUco metric measurement |
| temporal model | LSTM | LSTM |
| output frame | body frame | **world frame** (keeps the 15-D TD3 interface unchanged) |
| validity signal | implicit in the embedding | explicit bit, **inside the estimator only** |
| training | **jointly** with PPO via auxiliary loss | **offline supervised**, then frozen |
| critic | asymmetric privileged | unchanged symmetric TD3 |
| active perception | yes | no |

Only the recurrent-temporal-estimation idea is shared. The two should not be
described as the same system.

Freezing the estimator is what lets TD3 stay untouched: the estimator becomes part
of the observation pipeline, exactly where the constant-velocity predictor sat. Joint
recurrent training would require sequence-based replay, which TD3 fights, and that is
what eventually forces PPO — deferred on purpose.

---

## 3. The estimator

**Input, 13-D per control step:** `d_meas` (3), `valid` (1), drone velocity (3),
attitude rpy (3), angular velocity (3).

`previous_action` was deliberately **excluded**. It would give the LSTM a cue the V3a
predictor never received, weakening the estimator-only comparison. Realised velocity
and attitude already describe ego-motion after the PID dynamics have acted. It is
stored in the dataset for a later ablation.

During blindness `d_meas = [0,0,0]` and `valid = 0`. The previous measurement is
**not** held at the input — carrying temporal information is the hidden state's job,
and handing it a stale value would let it avoid learning that.

**Output:** `[d, delta_v] ∈ R⁶`, world frame.

**Architecture (our choice, not the paper's):** LSTM(13 → 128, 1 layer) → MLP
(128 → 128 → 128 → 6). Small by design — the input is 13 metric numbers, not a
512-dim image embedding.

**Standardisation:** per-component, from the **training split only**. The actor's
clipping bounds (`REL_POS_BOUND=5`, `REL_VEL_BOUND=4`) were explicitly **not** used as
loss scales; they are observation-clipping limits far wider than the data spans, and
using them would leave the loss dominated by whichever component has larger raw
magnitude. All errors reported in physical units.

**Sequence handling:** full episodes (~300 steps), padded with a loss mask. No
truncated BPTT. Hidden state zeroed **only** at real episode reset, exactly as at
deployment — resetting every short training window would train a different estimator
from the one deployed.

**Split:** by episode, stratified by curriculum level. No timestep or sequence appears
in more than one split.

---

## 4. Two collection rounds, and why a second was needed

### 4.1 Round 1 and the offline gate

538 episodes / 63,622 steps from privileged `moving_025` checkpoints, V3a checkpoints,
a scripted controller and a small capped quantity of random actions, stratified across
all 12 levels.

Held-out validation passed decisively — LSTM 4–5× better than constant velocity beyond
15 blind steps, and 5–7× better on velocity in every bin.

### 4.2 The closed-loop check failed, and the first verdict was misleading

A passive 2×2 cross-check (policy × level, LSTM never driving) exposed a failure the
aggregate hid. In the `v3a @ L12` cell the LSTM **lost in four of eight motion bins**:

| motion bin | CV | LSTM | ratio |
|---|---|---|---|
| slow 0.05–0.15, 1–60 | 0.1119 | 0.1543 | **1.38** |
| med 0.15–0.30, 1–60 | 0.1449 | 0.1599 | **1.10** |
| med 0.15–0.30, >60 | 0.4460 | 0.5406 | **1.21** |
| fast >0.30, >60 | 1.1831 | 1.2234 | **1.03** |

**The pooled verdict reported "LSTM better" anyway**, because that cell's long-blind
frames were dominated by 1033 near-stationary samples where the LSTM trivially wins,
dragging the mean down. The verdict was subsequently changed to be computed **per
motion bin**, with `still` bins excluded from the pass criterion — winning where
nothing moves is not evidence of bridging dropout.

An earlier single-cell check had also reported the LSTM 17× better at `>120` blind
age, with error *decreasing* as dropout lengthened. That was an artefact of the same
effect: at level 9 the V3a policy hovers, and its

> 60-step stretches averaged `|delta_v|` = 0.089 m/s against 0.161 over all stretches.
> Long dropout there happens precisely when nothing is moving.

### 4.3 The gap, and the targeted second round

The failure needed **both** a fast platform and long dropout. Neither alone broke the
estimator, and the conjunction was structurally absent from round 1:

- privileged sources at levels 11–12 **land too fast to go blind for long** — 30
  episodes at L12 produced 5 stretches over 60 steps, max 94
- V3a sources that hover for hundreds of steps live at levels 3–10, where the platform
  is slow or stationary

Round 2 targeted exactly that conjunction: V3a checkpoints and a deliberately sloppy
scripted controller at levels 10–12, filtered on a **measured** criterion — an episode
kept only if it contained a blind stretch ≥60 steps during which mean ground-truth
`|delta_v|` ≥ 0.12 m/s.

**152 episodes kept of 270 (56%), 28,896 steps.** Hard stretches: mean 155 steps, max
288; their mean `|delta_v|` 0.194 m/s, max 0.474. Merged with round 1 by episode,
giving ~31% hard-regime — enough to matter without crowding out the easy regimes the
estimator also has to handle.

### 4.4 Both gates passed on round 2

Held-out validation (harder test set than round 1, so absolute numbers rose for
**both** estimators):

| blind age | CV pos | LSTM pos | CV vel | LSTM vel |
|---|---|---|---|---|
| 61–120 | 0.6492 | 0.1768 | 0.2234 | 0.0504 |
| >120 | 1.1623 | 0.2635 | 0.1885 | 0.0514 |

Passive 2×2 cross-check: **18 moving motion bins, 0 lost, all four cells PASS.** The
previously failing `fast >0.30, >60` bin went from 1.2234 to 0.5861 against CV's
1.1831.

**The cost, recorded honestly:** on *visible* frames CV remains better (0.0118 vs
0.0230 held-out; ratios 1.88–2.11 across the 2×2 cells). The estimator trades roughly
1–1.5 cm of accuracy on fresh detections for its gains elsewhere.

---

## 5. Results

### 5.1 Curriculum progress

| | V3a (constant velocity) | V3b seed 1 | V3b seed 2 |
|---|---|---|---|
| level reached in 600k | 4/12 | 5/12 | 7/12 |
| highest level **held** | 4/12 | 5/12 | **6/12** |

"Level reached" counts promotions; "held" is the highest level the policy actually
flew at the promotion threshold. Seed 2 was promoted to level 7 and then failed it in
seven consecutive evaluations (0, 10, 10, 10, 0, 10, 20%), never recovering. Seed 2
held level 6 — one above seed 1, not two.

Promotions, seed 1: 1→2 at 250k, 2→3 at 325k, 3→4 at 390k, 4→5 at 550k.

Promotions, seed 2: 1→2 at 465k, 2→3 at 535k, 3→4 at 545k, 4→5 at 550k, 5→6 at 560k,
6→7 at 565k.

#### Level-1 duration is seed noise, not an estimator effect

Seed 1 cleared level 1 at 245k, seed 2 at 465k — a 1.9× spread on identical
configuration. Level-1 duration has very high variance and cannot carry any claim.

**The "abrupt breakthrough" reading from the seed-1-only version of this report is
withdrawn.** Seed 1 went 0% at 200k → 25% → 50% → 80% → 90% at 245k, which looked like
a discrete transition. Seed 2's level 1 instead oscillates for 465k steps: 25% as early
as 40k, 65% at 300k, 0% at 435k and again at 445k, 80% at 450k, 5% at 455k, 80% at
460k, 90% at 465k. Nothing clicked. The seed-1 trace was one draw from a noisy process.

#### The promotion rule is a single 20-episode sample

Promotion requires ≥90% on one 20-episode evaluation. Seed 2 reached 80% four separate
times at level 1 (400k, 420k, 450k, 460k) without promoting, with a 5% evaluation
interleaved between two of them.

At a true success rate of 80%, a 20-episode sample clears the ≥90% bar about one time
in five; at 75%, about one in eleven. Once capability is near threshold, promotion
timing is dominated by sampling luck rather than by learning. **This is why "level
reached in 600k" is a weak statistic and why the matched-task evaluation in §5.2 is the
comparison that carries weight.**

#### What does differ after the breakthrough

Seed 1 took 300k steps for four promotions. Seed 2 took 100k for five, clearing levels
3, 4, 5 and 6 at 95%, 90%, 95% and 90% — levels 4 and 6 on their first evaluation at
that level. Clearing at threshold on first attempt is roughly a 9% event if true
capability were 75%, and it happened twice, so the cascade reflects genuine capability
rather than favourable draws.

Post-breakthrough promotion rate is therefore *faster* on seed 2 while level-1 duration
is *slower*. The two phases behave differently and should not be summarised as a single
number.

### 5.2 Evaluation at a matched task

**This section previously reported each run at whatever level it happened to reach.
That is not a comparison** — three runs on three different tasks. All numbers below are
at one fixed task.

100 deterministic episodes, seed base 9000, radius 0.11–0.23 m (level 5), stationary
platform, pinned via `--radius-min/--radius-max/--speed`.

| | V3a | V3b seed 1 | V3b seed 2 |
|---|---|---|---|
| run | `run_20260912_201312` | `run_20260914_195837` | `run_20260915_054052` |
| checkpoint | `best_success_model` | `best_success_model` | `best_model` |
| **success** | 71.0% | **85.0%** | 67.0% |
| precision | 6.21 ± 2.34 cm | **5.31 ± 2.45 cm** | 7.15 ± 2.22 cm |
| total failures | 29 | 15 | 33 |
| — timeout | 23 | 15 | 33 |
| — **`below_platform`** | **6** | **0** | **0** |
| `below_platform` / failures | **20.7%** | **0 / 15** | **0 / 33** |
| blind fraction | 60.6% | 53.1% | 58.7% |
| longest blind run | 108 steps | 78 steps | 111 steps |
| position est. error | **0.01104 m** | 0.0125 m | 0.0110 m |
| velocity est. error | **0.18371 m/s** | 0.218 m/s | 0.188 m/s |

**Protocol control.** The same V3a checkpoint re-evaluated at radius ≤0.20 returns
73.0%, 8 `below_platform`, 19 timeouts, precision 6.278 ± 2.170 cm, blind fraction
58.4% — reproducing the row from the earlier version of this report to every digit.
The checkpoint and protocol behind the V3a column are confirmed, not inferred.

#### Success rate cannot resolve this comparison; failure mode can

The two V3b seeds differ by **18 points** on identical task, protocol and episode seeds
(85.0% vs 67.0%). That spread is larger than the difference between either seed and
V3a (+14 and −4 points). **Depending on which seed was drawn, success rate says V3b is
substantially better or slightly worse.** At n=2 it does not answer the question.

`below_platform` is unambiguous over the same runs: 6 for V3a, 0 for both V3b seeds.
The two seeds disagree by 18 points on success and agree exactly on the failure mode.

If `below_platform` were a by-product of failing more often, seed 2's 33 failures at
V3a's matched-task rate of 6/29 would be expected to produce about 7. It produced none.
Probabilities of zero under that null: seed 1 alone (0 of 15) p ≈ 0.03, seed 2 alone
(0 of 33) p ≈ 5×10⁻⁴, pooled (0 of 48 against an expected 9.9) p ≈ 1.5×10⁻⁵.

**This refutes the confound rather than leaving it open**, and it does so with the
harder comparison: seed 2 has *lower* success than V3a (67.0% vs 71.0%), *more*
failures (33 vs 29), and comparable blind exposure (58.7% / 111 steps against 60.6% /
108 steps).

#### The best mean estimation accuracy belongs to the policy that crashes

V3a has the lowest closed-loop estimation error of all three runs on both axes —
0.01104 m and 0.18371 m/s, against V3b seed 2's 0.0110 m / 0.188 m/s and seed 1's
0.0125 m / 0.218 m/s. It also produces all six stale descents.

**Mean estimation accuracy and stale-descent safety are measured in opposite directions
here, on identical frames.** Whatever the LSTM provides, it is not mean accuracy, and no
mean-error statistic can detect it. §5.4 measures what it is.

#### A note on sample size

An earlier 50-episode measurement of the V3a matched-task figure returned 64.0% — seven
points below the 100-episode value of 71.0%, because 32 of the first 50 seeds succeeded
against 39 of the second 50. It was recorded as provisional and not used. 100 episodes
at a pinned seed base is the minimum for a cross-run comparison in this project; 50 is
not.

### 5.3 The estimator was stable under closed-loop feedback

24 perception evaluations across the seed-1 run:

| | mean | std | min | max | first 12 | last 12 |
|---|---|---|---|---|---|---|
| position error (m) | 0.0118 | 0.0017 | 0.0093 | 0.0184 | 0.0118 | 0.0119 |
| velocity error (m/s) | 0.206 | 0.028 | 0.164 | 0.268 | 0.201 | 0.212 |

**No drift.** First-half and second-half means are within noise on both. Both metrics
correlate negatively with blind fraction (−0.57 position, −0.36 velocity), i.e. they
track evaluation difficulty rather than training progress.

This matters because it was a registered risk. The estimator had only ever been
validated **passively**; once the policy flies on its output it creates a state
distribution neither collection round covered. That feedback loop was written into the
trainer docstring in advance as the first hypothesis to test if V3b underperformed. It
did not materialise as degradation.

*(An early reading of the single final row — `vel err 0.241` — was taken as evidence of
degradation. The full series does not support that; the final evaluation simply
coincided with a hard one, blind fraction 74.8% and longest run 212 steps. Recorded
here because the mistake is instructive: a single perception sample is not a trend.)*

**This now holds on two seeds.** Seed 2's final perception evaluation reports 0.0132 m
position error and 0.189 m/s velocity error at 76.2% blind fraction with a 147-step
longest blind run — inside seed 1's envelope despite a harder evaluation distribution.
The matched-task evaluation gives 0.0110 m and 0.188 m/s at 58.7% blind. No drift, on a
second independent run.

### 5.4 The mechanism: error distribution, not error mean

**The estimator's offline accuracy advantage did not transfer to deployment as mean
accuracy.**

| | offline held-out | V3b closed-loop | V3a closed-loop (CV) |
|---|---|---|---|
| position error | 0.0230 m (visible) – 0.2635 m (>120) | 0.0118 m | 0.0120 m |
| velocity error | 0.0389 – 0.0514 m/s | 0.206 m/s | 0.180–0.226 m/s |

Closed-loop velocity error is **4–5× worse** than offline and sits in the same range as
V3a's constant-velocity predictor. Position error is **equal** to V3a's, not better.

So V3b outperforms V3a while its estimator shows **no measurable mean-accuracy
advantage in deployment**. The improvement cannot be attributed to "better estimates"
on mean error.

The question left open was whether it could be attributed to the estimator at all. It
can. The difference is in the **tails**, and the measurement is the fraction of frames
exceeding a crash-relevant threshold, reported per blind-age bin.

Both estimators run on the same camera stream in the same episodes under one policy at
a time, so this is a paired comparison on identical frames.

#### Under the V3a policy — the decisive run

These are the states V3a actually visited, and therefore the closest available answer
to "what would V3a have seen if it had had the LSTM".

50 episodes, radius 0.11–0.23 m, 8,777 frames, 6,206 of them blind.

| blind age | n | CV mean | LSTM mean | CV max | LSTM max | CV >0.3 m | LSTM >0.3 m | CV >1.0 m | LSTM >1.0 m |
|---|---|---|---|---|---|---|---|---|---|
| visible | 2571 | **0.0111** | 0.0194 | 0.045 | 0.078 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1–15 | 914 | 0.0358 | **0.0250** | 0.723 | **0.095** | 0.030 | **0.000** | 0.000 | 0.000 |
| 16–60 | 1996 | 0.1835 | **0.0514** | 2.757 | **0.271** | 0.068 | **0.000** | 0.056 | **0.000** |
| 61–120 | 1177 | 0.5650 | **0.1433** | 5.332 | **0.815** | 0.251 | 0.162 | 0.103 | **0.000** |
| >120 | 2119 | 1.3369 | **0.3180** | 11.243 | **2.187** | 0.739 | **0.224** | 0.127 | 0.092 |

Verdict: **supported.** Four of six testable cells suppressed, none lost, two partial.
Cells where the constant-velocity predictor itself stays below 5% exceedance are not
testable and are reported as such rather than counted as wins.

#### The single clearest number

**At blind age 16–60 steps the constant-velocity predictor exceeds one metre of position
error on 5.6% of frames, with a worst case of 2.76 m. Over the same frames the LSTM's
worst case is 0.27 m and no frame exceeds one metre.** At 61–120 it is CV 10.3% against
the LSTM's 0.000.

Against a 0.1 m success radius, a metre-scale position error during descent is a
commanded descent into empty space. CV begins producing those after roughly 1.6 seconds
blind; the LSTM produces none until past 12 seconds. That is the `below_platform`
mechanism, and it is invisible to any comparison of means — at 16–60 the means differ by
3.6×, which would not predict the difference between 5.6% and zero catastrophic frames.

CV's worst single error over the run is **11.24 m**. That is what V3a was flying on.

#### The offline visible-frame cost reproduces in closed loop

§4.4 recorded that on visible frames CV remains better offline — 0.0118 vs 0.0230
held-out, ratios 1.88–2.11. **That prediction holds in deployment: 0.0111 vs 0.0194
here (1.75×), and 0.0121 vs 0.0223 under the V3b policy (1.84×).** The crossover sits
at blind age 1–15; from 16 steps onward the LSTM leads on every statistic.

This is the same structure Shin et al. report for EKF against their learned estimator
(Fig. 6): higher accuracy for the filter under consistent detection, divergence during
extended dropout where the learned estimator continues to update. Observing it on a
different simulator, marker and estimator architecture — and having it transfer from
offline held-out to closed loop at the same ratio — is an independent confirmation of
the paper's central perceptual claim.

It also explains why mean closed-loop error came out equal in §5.2: the run is roughly
30–45% visible frames, where the LSTM is 1.8× worse, and 55–70% blind frames, where it
is 3–4× better. The two cancel in the mean.

#### Measuring under the improved policy understates the estimator

The same comparison under the **V3b policy** (41/50 success, 6,767 frames, 4,341 blind):

| blind age | n | CV mean | LSTM mean | CV max | LSTM max | CV >0.3 m | LSTM >0.3 m | CV >1.0 m | LSTM >1.0 m |
|---|---|---|---|---|---|---|---|---|---|
| visible | 2426 | **0.0121** | 0.0223 | 0.042 | 0.082 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1–15 | 825 | 0.0240 | **0.0219** | 0.191 | **0.074** | 0.000 | 0.000 | 0.000 | 0.000 |
| 16–60 | 1536 | 0.0892 | **0.0751** | 0.538 | **0.335** | 0.023 | **0.003** | 0.000 | 0.000 |
| 61–120 | 675 | **0.2245** | 0.2355 | 0.595 | **0.573** | 0.219 | 0.193 | 0.000 | 0.000 |
| >120 | 1305 | 0.4524 | **0.2592** | 1.346 | **1.248** | 0.661 | **0.174** | 0.053 | 0.027 |

Verdict: supported, but only **one of three** testable cells, because the
constant-velocity predictor is no longer dangerous in most bins under this policy.

**The same constant-velocity estimator, on the same task, is 3× worse in mean and 8×
worse in maximum under the V3a policy than under the V3b policy** (>120 bin: 1.337 m and
11.24 m against 0.452 m and 1.35 m). The policy does not merely experience the
estimator; it determines how hard the estimation problem is.

Two consequences:

1. **A policy flying on good estimates visits states where estimation is easier.** Part
   of the estimator's contribution is consumed by improving behaviour and is therefore
   invisible to any measurement taken under the improved policy. The number of
   *testable* cells falls from six to three — the measurement degrades as the thing it
   measures succeeds.
2. **Closed-loop estimator evaluation must state which policy generated the
   trajectories.** A single "closed-loop error" figure is not well defined without it.

#### The mechanism is conditional on blind fraction

The discarded level-7 checkpoint (§5.5) supplies the control this argument previously
lacked: the **same frozen estimator**, the **same task**, 34 `below_platform` failures
in 100 episodes. What differs is behaviour — 77.9% blind fraction and a 175-step longest
run, against 58.7% and 111 steps for the good checkpoint.

The tables above show why that matters: the LSTM's own exceedance is not zero past 120
steps blind (0.224 above 0.3 m, 0.092 above 1.0 m). A policy that spends most of its
time there defeats it.

So the defensible statement is conditional:

> The LSTM eliminates stale-descent failures for policies that maintain the visual
> contact the curriculum produces at levels ≤6. It does not eliminate them
> unconditionally.

This is stronger than the unconditional claim, because it names the governing variable —
blind fraction — and that variable is directly controllable by an active-perception
objective (§9).

### 5.5 Checkpoint selection failed a second time, in mirror image

The checkpoint callback compares candidates lexicographically on
`(curriculum_level, success_rate, −distance)`. Curriculum level dominates absolutely.

On seed 2 at 570k steps the rule compared a level-7 policy at 0% against the saved
level-6 policy at 90% and saved the level-7 one. The final log line — `NEW BEST saved at
level 7 (20.0%)` — is the rule working exactly as specified.

Evaluated on one fixed task (radius 0.11–0.23 m, stationary, 100 episodes, seed base
9000):

| | `best_model.zip` | `best_success_model.zip` |
|---|---|---|
| saved by | SB3 `EvalCallback` (mean reward) | `SuccessRateCheckpointCallback` |
| **success** | **67.0%** | 17.0% |
| precision | 7.15 ± 2.22 cm | 4.94 ± 2.54 cm |
| total failures | 33 | 83 |
| — timeout | 33 | 48 |
| — `below_platform` | **0** | **34** |
| — tilt_bound | 0 | 1 |
| blind fraction | 58.7% | 77.9% |
| longest blind run | 111 steps | 175 steps |

**The project's own checkpoint-selection rule discarded a policy four times better on
the same task, and the run's designated artefact is the worse one.** Seed 2 would have
been reported as a failed run had the file been taken at face value.

**Precision inverts.** The worse checkpoint has *better* touchdown precision (4.94 vs
7.15 cm) because it only succeeds on the easiest episodes. Precision conditioned on
success is not a quality measure when success rates differ this much.

**This is the mirror image of the earlier bug.** The ceiling bug froze saves at an easy
level and never promoted; this one promotes onto a level the policy cannot fly and can
never go back. Both follow from one mistake:

> **Curriculum level is a state variable, not a performance measure. No checkpoint
> selection rule should treat it as dominant.**

#### Recommended fix, before any further training runs

Keep a separate best checkpoint **per curriculum level** rather than one global best. No
level can then overwrite another, the best policy at any level stays recoverable, and
both failure modes are eliminated by construction. Cost is disk space, already dominated
by replay buffers.

#### A second analysis defect found and fixed in the same pass

`compare_error_distributions.py` decided support with `p99_ratio < mean_ratio × 0.6`.
When the LSTM wins everywhere, both ratios sit below 1 and the tail ratio cannot be 40%
smaller than an already-small mean ratio, so the rule printed "hypothesis NOT supported"
while the tables it had just printed showed the LSTM winning substantially. It also
pooled across all blind frames — the same pooling error §4.2 records for the round-1
offline check.

Replaced with per-bin threshold-exceedance verdicts. **Both defects had the same shape:
a summary statistic that could not express the effect it was asked to detect.** Neither
changed a measurement; both changed what the measurements appeared to say.

---

## 6. What V3b established

- Replacing the constant-velocity predictor with a frozen LSTM estimator advanced the
  curriculum from 4/12 to **5/12 and 6/12 held across two seeds**, at matched 600k
  budget and configuration.
- **`below_platform` failures are eliminated across both seeds at matched task: 0 of 48
  failures, against V3a's 6 of 29.** Under the null that the rate is unchanged,
  p ≈ 1.5×10⁻⁵ pooled (seed 1 alone p ≈ 0.03, seed 2 alone p ≈ 5×10⁻⁴). The
  descend-onto-stale-position failure mode identified in V2 and V3a is gone.
- **The elimination is not explained by higher success rate.** Seed 2 has *lower* success
  than V3a at the same task (67.0% vs 71.0%), more failures (33 vs 29) and comparable
  blind exposure, and still produces zero. The confound was tested, not assumed away.
- **It is also not explained by better mean estimation accuracy. V3a has the best of the
  three** — 0.01104 m and 0.18371 m/s, against seed 2's 0.0110 m / 0.188 m/s and seed 1's
  0.0125 m / 0.218 m/s — and produces all six stale descents.
- **The mechanism is the tail of the error distribution.** At blind age 16–60 the
  constant-velocity predictor exceeds 1 m on 5.6% of frames (worst case 2.76 m) while the
  LSTM exceeds it on none (worst case 0.27 m); at 61–120, 10.3% against 0.000. CV's worst
  single error under the V3a policy is 11.24 m.
- **The offline visible-frame cost reproduces in closed loop.** §4.4 measured CV 1.88–2.11×
  better than the LSTM on visible frames held-out; deployment gives 1.75–1.84×. The
  estimators cross over at blind age 1–15. This reproduces the EKF-versus-learned-estimator
  crossover reported by Shin et al. (Fig. 6) and explains why the means cancel in §5.2.
- **The effect is conditional on blind fraction.** At 77.9% blind the same frozen estimator
  yields 34 `below_platform` failures in 100 episodes. Every policy the curriculum produced
  at levels ≤6 satisfies the condition.
- **Closed-loop estimator error is not well defined without naming the policy.** The same
  predictor is 3× worse in mean and 8× worse in maximum under the V3a policy than under the
  V3b policy, on the same task.
- The estimator is **stable under closed-loop feedback** on both seeds — no drift (§5.3).
- Offline validation **can be actively misleading** when pooled across motion regimes
  (§4.2), and so can a closed-loop verdict rule (§5.5).
- The hard regime for a temporal estimator here is the **conjunction** of long dropout and
  high relative motion; neither alone breaks it (§4.3).

---

## 7. What V3b did NOT establish

- **It did not reach the moving task.** Levels 1–7 are all stationary radius expansion —
  `platform_speed_range` is 0.0 at every level either seed reached. Neither seed has
  encountered a moving platform. No comparison against V2's 98.0% / 53.3% is valid, and
  "6/12" is not partial progress toward the moving benchmark.
- **Whether the success-rate improvement is real.** Two seeds give 85.0% and 67.0% on
  identical task and protocol. At n=2 the success figure supports no claim. *(The
  failure-mode result does not share this weakness — §6.)*
- **Whether the level count means anything.** Level-1 duration varied 1.9× between seeds and
  promotion fires on a single 20-episode sample, which at near-threshold capability clears
  about one time in five. Level reached is substantially a draw.
- **Whether the timeout/hovering failure mode would be overcome with more budget.** All 48
  V3b failures at matched task are timeouts. The hovering that dominated V3a survives a
  better estimator intact.
- **Whether a validity flag would add anything** — deliberately deferred.
- **Whether the LSTM handles non-constant target dynamics.** It cannot be claimed from this
  work: `_samplePlatformMotion()` draws one constant velocity per episode, so the
  constant-velocity predictor has the **correct model class**, and all measured advantage is
  denoising, multi-measurement integration and dropout bridging. The crossover finding
  sharpens this — under consistent detection the correctly-specified predictor wins.

---

## 8. Interpretation

The defensible statement:

> A frozen, offline-trained LSTM temporal estimator eliminated the
> descend-onto-stale-position failure mode entirely across two seeds — zero occurrences in
> 48 failures at matched task, against 20.7% for the constant-velocity predictor it
> replaced — while producing no resolvable change in success rate and no improvement in
> mean closed-loop estimation accuracy. The elimination is attributable to the suppression
> of metre-scale excursions during dropout, and holds conditionally on the policy
> maintaining the visual contact the curriculum produces at levels ≤6. Neither seed reached
> the moving-platform task within 600k.

### The central result is a metric result, not a performance result

At matched task the two V3b seeds score 85.0% and 67.0% against V3a's 71.0%. The
between-seed spread of 18 points is larger than either seed's difference from V3a (+14 and
−4). **Success rate points in opposite directions depending on which seed was drawn, so at
n=2 it does not resolve the comparison at all.**

`below_platform` resolves it immediately, and the two seeds agree exactly despite that
18-point disagreement: V3a commands a descent onto empty space 6 times in 100 episodes,
both V3b seeds never once — seed 2 while failing *more often* than V3a (33 vs 29) and
succeeding *less* (67.0% vs 71.0%).

**The high-variance metric cannot decide; the low-variance one decides decisively.** That
is the claim this project was built to demonstrate, and this is the cleanest instance of
it: a binary success metric records a categorical safety improvement as seed noise.

The pattern recurs three more times inside this stage:

- V3a has the **best mean estimation accuracy of the three runs** (0.01104 m, 0.18371 m/s)
  and produces all six stale descents (§5.2).
- In §5.4 mean closed-loop error is equal between an estimator whose worst excursion is
  11.24 m and one whose worst is 2.19 m, because the visible-frame penalty cancels the
  blind-frame advantage.
- In §5.5 the discarded checkpoint has **better touchdown precision** than the one four
  times more likely to land at all.

**Four summary statistics, in one stage, each hiding or inverting the effect that actually
determines outcomes.** Success rate, mean position error, mean velocity error, and
precision. This is no longer an argument about one metric; it is a demonstration that
aggregate scalars systematically fail on this task.

### What the improvement is not

Attributing it to "better state estimation" overstates what was measured. Mean accuracy was
equal, and under consistent detection the LSTM is measurably *worse*. What it provides is
bounded error during dropout — robustness, not accuracy — and §5.4 supplies that
measurement directly.

### The persisting failure is still hovering

All 48 matched-task failures across both seeds are timeouts. The hovering behaviour that
dominated V3a survives a better estimator completely unchanged. This is consistent with the
V3a §8 argument that hovering is a rational response to an input the policy cannot verify —
and it is notable that improving the estimate without telling the policy *when to trust it*
did not reduce it at all.

---

## 9. Suggested next step

### Fix the checkpoint rule before any further training runs

Per-level best checkpoints (§5.5). Every run launched before this change carries a defect
that has now produced two distinct failures in opposite directions.

### Then: the validity-flag ablation

**The motivation is now sharper.** V3b improved the estimate on two seeds, eliminated the
stale-descent failure mode, and hovering persisted *completely unchanged* — 48 of 48
matched-task failures are timeouts. If hovering is driven by the actor's inability to tell
fresh from stale rather than by estimate quality, a validity or estimate-age signal should
affect it where a better estimator demonstrably did not. Clean one-variable test, now with
a control on both sides.

Requires a 16-D observation and therefore a fresh run; cannot reuse these checkpoints.

### Then: active perception

§5.4 identifies blind fraction as the variable governing whether the estimator's advantage
appears at all, and §5.5 shows a policy at 77.9% blind fraction defeating the same
estimator entirely. That makes an active-perception reward — penalising actions that
increase future estimation error — a directly motivated next variable rather than a
borrowed idea from the reference paper.

Sequence it after the validity-flag ablation: it changes the reward, the validity flag
changes the observation, and they should not move together.

**Algorithm note, unchanged.** TD3 stays through the validity-flag ablation. The switch to
PPO belongs where the asymmetric privileged critic is introduced, because that is where TD3
fights the architecture — and when it happens, a privileged-state PPO control run is
required or vision and algorithm changes will be confounded.

---

## Appendix — provenance

Every figure in §5.2 and §5.5 comes from a command with the task explicitly pinned, run on
19 Sep 2026. No number is carried forward on trust.

| Row | Verification |
|---|---|
| V3a @ ≤0.20 | Re-run reproduced 73.0% / 8 `below_platform` / 19 timeouts / 6.278 ± 2.170 cm / 58.4% blind — the earlier §5.2 row to every digit |
| V3a @ ≤0.23 | New measurement, 100 ep |
| V3b seed 1 | Re-run reproduced 85.0% / 15 timeouts / 0 `below_platform` / 5.305 ± 2.448 cm / 53.1% blind — the earlier §5.2 row to every digit, confirming `best_success_model.zip` was the checkpoint |
| V3b seed 2, both checkpoints | New measurements, 100 ep each |

Both historical rows reproduced exactly, which also validates the evaluation protocol as
deterministic at a fixed seed base.

### Standing requirement for every evaluation in this project

`--speed 0.0` must be passed explicitly. The scripts default to 0.25 m/s, and no checkpoint
in this project has trained on a moving platform; omitting it silently evaluates on a task
the policy has never seen. This is not hypothetical — it invalidated one evaluation during
this analysis and produced an 11% success figure that meant nothing.

Likewise `--radius-min/--radius-max`: an evaluation without them scores whatever level the
run happened to reach, which is not a comparison. This is the defect §5.2 exists to correct.

### Commands

```bash
cd ~/mtp/drone-landing-rl/src/scripts

# V3b checkpoint at a pinned task
python train_lstm_v3b.py --eval \
  --model results_lstm_v3b/run_<TIMESTAMP>/best_model.zip \
  --estimator ../models/temporal_estimator_v3b_r2.pt \
  --episodes 100 --eval-seed-base 9000 \
  --radius-min 0.11 --radius-max 0.23 --speed 0.0

# V3a checkpoint at a pinned task (no --estimator; flies on constant velocity)
python train_aruco_v3a.py --eval \
  --model results_aruco_v3a/run_20260912_201312/best_success_model.zip \
  --episodes 100 --eval-seed-base 9000 \
  --radius-min 0.11 --radius-max 0.23 --speed 0.0

# Paired estimator error distributions under one policy
python compare_error_distributions.py \
  --estimator ../models/temporal_estimator_v3b_r2.pt \
  --policy <POLICY.zip> \
  --obs-source <lstm|predict> \
  --radius-max 0.23 --speed 0.0 --episodes 50 \
  --out dist_<NAME>.json
```

Curriculum levels used above: L4 = radius ≤0.20, L5 = ≤0.23, L6 = ≤0.26, L7 = ≤0.30, all
at `platform_speed_range` 0.0.
