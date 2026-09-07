# Mid-Semester Presentation — Slide Structure

**Project:** RL-based autonomous multirotor landing on a moving platform
**Format:** ~15 slides, ~15 minutes, panel of two reviewers plus supervisor
**Weight:** 25% of the MTP grade

---

## How to use this document

Each slide below has three parts:

- **ON THE SLIDE** — the minimum that should appear. Keep slides sparse; the
  panel reads faster than you talk, and a dense slide means they stop
  listening.
- **SAY** — roughly what you say while it is up. Not a script to memorise, but
  the argument you are making.
- **WHY** — why this slide exists in this position. Read these once; they are
  the reasoning behind the order.

**The single most important structural rule:** establish that success rate is
inadequate *before* showing any of your numbers. Your headline result is 45.7%
success where the literature reports 97–100%. Presented without setup, that
reads as "my method is worse." Presented after the gap slide, the same number
reads as "the literature is measuring the wrong thing, and here is the
evidence." Same data, opposite conclusion. Order is doing that work.

---

# SLIDE 1 — Title

**ON THE SLIDE**
- Reinforcement Learning for Autonomous Multirotor Landing on Moving Platforms
- Your name, roll number
- Supervisor's name
- Department of CSE, IIT Ropar
- Mid-semester evaluation, date

**SAY**
Name, one sentence on the project, and move on within fifteen seconds.

---

# SLIDE 2 — The problem

**ON THE SLIDE**
- A diagram: drone above a platform, arrow showing the platform sliding
- Three bullets:
  - Track a target that does not stay still
  - Match its velocity — right place at the wrong speed means a sideways impact
  - Decide *when* to descend

**SAY**
Landing on a stationary pad is solved; a PID controller does it. Three things
make a moving platform different. The drone has to follow something that moves.
It has to match the platform's velocity, because arriving in the right place
while still sliding relative to the platform means hitting sideways and
tumbling. And it has to decide when to commit to the descent — too early and it
lands on open floor, too late and it runs out of time. That third one is where
most failures happen.

**WHY**
The panel needs to know the task is non-trivial before they will care about
your metrics. Thirty seconds, no more.

---

# SLIDE 3 — Related work

**ON THE SLIDE**

| Work | Method | Reported success |
|---|---|---|
| Goldschmid & Ahmad, *Auton. Robots* 2024 | Tabular Double Q-Learning + curriculum | 99% sim, 68–79% real |
| TornadoDrone, 2024 | TD3, gym-pybullet-drones | 100% on several scenarios |
| Shin et al., *IEEE RA-L*, May 2026 | PPO + LSTM, vision | 97% |

**SAY**
Three works define the current state. Goldschmid and Ahmad use tabular
Q-learning with a curriculum and are the only ones reporting real-hardware
statistics. TornadoDrone uses TD3 on the same simulator I use. Shin et al. in
RA-L this year is the state of the art — vision-based, with a learned state
estimator.

**WHY**
Establishes you know the field, and sets up the next slide. Do not editorialise
here; just present what they did.

---

# SLIDE 4 — The gap ★ THE PIVOTAL SLIDE

**ON THE SLIDE**

> All three report **binary success rate**.
> All three report **97–100%**.
> **None reports how hard the drone hit.**

Then, in a box:

> A saturated metric cannot discriminate between methods.

**SAY**
Here is what they have in common. All three report a binary success rate —
did the drone end up on the pad, yes or no. And all three report between 97
and 100 percent.

A metric where everyone scores 99% cannot tell you which method is better. And
none of them reports the quantity that determines whether a real airframe
survives: how fast it was going when it touched down, how much sideways motion
it had, how tilted it was.

Shin et al. state this themselves — their listed limitation is the absence of
any safety guarantee.

**WHY**
This is the slide the whole presentation turns on. Spend time on it. Everything
after this is either evidence for this claim or a consequence of it.

---

# SLIDE 5 — What I measure

**ON THE SLIDE**
- Diagram: drone at the moment of contact, four labelled arrows
  - Vertical speed at touchdown (m/s)
  - Lateral speed relative to platform (m/s)
  - Tilt angle at contact (degrees)
  - Offset from pad centre (m)
- Proposed thresholds: vz ≤ 0.5 m/s · lateral ≤ 0.3 m/s · tilt ≤ 10°

**SAY**
So I measure four quantities at the instant of contact, and report a violation
rate against physically motivated thresholds. A drone touching down beyond ten
degrees of tilt catches a leg and flips. One arriving with lateral velocity
tumbles. These are the things that break hardware, and they are not in any of
the three papers.

**WHY**
Your contribution, stated plainly, immediately after the gap it fills.

---

# SLIDE 6 — Research questions

**ON THE SLIDE**
1. Does success rate hide safety failures that touchdown metrics reveal?
2. How much do abstract sensing models overstate real performance?
3. Does a recurrent policy recover performance lost to sensor dropout?
4. How much does seed variance affect each metric?

**SAY**
Four questions. I have answers to all four, and one of them surprised me.

**WHY**
The evaluation policy explicitly requires "hypotheses or research questions are
well defined." This slide is graded. Keep it.

---

# SLIDE 7 — Method: the environment

**ON THE SLIDE**
- Screenshot of the PyBullet simulation showing drone, pad, and ArUco marker
- Left column, "Built on":
  - PyBullet + gym-pybullet-drones
  - Crazyflie 2.X — 27 g, thrust-to-weight 2.25
  - Ground effect modelled
- Right column, "Written for this project":
  - Moving platform, randomised every episode
  - Relative-state observations
  - Five-term reward
  - Touchdown detection and metrics
  - Sensing model

**SAY**
The physics come from gym-pybullet-drones — rotor thrust, drag, and
importantly ground effect, which acts precisely during the final descent where
my metrics are measured. Goldschmid identified the absence of ground effect in
their simulator as one cause of their thirty-one point sim-to-real collapse.

The drone is a Crazyflie 2.X, a real commercial nano-quadrotor. I did not
design it.

Everything on the right I wrote. There is no landing environment in the
library.

**WHY**
Pre-empts "what did you actually build?" — a question that will otherwise come
during questions and cost you more time.

---

# SLIDE 8 — Method: the reward

**ON THE SLIDE**

```
reward = alignment + velocity_matching + progress + tilt_penalty + touchdown_bonus
```

Two boxes:

**Progress, not proximity**
Distance rewards create places to park. Progress is zero when hovering, so it
cannot be farmed by standing still.

**The descent gate**
Progress only pays within 1.5 pad-widths. Encodes the commit-timing decision.

**SAY**
Five terms. Two of them are design decisions worth explaining.

Early versions rewarded being *near* the pad. That creates a place to park — if
hovering at some altitude pays well per step, the policy hovers there forever.
Six reward variants did exactly that. So the main term is now progress, a
function of change: positive descending, negative climbing, exactly zero
hovering. It cannot be farmed by standing still.

Second, progress only pays while the drone is horizontally over the pad.
Without that gate the policy dives immediately from wherever it happens to be
and lands on open floor. That gate is the commit-timing decision.

**WHY**
The policy requires "justification for the chosen approach and parameters."
These two decisions have real reasoning behind them; show it.

---

# SLIDE 9 — Method: sensing ★

**ON THE SLIDE**
- Camera image from the simulation with the ArUco marker visible
- Three modes:

| Mode | What the drone gets |
|---|---|
| Privileged | Exact relative state from the simulator |
| Abstract | Modelled UWB noise + hand-set dropout threshold |
| Camera | Rendered 128×96 image → `cv2.aruco` → `solvePnP` |

- The geometry, boxed:

> Camera at height *h* sees a footprint of 2h·tan(θ/2).
> Marker of side *s* leaves the frame at h = s/(2·tan(θ/2)) = **0.18 m**
> Measured: 0.195 m → 0.123 m

**SAY**
Three sensing modes, selectable by one argument, which makes them a clean
ablation.

Privileged state is exact values from the simulator — not physically
realisable, used to isolate the control problem from perception.

Abstract mode models UWB noise and an ArUco dropout at a threshold I set by
hand.

Camera mode renders an actual image and runs real OpenCV ArUco detection on it.
Pose comes from solvePnP on the four detected corners.

The important number is at the bottom. The marker leaves the camera's frame at
a height determined purely by geometry — I derived 0.18 metres, and measured
between 0.195 and 0.123. So the drone goes blind for the last twenty
centimetres of every landing, and that is a consequence of optics, not a
threshold I chose.

**WHY**
Directly answers "how does the drone see the platform?" — the question your
supervisor already asked. The derived-then-verified number is the strongest
technical detail in the presentation.

---

# SLIDE 10 — Experimental setup

**ON THE SLIDE**
- Platform motion: amplitude 0.2–1.0 m, ω 0.2–1.5 rad/s, phase 0–2π,
  **resampled every episode**
- PPO, 1.5M timesteps, 4 parallel environments
- **3 training seeds per condition**, 100 evaluation episodes each
- Deterministic evaluation, fixed seeds

**SAY**
Platform motion is randomised every episode — amplitude, frequency, and phase.
I will explain on a later slide why that matters more than it sounds.

Three training seeds per condition, a hundred evaluation episodes each. That is
more than any of the three comparison papers reports.

**WHY**
Rigour is graded. Say the seed count out loud.

---

# SLIDE 11 — Results

**ON THE SLIDE**

| Condition | Seeds | Success | vz (m/s) | lateral (m/s) | vz > 0.5 | lat > 0.3 |
|---|---|---|---|---|---|---|
| Privileged state | 3 | 83.3 ± 3.1% | 0.047 | 0.068 | 0% | **0%** |
| Abstract sensing | 3 | **96.7 ± 0.9%** | 0.162 | 0.270 | 0% | **34.9%** |
| Abstract + LSTM | 1 | 79.0% | 0.302 | 0.209 | 0% | **24.1%** |
| Camera + ArUco | 3 | **45.7 ± 0.5%** | 0.461 | 0.290 | **43.5%** | **40.7%** |

Highlight the success column and the two violation columns in different
colours.

**SAY**
Four conditions. Read the success column first, then the violation columns.
Three findings come out of this table, one per slide.

**WHY**
One table, then unpack it. Do not try to explain everything while it is up.

---

# SLIDE 12 — Finding 1: worse sensors, higher success ★

**ON THE SLIDE**

| | Privileged | Abstract sensing |
|---|---|---|
| Success | 83.3% | **96.7%** ↑ |
| Lateral violations | **0%** | **34.9%** ↑ |

> Under the metric the field reports, degrading the drone's sensing
> **improved** the system.

**SAY**
Degrading the sensing raised the success rate by thirteen points — and created
a thirty-five percent safety violation rate that did not previously exist.

The reason is behavioural. With perfect information the policy can hover,
refine its alignment, and descend when conditions are ideal. With noisy
information that cuts out near the pad, waiting does not help — more time means
more noise and more chance of losing the marker. So committing early genuinely
maximises reward. It lands more often, and it lands harder.

The reward pays for landing. It does not sufficiently punish landing badly.
And success rate cannot tell the difference.

**WHY**
Your first finding, and the one that most directly validates the gap slide.

---

# SLIDE 13 — Finding 2: abstract models are optimistic ★

**ON THE SLIDE**

| | Abstract sensing | Camera + ArUco |
|---|---|---|
| Success | **96.7%** | **45.7%** |
| Blind fraction | ~30% (assumed) | **~88%** (measured) |

> A 51-point gap. Same policy, same reward, same platform motion, same budget.

**SAY**
This one surprised me. My abstract sensing model, with plausible hand-set
parameters, gave 96.7 percent. Replacing it with the rendered camera and real
detection — nothing else changed — gave 45.7.

The mechanism is measurable. My model assumed the drone would be blind about
thirty percent of the time, from a vertical field-of-view threshold. Measured
blind fraction with a real camera was eighty-eight percent. The reason: with a
moving platform the drone drifts up to seventy centimetres sideways while
descending, and the marker leaves the frame laterally long before height
matters. My model accounted for only one of two dropout mechanisms.

This is a methodological claim, not just a claim about my drone. Abstract
sensor models should be validated against a rendered pipeline before
conclusions are drawn from them.

**WHY**
The strongest generalisable claim in the work, and something none of the three
papers does.

---

# SLIDE 14 — Finding 3: success is stable, safety is not ★★

**ON THE SLIDE**

Camera condition, three seeds, identical code:

| Seed | Success | vz > 0.5 m/s |
|---|---|---|
| 1 | 45% | **0.0%** |
| 2 | 46% | **65.2%** |
| 3 | 46% | **65.2%** |
| **spread** | **± 0.5%** | **± 30.7%** |

> Seed variance is not uniform noise. It is concentrated entirely in the
> dimension nobody measures.

**SAY**
This is the sharpest result in the project.

Three training runs. Same code, same reward, same budget — only the random seed
differs. Success rate: 45, 46, 46 percent. A spread of half a point. A reviewer
looking at that would call it highly reproducible and move on.

The same three policies: one never exceeds the touchdown velocity limit. The
other two exceed it in roughly two landings out of three.

So seed variance is not uniform noise spread across all measurements. It is
concentrated entirely in the dimension nobody measures.

**SAY IF ASKED "so what?"**
It means two things. Single-run results in this area are not trustworthy on
safety. And it is a direct argument for constrained RL — a hard constraint
would have bound seeds two and three, where a fixed reward weight did not.

**WHY**
Your best slide. Pause after saying the numbers.

---

# SLIDE 15 — Method development: two bugs found

**ON THE SLIDE**

**Bug 1 — the landing bonus was never paid**
`_computeReward()` runs *before* `_computeTerminated()`, where touchdown was
detected. Eight reward variants failed for this one reason.
*Found by:* a scripted descent with fixed motor commands, printing reward
against height.

**Bug 2 — the policy memorised a timed routine**
Reported 100% success. A frequency sweep gave non-monotonic results — failing
at slower motion than it trained on, succeeding at a harmonic.
*Fix:* randomise amplitude, frequency, **and phase** every episode.
*Honest result:* 83.3%.

**SAY**
Two things worth showing about how the work was done.

The first: eight consecutive reward designs gave zero successful landings. The
drone tracked the platform perfectly and then hovered above it. The cause was
an ordering bug — the reward function runs before the code that detects
touchdown, so the landing bonus was never paid, at any value. The agent had no
evidence landing was worth anything and correctly learned to hover. I found it
by flying a scripted descent with no neural network at all and printing the
reward at each height, which separated "can the environment produce a landing"
from "is the reward encouraging one."

The second: after fixing that, I got 100 percent success. I did not trust it,
so I tested the policy at platform speeds it had not trained on. The results
were non-monotonic — it failed at motion *gentler* than its training, and
succeeded again at a harmonic of the trained rhythm. A physical limit produces
a clean threshold; that pattern is a timing artefact. The policy had memorised
a timed routine rather than learning to track, because the platform did the
same thing every episode. Randomising the motion brought the honest number down
to 83 percent.

**WHY**
Panels respond to demonstrated diagnostic ability. "I got 100%, distrusted it,
and found out why" is stronger than a clean 100% would have been. Do not skip
this slide to save time.

---

# SLIDE 16 — Video

**ON THE SLIDE**
A 10–15 second screen recording of one successful landing, on loop. Overlay the
platform speed and the four touchdown metrics if you can.

**SAY**
Very little. Let it play. "Platform moving at 0.63 metres per second, touchdown
at 4.3 centimetres per second, four centimetres from centre."

**WHY**
The panel has been reading tables for ten minutes. One moving image resets
their attention and proves the system is real. Record this well before the
presentation, not the night before.

---

# SLIDE 17 — Limitations

**ON THE SLIDE**
- Camera model: no motion blur, lighting variation, or lens distortion
- No classical baseline yet — Goldschmid's tuned PI scored 100% in simulation
- Reward-weight sensitivity not yet ablated
- LSTM condition has one seed
- Simulation only; hardware is semester 2

**SAY**
Five honest limitations. The two that matter most: I do not yet have a tuned
classical baseline, and Goldschmid's PI controller scored a hundred percent in
every simulation scenario, so that is not a straw man. And I have not yet shown
that finding one survives changes to the reward weights — a reviewer could
reasonably say my reward is badly shaped, and answering that is the next
experiment.

**WHY**
State your weaknesses before the panel does. It converts an attack into
evidence of self-awareness, and it lets you control the framing. The two you
name are the two they would have raised anyway.

---

# SLIDE 18 — Semester 2 plan

**ON THE SLIDE**
1. Reward ablation — does finding 1 survive different weights?
2. Classical baseline — tuned cascaded PI
3. **Constrained RL (PPO-Lagrangian)** — state the safety requirement as a
   constraint instead of hand-balancing weights
4. ArduPilot SITL validation — zero-shot transfer to production firmware
5. Hardware deployment — the lab's ArUco + UWB stack on ArduPilot

**SAY**
Five steps. The one I want to highlight is the third.

Nine reward iterations in this project were spent hand-balancing weights to
express a safety requirement, and it never fully worked. The right tool is
constrained RL: instead of tuning a penalty coefficient, you state the
requirement directly — maximise success *subject to* touchdown velocity at most
half a metre per second — and a Lagrange multiplier tunes itself. Finding three
strengthens that case: a hard constraint would have bound seeds two and three,
where a fixed reward weight did not.

**WHY**
Shows the work has a trajectory, and that your semester-1 difficulties motivate
your semester-2 method rather than being setbacks.

---

# SLIDE 19 — Summary

**ON THE SLIDE**
1. Success rate is saturated at 97–100% and cannot discriminate
2. Degrading sensing **raised** success 83% → 97% while creating a 35%
   safety-violation rate
3. An abstract sensing model overstated success by **51 points** versus a
   rendered pipeline
4. Success is stable to **±0.5%** across seeds while safety varies **±31%**

> Touchdown kinematics, not success rate.

**SAY**
Four points. The last one is the argument: the metric the field reports is
stable and uninformative; the metrics it omits are where all the variation
lives.

**WHY**
The panel remembers the last slide. Make it the claim, not "thank you."

---

# APPENDIX SLIDES (do not present; have ready for questions)

Keep these after the summary so you can jump to them.

**A1 — Reward function in full**, every term with its coefficient
**A2 — All three seeds for every condition**, full metrics table
**A3 — The frequency sweep** showing non-monotonic success (the memorisation
evidence)
**A4 — Camera verification** — detection error 0.020 m, dropout height derived
vs measured
**A5 — LSTM training runs** — the four attempts and the collapse diagnostics
(`approx_kl` 1.047, `std` 0.077)
**A6 — Folder structure and file list**
**A7 — Hyperparameters**

---

# PREPARATION CHECKLIST

**Two weeks out**
- [ ] Record the landing video. Do not leave this to the last week.
- [ ] Build slides 1–19
- [ ] Build appendix slides

**One week out**
- [ ] Full run-through, timed. Aim for 13 minutes, not 15.
- [ ] Cut anything that pushes past 15.
- [ ] Rehearse slides 4, 12, 13, 14 until you can deliver them without notes.
      Those four carry the argument.

**Three days out**
- [ ] Run-through in front of a labmate
- [ ] Read Part 13 of `project_reference_and_viva.md` aloud
- [ ] Prepare answers to: *why not PID?* · *did you build the drone?* ·
      *how does the drone see the platform?* · *why did success go up with
      worse sensors?*

**The day before**
- [ ] Test the video plays on the presentation machine
- [ ] Export to PDF as a backup
- [ ] Sleep

---

# TIMING

| Slides | Content | Minutes |
|---|---|---|
| 1–3 | Title, problem, related work | 2 |
| 4–6 | **The gap**, what I measure, RQs | 3 |
| 7–10 | Method | 3 |
| 11–14 | **Results and three findings** | 4 |
| 15–16 | Bugs, video | 2 |
| 17–19 | Limitations, plan, summary | 1 |
| | **Total** | **15** |

If you are running long, cut from the method section. Never cut slides 4, 12,
13, or 14 — those are the argument.
