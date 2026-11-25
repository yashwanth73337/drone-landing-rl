# V7 notes: moving pad with speed changes, waves and corners (after v6_drop)

Living notes for Stage 2 (V7). Newest results at the end of section 4 (the dated log).
**How to add notes:** add a new dated entry at the END of section 4 (`### YYYY-MM-DD ...`),
update the results table (section 5) and the open problems (section 8). Keep old entries:
they are the history. Simulation only. Numbers are from Yashwanth's laptop unless marked
"sandbox" (Claude's cloud copy of the code).

Repo: github.com/yashwanth73337/drone-landing-rl. V7 code lives in `v7/` (it imports the
unchanged V6 modules from `v6_drop/`; no V6 file was edited). Results live in `v7/results/`
(git-ignored, laptop only).

---

## 1. Where V7 started

**Stage 1 (V6, `v6_drop/`, done and reported 19 Nov 2026):** vision-only landing on a pad
moving in a STRAIGHT line at a constant 0 to 4 m/s. Pipeline: privileged teacher, demos with
DART troubles, behaviour cloning (BC), DAgger 4 rounds, PPO 300 rounds with an asymmetric
critic, drop shield 0.8 m/s, onboard touch detector, referee with 5 safety checks. Final
test (fresh seeds 20000 to 20499, 3 training seeds): 1497/1500 = 99.80 %.

**Supervisor meeting, 19 Nov 2026 (for the AIR paper):** faster pad, zigzag (wave) motion,
90 deg turns (corners), a PID baseline for comparison. Simulation only. Experiments until
10 Dec, paper from 11 Dec (AIR last date 31 Dec 2026).

---

## 2. Settings (decisions, with dates)

| Topic | Decision | Date |
|---|---|---|
| Method | Same pipeline as V6 (teacher, demos, BC, DAgger, PPO, shield, touch detector, referee) | 19 Nov |
| Pad motion | Random driver: keep speed, speed up, slow down, smooth wave, 90 deg corners (brake, turn, speed up). Speed changes and sideways acceleration at most 2 m/s^2 | 19 Nov |
| Named test paths | (1) straight, (2) wave, (3) corners (left and right); plus "random" (the driver) | 19 Nov |
| Pad top speed | 9 m/s (lab drone max 10 m/s; must be faster than the pad) | 19 Nov; confirmed 20 Nov |
| Drone start | Almost at rest (+-1 m/s), as Stage 1. No running start (he rejected it) | 19 Nov |
| Flight time limit | 60 s (600 decisions) | 20 Nov |
| Drone | S500 size (skids, standing height 0.22 m); LMF2 mass/motors kept | 19 Nov |
| Camera | 2-axis gimbal (steady in pitch and roll, yaw follows the drone). 45 deg first, **60 deg from 20 Nov 22:00**; **tilt by height from 25 Nov** (section 4) | 19 to 25 Nov |
| Drone speed limits | vx, vy +-10 m/s; vz +-1 m/s; yaw +-60 deg/s | 19 Nov |
| Speed-change limit | 3 m/s^2 (rate 0.3 m/s per decision) | 20 Nov |
| Referee check 5 | Against the pad point velocity under the drone (turning pad) | 20 Nov |
| Reward (step R) | V6 reward, except unsafe landing -8 (was -3) and hard landing -8 - 2 x (impact - 1), floor -10 | 24 Nov |
| PID baseline | PID-cam (camera + IMU, fair) and PID-true (true state, upper bound). NOT built yet | 19 Nov |

**Seeds (never reuse):** dev 30000+; final test 40000 to 40499; teacher study grid 32000+,
check 33000+, images 34000+, sandbox tuning 31000+, 35000+; demos 500000+; DAgger round N:
600000 + 10000 x N (610000, 620000, ...); PPO training flights 3,000,000 + 1,000,000 x run
seed + 100,000 x worker + k (the per-seed part from 25 Nov, S4s).

---

## 3. Code map (`v7/`)

| File | What |
|---|---|
| `config.py` | Fixed settings; `make_env(student_rules=...)`; `make_teacher("v7"|"view")`; one teacher per data folder (`teacher.txt`) |
| `envs/platform_v7.py` | Pad driver (random + named paths), up to 9 m/s |
| `envs/s500.py`, `assets/s500.urdf` | S500 shape (skids, 0.22 m standing height) |
| `envs/gimbal_camera.py` | Gimbal camera, marker reading; `set_pitch()` (25 Nov) |
| `envs/world_v7.py` | World: pad, drone, camera, referee; `GIMBAL_SCHEDULES` (25 Nov) |
| `envs/landing_env_v7.py` | Gym env (actions +-10 m/s, 600 steps, priv for the critic) |
| `envs/reward_v7.py` | V6 reward with the 0.22 m height and the step R end marks |
| `experts/teacher_v7.py` | Privileged teacher (S2a) |
| `experts/teacher_view_v7.py` | View-keeping teacher (S2', 21 Nov) |
| `experts/troubles_v7.py` | DART troubles for demos |
| `scripts/teacher_study_v7.py` | Teacher study (grid / check / images; `--teacher view`) |
| `scripts/record_demos_v7.py` | Demos with images (`--teacher`) |
| `scripts/train_bc_v7.py` | BC (V6 trainer, V7 numbers), `--blind-mask` |
| `scripts/train_aux_v7.py` | BC/DAgger training with the HELPER task (S4h, 22 Nov) |
| `scripts/dagger_collect_v7.py` | DAgger collection (`--beta`, `--teacher`) |
| `scripts/eval_student_v7.py` | Dev test: 4 paths, student rules (`--gimbal-schedule`) |
| `scripts/train_ppo_v7.py` | PPO (V6 PPO pieces; `--reward-source`, `--critic-rewarm`, `--gimbal-schedule`, `--seed`) |
| `scripts/diag_student_v7.py` | Flight diagnostic, student vs teacher every 0.5 s (`--teacher`, `--mode`, `--pad-speed`, `--tag`; output `diag_<ckpt stem><tag>.txt`) |
| `scripts/diag_view_v7.py`, `diag_inputs_v7.py` | Error by visibility; input check |
| `tests/test_s*.py` | One test file per step; run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q v7/tests/...` |

---

## 4. Dated log

### 2026-11-19: plan, S0, S1a to S1c (build)
- Plan `claude/V7_PLAN_2026-11-19.md` (project). S0: code check of every V6 place that
  assumed a straight line, constant speed, 0 to 4 m/s, +-6 m/s, 30 s, fixed camera, old legs.
- S1a pad driver (71ba57d), S1b S500 shape (0f462f7), S1c gimbal camera (ef03479).
- Findings: S500 can TIP on a forward/backward speed mismatch at touch (0.3 m/s rocks 7 deg,
  1.0 m/s flips); gimbal at 60 deg sees ~1.9 x height ahead; pad friction holds a landed drone
  up to 4.9 m/s^2 (pad max 2 m/s^2).

### 2026-11-20: S1d, teacher study, settings fixed, demos, BC, DAgger at 45 deg, switch to 60 deg
- S1d world + env (ac45f88): 44 decisions/s with images, 870 headless (laptop).
- S2a teacher (e2e7bbd): speed-change learning 0.2 s, lead 0.3 s, lowest-point height,
  safe30m drop rule, wait/abort.
- S2b teacher study (d4cf8d1, af71b90): 3 m/s^2 / 9 m/s 791/800 = 98.9 %; 5 m/s^2 worse.
  Fresh check 1977/2000 = 98.85 %; with images + detector + shield 100/100 at 45 and 60 deg.
  **Settings fixed** (section 2).
- S3a demos (2834b39): 1200 attempts, 1174 kept, 197,004 samples. S3b BC (0951c03): val
  MAE vx 0.64 m/s. S3c dev test (7fe3f4a). BC dev 0/400.
- S3 diagnostic (e04054c): student memory fine; it overshoots, then flies away blind.
- S4a DAgger collector (88121c8), S4b beta mixing (b3ff6b9), S4c blind mask (fda1a76:
  no loss after 3 s without the pad in view), input check (b0aaf0e: the student really uses
  the camera, not a copycat).
- DAgger at 45 deg: r1 1, r2 0, r2m (blind mask) 0, r3m 1 /400. Failure: small overshoot
  or swing at low height, the pad goes under the 45 deg camera, no recovery.
- **Camera to 60 deg** (f8fc550, 22:15). p60: BC 0, r1 1, r2 2 /400.

### 2026-11-21: root cause, view-keeping teacher, helper task proposed
- Heading diagnostic (55fe03f): **the teacher flies where the camera cannot see.** From
  rest it falls 6 to 16 m behind a fast pad (beyond the camera's reach) and acts on turns
  and speed changes out of view. Labels there are impossible for a camera-only student.
- **S2' view-keeping teacher** (6caeb00), his "yes view keeping": only while far, (1) climb
  to 0.625 x distance (cap 6 m, 1 m/s) so the pad stays below the top edge of the picture,
  (2) nose toward the pad beyond 6 m. Near the pad exactly the old teacher.
  Sandbox: markers read to ~8 to 9 m slant (so the 6 m cap). Laptop check 1978/2000 = 98.9 %;
  images 100/100; pad in view while far 75.7 % (old ~50 %).
- S3v (8140bcf): `--teacher view` in demos, DAgger, diagnostic.
- p60v (60 deg + view teacher): demos 1170 kept (206,514 samples), BC 0, r1 1, r2 5, r3 7.
  Collection got much better (mixed safe 58 -> 75 -> 81 %, gap to teacher vx 1.60 -> 0.61).

### 2026-11-22: helper task works, beta 0.2 works, disk full
- **S4h helper task** (33d86e0): during training only, an extra head predicts the pad's
  position and speed relative to the drone (the recorded "priv"); loss + 0.1 x MSE. The
  flying network and its inputs do not change.
  Control test (same data r1 to r3, same start, 8 epochs): control 2 vs helper 15 /400,
  McNemar 15 vs 2, **p = 0.0023**.
- r4aux (beta 0.5): 9 (vs 15, p = 0.31). Diagnostic: reaches the pad, swings +-1 m along
  the path, sits 0.3 to 0.7 m to the side, never drops.
- Disk full (111 MB left): dataset build crashed ("Bus error"). Old built datasets
  (45 deg, old p60) proposed for deletion (raw recordings kept). Check `df -h ~`.
- **beta 0.2** (student flies 80 %): r5aux 41, r6aux 59, r7aux 96 /400.

### 2026-11-23: DAgger plateau
- r8aux 125, r9aux 125 /400 (31.2 %). Drop recall in training falls each round (44 -> 28 %).
- Wave diagnostic (r8aux): follows the pad well for up to 50 s, but the drop gate never holds
  (sideways lag 0.5 to 1.4 m in turns): timeouts. Remaining gap = precision near the pad.

### 2026-11-24: step R, PPO
- **Step R** ("yes R"): he asked for a more negative hard landing. Final: unsafe -8, hard
  -8 - 2 x (impact - 1) floor -10; rest as V6. Order: success > timeout > unsafe > hard > crash.
- **S4p PPO** (df103a7): V6 PPO pieces; V7 env with student rules and air-drop bonus;
  12 workers, 256 steps, 300 rounds, ~182 decisions/s, ~2 h.
- **ppo_p60v_s0 (from r9aux), r0299: 263/400 = 65.8 %** (61.0 to 70.2): slow pads 90.5 %,
  medium 80.7 %, fast 46.4 %; timeouts 124 -> 8; lost 87 (73 on fast pads).
- Diagnostic r0299: on fast pads the pad gets ~15 m ahead from rest; at 60 deg the camera
  sees ~11 m ahead even at 6 m height; the blind student wobbles (yaw +-50 deg/s) instead of
  chasing straight. Corners: lands while the pad brakes (slide) or drifts ahead (pad under
  the camera), then turns the wrong way.
- S4q (1ec504c): `--reward-source true` (progress from the simulator truth; training only),
  `--critic-rewarm N`, `--resume X --name NEW`.

### 2026-11-25: A/B, gimbal tilt, one-phase recipe, 98 % target
- A/B, 150 more rounds from r0299: A camera reward 234 (worse, p = 0.008 vs r0299);
  B true-state progress 258 (vs A p = 0.03; vs r0299 p = 0.71). Lesson: stop at 300 rounds
  with the camera reward; true-state progress keeps longer training stable; fast pads not fixed.
- **S4g gimbal tilt by height** (0b0a12f), his "yes gimbal": 60 deg below 3 m, 45 deg above
  6 m, linear between (height = body above the pad top minus 0.22 m). At 15 m behind and 6 m
  high, the pad is in the picture only with the tilt. Starts are chosen at 60 deg, so every
  seed starts at exactly the same place (paired tests). 3 of 60 starts begin with the pad
  outside the first picture.
  - T1 r0299 with the tilt, no training: 276 (69.0 %), fast pads 55 % (p = 0.29 vs r0299).
  - **T2 ppo_p60v_s0_true_g r0449 (r0299 + 150 rounds, true-state progress, tilt): 290/400
    = 72.5 %** (67.9 to 76.6). vs B (same but no tilt): 80 vs 48, **p = 0.006**. vs r0299:
    p = 0.02. Slow 93.5 %, medium 79 %, fast 60 %; corners 50 %.
  - One-phase recipe ppo_p60v_g_s0 (450 rounds from r9aux, tilt + true-state): 274 (68.5 %);
    vs T2 68 vs 52, p = 0.17; medium 88 % but fast 46 % with 30 fast-pad timeouts.
- S4s seed fix (sent, not yet committed): `--seed N` also moves the training flights.
- **Target set by Yashwanth: at least 98 % with ONE seed before training seeds 1 and 2.**
  Next: a failure study of T2's 110 failed dev flights (section 8).

---

## 5. Dev results (eval_student_v7, 400 flights: 4 paths x 100, seeds 30000+)

| Student | Camera | Safe /400 | % | slow 0-3 / mid 3-6 / fast 6-9 m/s | lost / timeout |
|---|---|---|---|---|---|
| Teacher (true state, headless check) | n/a | 1977/2000 | 98.85 | | |
| BC 45 deg, DAgger r1 to r3m | 45 | 0 to 1 | ~0 | | |
| BC 60, r1, r2 (old teacher) | 60 | 0, 1, 2 | ~0 | | |
| p60v BC, r1, r2, r3 (view teacher) | 60 | 0, 1, 5, 7 | ~1 | | |
| r3aux (helper) / r3c control | 60 | 15 / 2 | 3.8 / 0.5 | 10/50, 5/149, 0/201 | 329 / 46 |
| r4aux (beta 0.5) | 60 | 9 | 2.2 | | 282 / 97 |
| r5aux, r6aux, r7aux (beta 0.2) | 60 | 41, 59, 96 | 10.2, 14.8, 24.0 | r7: 43/52, 34/142, 19/206 | r7: 145 / 147 |
| r8aux, r9aux | 60 | 125, 125 | 31.2 | r9: 39/50, 48/142, 38/208 | r9: 143 / 124 |
| PPO s0 r0299 (camera progress) | 60 | 263 | 65.8 | 57/63, 117/145, 89/192 | 87 / 8 |
| A: +150 camera | 60 | 234 | 58.5 | | 88 / 5 |
| B: +150 true-state | 60 | 258 | 64.5 | | 90 / 4 |
| T1: r0299 with tilt, no training | tilt | 276 | 69.0 | 58/64, 113/145, 105/191 | 69 / 17 |
| **T2: +150 true-state with tilt** | tilt | **290** | **72.5** | 58/62, 118/149, 114/189 | 56 / 7 |
| One-phase 450 rounds, tilt + true | tilt | 274 | 68.5 | 57/62, 127/144, 90/194 | 66 / 34 |

T2 by path: straight 89, random 73, wave 78, corners 50. T2 failures (110): lost 56
(corners 34), crash into the pad 17, slide off 14, tip over 12, timeout 7, crash ground 3,
not still 1. Landed by touch detector 213, own drop 107; mean impact 0.32 m/s, max 0.75.

---

## 6. What we learned (keep)

1. **A privileged teacher must not rely on what the camera cannot show.** The old teacher
   chased far behind and reacted to things out of view; the student could not copy that.
2. **The helper task (predict the pad's position and speed) is a big gain** for a
   camera-only student (p = 0.0023 in a controlled test).
3. **Let the student fly in DAgger** (beta 0.2 instead of 0.5) once it can follow the pad:
   its own mistakes then get labels (9 -> 41 -> ... -> 125).
4. **DAgger alone plateaued at ~31 %**; PPO took it to ~66 % (timeouts 124 -> 8).
5. **Longer PPO with the camera-based progress reward made it worse**; true-state progress
   (training only) keeps it stable but did not fix fast pads by itself.
6. **The camera's reach limits fast pads.** Tilting the gimbal up when high is a clear gain
   (p = 0.006).
7. About 60 flights switch between safe and unsafe from one student to another: use paired
   tests on the same seeds, and several training seeds for the paper.
8. Practical: files sent from the phone go to the phone; `nohup` hides output (read the
   summary files); run dependent commands one at a time; watch the disk.

---

## 7. How to run (laptop, repo root, venv active)

```
# dev test (400 flights, ~10 min); add --gimbal-schedule h3to6 for the tilt
python -m v7.scripts.eval_student_v7 --ckpt <ckpt.pt> --n 100 --workers 4 [--gimbal-schedule h3to6]
# PPO from a student (~2 h for 300 rounds)
nohup python -m v7.scripts.train_ppo_v7 --init <student.pt> --name <run> --workers 12 --rounds 300 > v7/results/p60v/<run>.log 2>&1 &
# continue a PPO checkpoint into a new folder (T2 recipe)
nohup python -m v7.scripts.train_ppo_v7 --resume <run>/r0299.pt --name <new> --workers 12 --rounds 450 --critic-rewarm 5 --reward-source true --gimbal-schedule h3to6 > v7/results/p60v/<new>.log 2>&1 &
# flight diagnostic (3 flights, student vs teacher)
python -m v7.scripts.diag_student_v7 --ckpt <ckpt.pt> --teacher view --skip-a --mode straight --pad-speed 8 --tag _fast
```
Paired check (McNemar) between two dev result files: load both `dev_*.jsonl`, key by
(mode, seed), count flights safe in only one, `scipy.stats.binomtest(i, i + j, 0.5)`.

---

## 8. Open problems and next steps (as of 25 Nov)

- **Goal (his): >= 98 % on dev with one seed** before seeds 1 and 2. Now 72.5 % (T2).
  The privileged teacher itself is 98.85 %.
- **Next: failure study of T2's 110 failed flights** (no training): for landing accidents,
  the sideways offset and speed relative to the pad at touchdown, vertical speed, pad braking
  or turning, who cut the motors; for lost flights, when/where the pad was last seen.
- Candidate fixes after the study: (1) extended safety rule near the pad (go around when the
  camera shows too much sideways speed), up to ~+40; (2) wider camera view so the pad stays
  visible under/behind the drone (corners), up to ~+30, may need new data; (3) fast pads
  pulling away (~20), partly physics (pad 9 m/s, drone 10 m/s, from rest).
- Recipe for seeds (when allowed): two-phase as T2 (300 rounds camera progress fixed 60,
  then 150 rounds true-state + tilt), with the S4s seed fix.
- Not started: S5 PID baseline (PID-cam, PID-true); final test on seeds 40000 to 40499;
  update of `claude/V7_PLAN_2026-11-19.md`.

## 9. Do not claim
- Not "works on a real drone" (simulation only). Not "better than Shin et al.".
- No speed or path result before a test on fresh seeds (dev numbers are development only).
- Paper numbers only from `eval_student_v7` dev/final tests, never from training exams.
