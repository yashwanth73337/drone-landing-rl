# V5 — Faithful reimplementation of Shin et al. (RA-L 2026): SPEC

**Status:** decisions made 30 Sep 2026 (§0). Block 1 built and verified (§12).
**Reference:** W. Shin et al., "Vision-Based Autonomous Drone Landing on Moving
Platforms With Uncertain Motion via DRL", IEEE RA-L 11(5), pp. 5542–5549, May 2026.
Page numbers below are journal pages (5542–5549).
**Perception reference:** T. Park et al., "PACMAN", Image Vis. Comput. 165 (2026) 105821.

Status tags:
- **[paper]**: stated in the paper, value quoted
- **[inferred]**: read off a figure or deduced; the source is given
- **[unspecified]**: the paper is silent; a value is proposed here and **you decide**
- **[deviation]**: we cannot match the paper; the reason is given
- **[ambiguous]**: the paper states it, but it can be read more than one way

V5 does not import from or copy V1–V4 code (`src/envs`, `src/policies`,
`train_shin_v4.py`). Lessons are carried over as written notes only.

---

## 0. Decisions (made 30 Sep 2026)

| ID | Question | Decision |
|---|---|---|
| D1 | Env base: gym-pybullet-drones `BaseAviary`, or our own gymnasium env on raw PyBullet? | **DECIDED: raw PyBullet, own env.** BaseAviary calls `_computeReward()` before `_computeTerminated()` (notes §10), is CF2X-centric, and bundles DSLPIDControl. We still use PyBullet (the decided simulator) but none of its drone scaffolding. |
| D2 | Quad model | **DECIDED: AerialGym LMF2 (1.24 kg).** The Table II gains are exactly AerialGym's `lmf2_controller_config.py`, so LMF2 is the airframe the paper's controller was tuned for [inferred]. The lab quad matters only for Semester-2 sim-to-real. |
| D3 | Vertical-speed-penalty sign (§6) | **DECIDED 30 Sep (second decision, after the Block 9 landscape): the LITERAL printed equation −[v_z + 0.5]⁺ is the default.** The prose reading (penalise descent > 0.5 m/s) is kept as `vz_penalty='prose'`, a single-change ablation. Impact speed is reported with every result. |
| D4 | Δz sign convention (§6) | **ACCEPTED: Δz = z_pad_top − z_drone** (negative while above), so "undershoot" = below the pad top. |
| D5 | Curriculum rule (§7) | **ACCEPTED:** Levels 10, 20, …, 80; c = level/80; promote at ≥80% success over each 512-episode window; no demotion. |
| D6 | Clip on platform speed and yaw-rate random walk (§3) | **ACCEPTED:** v ∈ [0, 8c] m/s, \|ω\| ≤ 30°/s |
| D7 | Table II external torque "±4e3 N·m" | **ACCEPTED:** Treat as a typo for **±4e-3 N·m**. |
| D8 | PPO implementation | **ACCEPTED:** **Own compact recurrent PPO in PyTorch** (cleanRL-style). The auxiliary loss inside the objective, the asymmetric critic, and reward injection before GAE are all awkward in SB3/sb3-contrib. |
| D9 | CNN input resolution | **DECIDED 30 Sep: render 512×320 [paper], INTER_AREA downsample to 256×160.** Lab desktop (T400, EGL, 16 envs): 326 steps/s (direct 256×160 would be 1,029). |
| D10 | Auxiliary-loss weight λ_est | **ACCEPTED:** 1.0 |
| D11 | Include the privileged-actor variant P (§11) | **ACCEPTED:** **Yes.** It is the control ceiling for the state-vs-control question. |
| D12 | Initial drone yaw | **ACCEPTED:** Face the pad ± U(−15°, 15°); rejection-sample until the pad centre is in the image. |
| D13 | Ground textures (Table II: 50 IDs) | **ACCEPTED:** 50 procedurally generated textures from a seeded script. |
| D15 | Collision geometry / success region (§9, Block 4) | **DECIDED 30 Sep: keep the paper's definition (pad-top contact) with the AerialGym 0.5 m box; also report strict success (`com_over_pad`) in every evaluation.** |
| D16 | Vertical-speed action limit (§5) | **DECIDED 30 Sep, after P_smoke_s1: v_z limit ±1 m/s** (was ±3, [unspecified]). The paper's Fig. 9 shows descents of ≲ 1 m/s in every scenario; with ±3 the privileged policy dived at ~4.1 m/s impact. Env parameter `vz_max` (default 1.0); `reward_profile.py` keeps 3.0 because it studies the reward, not the limit. |
| D14 | Training seeds per configuration | **ACCEPTED:** 3 for configurations that are reported; 1 for exploratory runs. |

---

## 1. Simulator and timing

| Item | Paper | V5 | Status |
|---|---|---|---|
| Simulator | AerialGym (Isaac Gym) [25], p.5546 §IV-A | PyBullet 3.2.7 | [deviation]: decided; lab desktop, no Isaac Gym |
| Env wrapper | — | own gymnasium env (D1) | [unspecified] |
| Policy / control step Δt | 0.1 s (10 Hz), p.5543 §II-B and p.5546 | 0.1 s | [paper] |
| Episode horizon | 300 steps = 30 s, p.5546 | 300 | [paper] |
| Physics rate | not stated | **100 Hz** (10 substeps per action) | [inferred] AerialGym `base_sim_config.dt = 0.01` |
| Low-level controller rate | sim: not stated. Real: geometric controller 50 Hz, PX4 250 Hz (Fig. 10a, p.5548) | 100 Hz (every physics step, as in AerialGym) | [inferred] |
| Parallel envs | "multiple environments … in parallel", p.5546 | SubprocVecEnv, 16 envs (desktop has 20 threads) | [unspecified] |
| Rendering | GPU (Isaac) | PyBullet EGL plugin, one render per policy step (10 Hz), not per physics step | [deviation]: throughput |
| Training budget | Fig. 5 x-axis up to ~125k **episodes**; proposed method ~95% by ~20k episodes; "converges within approximately 3.5 hours" on an RTX 4090, p.5546 | report ours in episodes **and** env steps | [paper]/[inferred]. Env steps not stated; upper bound ≈ 125k × 300 = 37.5M |

**Throughput gate (before any training run):** `scripts/bench_env.py` measures
steps/s for {1, 8, 16} envs × {EGL on, off} × {512×320, 256×160}, with the
render count per step asserted to be 1. The achieved budget is reported as a
fraction of the paper's episodes.

## 2. Vehicle and controller

Source for everything marked [AG]: AerialGym (github.com/ntnu-arl/aerial_gym_simulator),
files `config/robot_config/lmf2_config.py`, `config/controller_config/lmf2_controller_config.py`,
`resources/robots/lmf2/model.urdf`, `control/controllers/{base_lee_controller,velocity_control}.py`,
`control/motor_model.py`, `config/sim_config/base_sim_config.py`. Values are copied into
`envs/lmf2_params.py`; no AerialGym code is imported.

### 2.1 Quadrotor (D2: LMF2)

| Item | V5 value | Status |
|---|---|---|
| Airframe | AerialGym LMF2: Table II gains = `lmf2_controller_config` exactly | [inferred] [AG] |
| Mass | 1.2 kg base + 4 × 0.01 kg props = **1.24 kg** | [AG] |
| Inertia | base diag(0.013, 0.014, 0.013) + props at (±0.1, ±0.1, 0) → diag(0.0134, 0.0144, 0.0138) kg·m² | [AG] |
| Collision | 0.5 m cube (drives touchdown geometry: bottom face 0.25 m below the CoM) | [AG] |
| Allocation | rows [Fz, τx, τy, τz]: τ arms ±0.13 m, thrust-to-torque ratio 0.07 | [AG] |
| Motor thrust | per motor [0.1, 10] N → thrust/weight 3.29 | [AG] |
| Motor model | first-order in √thrust (rpm) space, discrete factor dt/(dt+τ); τ_up ~ U(0.05, 0.08) s per motor, τ_down = 0.005 s; thrust constant cancels | [AG] |
| Damping | linear and angular rigid-body damping 0.01 | [AG] |
| Aero drag | none (LMF2 damping coefficients are all 0) | [AG] |
| PyBullet load flags | `URDF_MERGE_FIXED_LINKS \| URDF_USE_INERTIA_FROM_FILE`. **Without the second flag PyBullet computes inertia from the collision box (0.052 kg·m², ~4× too large). Found by the Block 1 test.** | ours |

### 2.2 Controller

| Item | Paper | V5 | Status |
|---|---|---|---|
| Type | Lee geometric controller [26], body-frame velocity + yaw-rate commands (p.5546) | numpy port of AerialGym `LeeVelocityController` | [paper] + [AG] |
| K_v (x, y, z) | U(2.7, 3.3), U(2.7, 3.3), U(1.3, 1.7) (Table II) | each component sampled independently | [paper] |
| K_θ (roll, pitch, yaw) | U(1.6, 1.85), U(1.6, 1.85), U(0.25, 0.4) (Table II) | same | [paper] |
| K_ω (roll, pitch, yaw) | not stated | U(0.4, 0.5), U(0.4, 0.5), U(0.075, 0.09) | [AG] |
| Gain scaling | not stated | velocity loop mass-scaled: F = m(K_v e_v − g). Attitude loop outputs **raw torque**: τ = −K_θ e_R − K_ω e_ω + ω×Jω (not inertia-scaled, so the gains are tied to the LMF2 inertia). *Corrects the earlier draft, which guessed inertia scaling.* | [AG] |
| Command frame | "heading frame" (p.5543) | vehicle frame = yaw-only rotation, rotated to world inside the controller | [paper] + [AG] |
| Yaw | yaw-rate command | desired attitude uses the **current** yaw (no yaw-angle hold); desired body rate from the euler yaw rate; \|ω_z\| clamped to π/3 rad/s | [AG] |
| Tilt limit | not stated | none (`max_inclination_angle_rad` is defined in the config but unused by the velocity controller) | [AG] |
| Gain sampling time | "Env. init" (Table II) | once per environment instance. AerialGym code re-samples on each reset when `randomize_params=True`; **the paper's table wins** | [paper] |

**Block 1 verified (30 Sep 2026), `tests/test_block1_controller.py`, 30/30 pass:**

| Check | Result |
|---|---|
| Mass / inertia loaded | 1.24 kg; inertia within 0.3% (merge re-diagonalises) |
| Hover 5 s, gains at nominal / min / max | drift 0.000 m, tilt 0.000° |
| Lateral 2 m/s step | t50 0.39–0.43 s (ideal first-order 0.21–0.26 s), overshoot 8–14%, steady-state error ≤ 1.1%, cross-axis ≤ 0.01 m/s |
| Vertical ±1 m/s step | t50 0.38–0.59 s (ideal 0.41–0.53 s), overshoot 0%, steady-state error ≤ 1.5% |
| Heading frame at yaw 90° | world velocity matches the heading-frame command within 0.04 m/s |
| Yaw rate 0.5 rad/s and 2.0 rad/s | 0.4986–0.4989 rad/s; 2.0 is clamped to 1.043 against π/3 = 1.047 |
| Chase: 10 m/s command | 9.5 m/s reached in 1.01 s, altitude loss 0.018 m, no motor saturation |

Dynamics notes for later blocks:
- **The attitude loop is slow.** Its dominant pole is about 0.23 s, so lateral velocity overshoots 8–14%.
- **Motors spin down in 5 ms but up in 50–80 ms.** So descent reacts slightly faster than ideal and climb slower.
- **Yaw drifts a few degrees during manoeuvres,** because there is no yaw-angle hold.

The test criteria were revised once, with the reasons recorded in the test docstring. The first version compared the 10–90% rise with first-order theory, which is the wrong model for a lagged, S-shaped response.

## 3. Landing platform

| Item | Paper | V5 | Status |
|---|---|---|---|
| Motion | planar; forward speed and yaw rate random walk: v_{t+1} = v_t + δv, ω_{t+1} = ω_t + δω, applied each 0.1 s step (p.5543 §II-B) | unicycle kinematics, integrated at physics rate, perturbed at 10 Hz | [paper] |
| v_0 | U(0, 8) m/s (Table I, p.5544) | U(0, 8c) | [paper]; c scaling [unspecified] (§7) |
| ω_0 | 0°/s (Table I) | 0 | [paper] |
| δv | U(−0.5, 0.5) m/s per step (Table I) | U(−0.5c, 0.5c) | [paper]; c scaling [unspecified] |
| δω | U(−3, 3)°/s per step (Table I) | U(−3c, 3c)°/s | [paper]; c scaling [unspecified] |
| Speed / yaw-rate bounds | not stated; the walk is unbounded (after 300 steps, speed SD ≈ 5 m/s) | v ∈ [0, 8c], \|ω\| ≤ 30°/s (D6) | [unspecified] |
| Initial platform yaw misalignment ψ_0 | U(−60°, 60°) (Table I) | angle between platform heading and drone heading | [paper]; reference frame [ambiguous] |
| Vehicle body | ground vehicle (golf cart in the real test, Fig. 10a) | box 1.5 × 1.5 × 1.0 m, kinematic (mass 0, pose reset every physics step) | [unspecified] |
| Pad | 1.5 × 1.5 m (p.5546) | top face of the box | [paper] |
| Pad top height | Fig. 9 z-plots show the platform at ≈1 m | 1.0 m | [inferred] Fig. 9, p.5548 |
| Vertical heave | only in the "Boat" evaluation (Fig. 9, Table V) | off in training | [paper] |

**Block 2 verified (30 Sep 2026), `tests/test_block2_platform.py`, 10/10 pass.** KS tests at α = 0.01 over 200 episodes × 300 steps, seed base 20000:

| c | v_0 ~ U(0, 8c) | δv ~ U(±0.5c) | δω ~ U(±3c °/s) | v clip rate | ω clip rate | mean v | \|ω\| p95 |
|---|---|---|---|---|---|---|---|
| 0.125 | pass | pass | pass | 4.4% | 0.0% | 0.51 m/s | 5.7 °/s |
| 0.5 | pass | pass | pass | 4.4% | 0.4% | 2.03 m/s | 22.2 °/s |
| 1.0 | pass | pass | pass | 4.4% | 2.7% | 4.05 m/s | 28.6 °/s |

Kinematics: straight line and circle exact to 1e-9 m, PyBullet body pose in sync, finite-difference velocity matches, c = 0 is stationary, runs are deterministic per seed. Order within a policy step: 10 physics substeps at the current (v, ω), then `perturb()`.

**Watch item.** At c = 1 the yaw-rate walk sits near the ±30°/s clip by late episode (its unclipped SD after 300 steps ≈ 30°/s). That means tight turns: a radius of about 8 m at 4 m/s. The D6 clip is therefore an active constraint and will be reported as such.

## 4. Pad marking (visual target)

| Item | Paper | V5 | Status |
|---|---|---|---|
| Main method | PACMAN hexagonal keypoint pad [17] with frozen keypoint encoder (Fig. 3) | **later, as a single change** | [paper] |
| Ablation "w/o keypoint encoder" | pad replaced by ArUco [19]; policy uses a "standard CNN trained end-to-end" (p.5546 §IV-B) | **V5 base configuration** | [paper] |
| ArUco dictionary / id / size | not stated | DICT_4X4_50, id 0. **Black square 1.2 m, centred, with a 0.15 m white margin** (a quiet zone is needed for detection by baseline H). Texture 768 px RGB. | [unspecified] |
| ArUco usage | image → CNN. **No detection, no PnP.** | same. OpenCV ArUco appears only in the EKF+RL baseline (§11). | [paper] |
| PACMAN weights | README says a pretrained detector is provided, but the link is a blank "(See)". The repo has C++/TensorRT inference only: no weights, no training code. It expects `pacman_fpn202506_1240_1624_int8.engine`. MIT licence. | requested from the authors (PyTorch checkpoint). Fallback: train our own from sim-projected keypoint labels using the Park et al. recipe (their Table 1, Eqs. 1–5). | [deviation] pending |

### Block 3 rendering implementation (30 Sep 2026)

- **Platform visual:** ONE closed box mesh (`assets/pad_box.obj`) with the marker texture on the top face only; the other faces sample a white texel. Collision is the plain 1.5 × 1.5 × 1.0 box. No extra bodies.
- **Ground:** our own 200 × 200 m textured quad (`envs/ground.py`) with a plane collision shape. **No URDF-embedded textures anywhere.**
- **Grayscale:** BT.601 luma (0.299, 0.587, 0.114), the same as `cv2.cvtColor`.
- **Camera offset:** 0.15 m ahead of the CoM [unspecified], clear of the 0.25 m visual body.
- **ArUco detector:** subpixel corner refinement (`CORNER_REFINE_SUBPIX`).

**EGL pitfalls found and fixed (each has a regression test):**

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Blank frames | EGL only draws bodies created after the plugin loads | `make_client(egl=True)` loads the plugin first |
| 2 | Pad shows the ground's checker | `plane.urdf`'s internal texture shifts `loadTexture` ids | no URDF textures; every texture comes from `loadTexture` |
| 3 | Marker sheared diagonally | 750 px grayscale texture (GL row alignment) | RGB, width a multiple of 4 (768 px) |
| 4 | Stripes; no detection beyond 3 m altitude | marker quad above the box top z-fights (EGL depth ≈ 15 mm at 7 m) | single box mesh, no coplanar surfaces |

**Block 3 verified, `tests/test_block3_camera.py`, 18/18 pass** (TinyRenderer and EGL/llvmpipe here):

- VFOV 64.01° and f_x = f_y = 256 px.
- OpenCV detects id 0 and its corners match the analytic projection in corner order, at 5 poses (2–6 m, yaw 30°, roll 10°, platform moved and rotated 120°). Max error: EGL 0.69–0.88 px, TinyRenderer 1.13–1.56 px, against a 2 px limit.
- Detected on the optical axis at every altitude from 2 to 8 m under both renderers.
- One render per call; ground texture swap does not affect the marker; marker texture is RGB with width % 4 = 0; platform is one body with box collision.

## 5. Sensors, observation, action

| Item | Paper | V5 | Status |
|---|---|---|---|
| Camera | grayscale pinhole, 512×320, 90° **horizontal** FOV (p.5546) → VFOV ≈ 64.0° | same | [paper] |
| Camera mount | "60° downward pitch to the forward axis" (p.5543) | optical axis 60° below body +x (30° from nadir), 0.15 m ahead of the CoM | [paper]; offset [unspecified] |
| CNN input | not stated | 256×160 (D9), pixel values / 255 | [unspecified] |
| Velocity sensor | body-frame velocity + 0.05 m/s Gaussian noise (p.5546) | N(0, 0.05²) per axis, every step | [paper]; per-axis σ [inferred] |
| Attitude sensor | quaternion + 0.5° small-angle noise (p.5546) | small rotation, axis-angle components ~ N(0, (0.5°)²), composed with the true attitude | [paper]; model [inferred] |
| Actor obs | o_t = (I_t, u_t), u_t = [v_b ∈ R³, q ∈ R⁴] (p.5543) | same; u_t noisy | [paper] |
| Critic obs | o_priv = [u_t, s_rel_t] (p.5545 §III-D1) | 13-D; u_t **clean** | [paper]; clean-vs-noisy [unspecified] |
| Estimation target | s_rel = [Δx_b, Δv_b] ∈ R⁶: platform position and velocity relative to the drone, **in the body frame** (p.5544) | Δx = p_pad − p_drone; Δv = v_pad − v_drone; both rotated into the body frame | [paper]; difference direction [inferred] |
| Action | a_t = [v_x, v_y, v_z, ω_z] velocity commands in the drone's **heading frame** (yaw-only rotated) (p.5543) | same | [paper] |
| Action limits | not stated | v_xy ±10 m/s, **v_z ±1 m/s (D16; was ±3)**, ω_z ±π/3 rad/s (= the controller clamp [AG]); policy output clipped to [−1, 1] then scaled | [unspecified]; v_z [inferred] from Fig. 9 |

Note on exploration: V4's white-noise failure came from position-offset actions
that did not accumulate (notes §5). Velocity commands held for 0.1 s do integrate
into position, so Gaussian PPO noise should give net motion. The oracle/random
reachability test (§12) checks this before any RL.

## 6. Reward

Final step reward (p.5545 §III-D4):

```
r_t = +10                          successful landing
      -10                          crash or excessive drift
      r_shaping_t + r_active_t     otherwise
```

Shaping terms, r_shaping = Σ w_i r_i (Table III, p.5545), where [x]_a^b = clip(x, a, b):

| Term | Equation (as printed) | w_i | Status |
|---|---|---|---|
| Lateral progress | [d_xy,t−1 − d_xy,t]_{−1}^{1} | 1.0 | [paper]; d_xy = horizontal drone–pad-centre distance |
| Vertical progress | [\|Δz_t−1\| − \|Δz_t\|]_{−1}^{1} / max(d_xy,t, 1) | 1.0 | [paper] |
| Vertical speed penalty | −[v_z + 0.5]_0^∞ as printed. **V5 (D3): as printed** (z up); prose reading −[−v_z − 0.5]_0^∞ = ablation | 0.5 | [paper] equation; the prose contradicts it |
| Undershoot penalty | −𝟙[Δz_t > 0] Δz_t | 1.0 | **[ambiguous] D4** |
| Yaw-rate penalty | −\|ω_z\| | 2.0 | [paper]; ω_z = commanded, in rad/s [unspecified] |

**D3, vertical speed (decided: the literal equation; first decision was the prose reading, revised after the Block 9 landscape).**
- The printed −[v_z + 0.5]⁺ penalises every v_z > −0.5 m/s, including hovering (0.25 per step).
- The prose ("penalise fast descent") would mean −[−v_z − 0.5]⁺ instead.
- The landscape below shows that the prose reading leaves hovering and following worth about as much as landing, while the literal one makes descent always pay. That is consistent with the paper's fast PPO convergence.
- Cost of the literal reading: it rewards diving, since success does not check impact speed. So **impact speed (pre-contact, Block 5) is reported with every result**, and `vz_penalty='prose'` is the single-change ablation.

**D4, Δz sign.** Table I gives the altitude offset Δz_0 ∈ U(2, 8) m as positive,
but "undershoot" only makes sense if Δz > 0 means the drone is below the pad
top. Proposal: Δz = z_pad_top − z_drone. |Δz| terms are unaffected.

Terminal and edge cases:

| Item | Paper | V5 | Status |
|---|---|---|---|
| Success | "contact with the pad upper surface" (p.5546) | contact between the drone and the pad-top face, drone CoM above the pad top. No speed or attitude condition (faithful); touchdown relative speed and tilt are **logged**. | [paper] |
| Crash | not defined | contact with the ground or a platform side face; tilt > 80° | [unspecified] |
| Excessive drift / workspace exit | not defined | d_xy > 15 m, or drone more than 12 m above the pad top | [unspecified] |
| Timeout at 300 steps | no terminal reward stated | shaping only; treated as truncation (bootstrap from V(s)) | [unspecified] |
| Terminal step | ±10 **replaces** the shaping on that step | same | [paper] |

**Test:** `tests/test_block9_reward.py` (below).

### Block 9 verified (30 Sep 2026), `envs/reward.py`, `tests/test_block9_reward.py`, 19/19 pass

- **Implementation:** `ShinReward(vz_penalty='literal')` is the env default (D3). The first build defaulted to 'prose'. All numbers below were re-run after the switch, and a test asserts the default. Shaping terms are hand-checked on 5 constructed cases: both clip bounds, the max(d, 1) branch, undershoot below the pad top, and yaw. Both D3 variants are tested.
- **Env integration:** reward equals an independent recomputation from the sim state every step.
- **Terminals:** +10 success; −10 for crash_ground, crash_platform, drift and tilt (all tested); the timeout step gets shaping only.
- **v_z** is the drone's world vertical velocity at the end of the step; **ω_z** is the commanded yaw rate in rad/s [unspecified].

**Reward landscape (`scripts/reward_profile.py`):** start 6 m above a stationary pad (c = 0, DR off), hold position laterally, fixed descent rate. Discounted return uses γ = 0.99.

| rate | prose (ablation): return / disc. | per-step shaping, h 0–4 m | **literal (D3)**: return / disc. | per-step, h 0–4 m |
|---|---|---|---|---|
| hover | 0.00 / **0.00** | 0 | −75.0 / −23.8 | −0.25 |
| 0.3 m/s | 15.74 / 3.76 | +0.030 | −5.23 / −5.78 | −0.072 |
| 0.5 m/s | 15.75 / **6.15** | +0.050 | 14.09 / 4.69 | +0.047 |
| 1.0 m/s | 2.48 / 0.03 | −0.145 | 15.37 / 9.02 | +0.099 |
| 1.5 m/s | −2.03 / −2.95 | −0.34 | 15.43 / 10.68 | +0.15 |
| 2.5 m/s | −6.07 / −6.14 | −0.72 | 15.60 / **12.23** | +0.24 |

(Starting 3 m off-axis gives the same ordering; hover earns +3.07 disc. from lateral progress under prose.)

**From Table I spawns (full DR, γ = 0.99, seeds 9000 + i; `reward_profile --hover-sweep / --policy-sweep`):**

Hover (zero command), 200 episodes per c. The platform drives away at c > 0: drift −10 in 48–96% of episodes.

| c | prose: mean disc. return | **literal: mean disc. return** |
|---|---|---|
| 0 | 0.05 | −22.4 |
| 0.125 | −4.54 | −25.8 |
| 0.25 | −8.09 | −26.2 |
| 0.5 | −12.05 | −25.7 |
| 1.0 | −15.43 | −24.9 |

Oracle policies (100 episodes each). `follow` = the oracle laterally, but it never commands descent.

| reward | policy | c = 0.125 | c = 1.0 |
|---|---|---|---|
| prose | follow | 2.59 (1% land) | 4.75 (78% land*) |
| prose | land, ≤ 0.5 m/s | **7.65** | 4.98 |
| prose | land, ≤ 1.5 m/s | 2.44 | 0.78 |
| literal | follow | −17.94 | −8.04 |
| literal | land, ≤ 0.5 m/s | 6.98 | 4.45 |
| literal | land, ≤ 1.5 m/s | **10.24** | **6.64** |

\* See the finding below: "follow" sinks and lands by accident at c = 1.

**Reading.**
- Under **prose**, following without descending is worth almost as much as landing at c = 1 (4.75 vs 4.98), and fast landings are worth less than following. That is a weak landing incentive with the same structure as the V3/V4 stalls.
- Under **literal**, not descending is strongly negative at every c, and faster landings are worth more. This is why D3 was switched.

**Finding: jittery commands make the drone sink (dynamics, faithful to AerialGym, not a bug).**
- With v_z command = 0 and zero-mean coloured lateral jitter (σ = 1), the drone sinks at **−0.66 to −0.72 m/s** (−1.1 m/s at σ = 4). Steady turns do not sink (−0.003 m/s at 30° tilt).
- **Cause (traced):** LMF2 motors spin *up* in 50–80 ms but *down* in 5 ms, so under changing commands delivered thrust averages below commanded (12.71 vs 14.06 N). The attitude lag's thrust projection costs a further ~0.6 N, and motors clip on only 0.3% of substeps.
- The vertical loop (K_v,z 1.3–1.7, no integral) cannot recover the deficit.
- **Consequences:**
  - (1) A tracking policy fed a random-walk platform velocity descends "for free"; at c = 1 it lands 78% of the time without commanding descent.
  - (2) Early PPO exploration noise pushes the drone down.
  - (3) Real PX4 attitude/thrust loops differ, so this is a sim property to remember for sim-to-real.
- Locked by `test_block1_controller.py::test_jitter_sinks_documented_property`.

## 7. Curriculum

| Item | Paper | V5 | Status |
|---|---|---|---|
| Scalar | c ∈ [0, 1] scales platform motion; c = 1 is the full task (p.5545 §III-D2) | same | [paper] |
| Levels | Fig. 3: "Level 10 → … → Level 80, level updated every 512 episodes" | levels 10, 20, …, 80 (8 levels), c = level/80 | [inferred] (D5) |
| What c scales | "platform motion … faster motion and stronger perturbations" | v_0 max, δv, δω (and the D6 clips). Spawn geometry is **not** scaled. | [unspecified] |
| Update rule | every 512 episodes; criterion not stated | after each 512 completed episodes (pooled over envs): promote if success ≥ 80%; no demotion | [unspecified] |

Why step-by-10: Fig. 5 marks "max level reached" at ≈8k–12k episodes. With
single-step levels, 70 promotions × 512 = 35,840 episodes minimum, which
contradicts the figure. Steps of 10 need 7 × 512 = 3,584 minimum, which is
consistent with it.

The 512-episode window also fixes the noisy 20-episode promotion problem (notes §7).

### Block 10 verified (30 Sep 2026), `envs/curriculum.py`, `tests/test_block10_curriculum.py`, 9/9 pass

- **Implementation:** `Curriculum.record(success, c_episode)` works on non-overlapping windows of 512 **counted** episodes, pooled over envs. It promotes at ≥ 80% (inclusive: 410/512 promotes, 409/512 does not), with no demotion, capped at level 80.
- **Stale episodes:** results are tagged with the c each episode *started* at (`info['c_episode']`), so episodes in flight during a promotion are ignored for the new level (tested).
- **Env hook:** `env.set_c(c)` takes effect at the next reset, never mid-episode (tested, also through `AsyncVectorEnv.call`). `state_dict()` round-trips for resuming.
- **Synthetic streams:** p = 0.95 promotes exactly every window and reaches 80 after 7 × 512 = **3,584 episodes** (the minimum possible); p = 0.70 never moves.
- **Promotion noise (exact binomial, per window):**

| true success p | 0.70 | 0.75 | 0.78 | 0.80 | 0.82 | 0.85 |
|---|---|---|---|---|---|---|
| P(promote) | 0.0000 | 0.0039 | 0.139 | 0.509 | 0.882 | 0.999 |

  The V-series rule (one 20-episode evaluation, 90% threshold) promoted a true-80% policy ~1 in 5 (notes §7). Here a true-75% policy promotes on 0.4% of windows.
- **Note:** training never runs at c = 0 (level 10 → c = 0.125). The only zero-motion condition is evaluation.


## 8. Domain randomisation (Table II, p.5545)

| Parameter | Range | When | Status |
|---|---|---|---|
| Control gains | §2.2 | env init | [paper] |
| External force F_x,y,z | U(−0.75, 0.75) N | each step | [paper]; "step" = policy step (held for 0.1 s) [unspecified] |
| External torque M_x,y,z | U(−4e-3, 4e-3) N·m | each step | [deviation]: printed as ±4e3 (D7) |
| Initial velocity v_0 | U(−1, 1) m/s per axis | episode start | [paper] |
| Initial angular rate ω_0 | U(−10, 10) °/s per axis | episode start | [paper] |
| Ground texture ID | 1…50 | episode start | [paper]; the textures themselves are [deviation] (D13) |
| Ground texture scale s | U(0.4, 1.2) | episode start | [paper] |
| Ground brightness I | U(0.5, 1.0) | episode start | [paper] |
| RGB scaling c_R,G,B | U(0.5, 1.0) | episode start | [paper]; applied before the grayscale conversion |
| Light direction φ | U(45°, 135°) | episode start | [paper]; PyBullet `lightDirection` elevation |

### Block 6 verified (30 Sep 2026), `envs/dr.py`, `envs/ground.py`, `tests/test_block6_dr.py`, 17/17 pass

**Implementation choices:**

| Item | V5 | Status |
|---|---|---|
| External force | world frame, U(±0.75) N per axis, re-sampled each **policy** step and held for its 10 substeps | [paper] range; frame and "step" [unspecified] |
| External torque | body frame, U(±4e-3) N·m per axis, same timing | [D7] |
| Initial v_0 / ω_0 | U(±1) m/s per axis (world); U(±10) °/s per axis (body) | [paper] |
| Ground texture | 50 procedural RGB 256 px textures (`assets/ground_tex/`, seed 1300 + i; noise, tiles, stripes, blobs), committed | [D13] |
| Texture scale s | **17 levels, 0.40–1.20 in steps of 0.05**, nearest to the U(0.4, 1.2) draw; each level is its own OBJ | [deviation]: PyBullet leaks ~40 KB per re-created visual |
| Ground brightness I | multiplies the ground texture colour (rgba) | [paper] |
| RGB scaling c | per channel, applied to the **whole image** before grayscale (folded into the luma weights) | [interpretation] |
| Light direction φ | U(45°, 135°) elevation in the world x–z plane, passed on every render (EGL light state is sticky) | [interpretation] |
| Velocity sensor | R^T v + N(0, 0.05²) per body axis | [paper] |
| Attitude sensor | q ⊗ exp(δθ), δθ ~ N(0, (0.5°)²) per body axis; [x, y, z, w] with q_w ≥ 0 | [paper] + sign [unspecified] |
| RNG | independent seeded streams (spawn/platform, gains, physical DR, visual DR, sensor noise): toggling a DR group never changes a pinned seed's spawns (tested) | ours |
| Default | `LandingSim` defaults to **all DR on**; bare-dynamics tests pass `DRConfig.off()` | ours |

**Verified:**

| Check | Result |
|---|---|
| Velocity noise (20k samples) | σ = 0.0500–0.0503 m/s per axis, \|corr\| ≤ 0.003 |
| Attitude noise | σ = 0.497–0.504° per axis |
| Constant 0.75 N force, hover command | v_ss = 0.2008 m/s; theory F/(m K_v) = 0.2016 |
| Constant 4e-3 N·m yaw torque | ω_z = 0.0484 rad/s; theory M/K_ω,z = 0.0485 (the controller has no yaw-angle hold) |
| Disturbances | fresh each policy step, held over its 10 substeps; KS passes |
| Initial state | the state after reset equals the sample; KS passes |
| Texture scale, **both renderers** | transition count within 3.7% of 10/(0.25 s) at all 17 levels |
| Brightness 0.5 / RGB 0.5 | ground mean ratio 0.498–0.499 / 0.500 |
| Light 45° vs 90° | mean \|Δ\| 18.9 (tiny) / 10.6 (EGL) grey levels |
| Oracle, full DR, 200 ep, seeds 9000 + i | c = 0: 100%; **c = 1: 99%** (2 timeouts, 0 crashes). DR off: c = 1 98.5% |
| ArUco at spawn, full DR, 150 ep | 88.7% detected, 0 false ids. With visual DR off on the **same spawns**: 89.3%. Visual DR changes 1/150. Misses are framing: in 12/150 spawns a marker corner is out of view (allowed, since only the pad centre must be); fully in view → 96–97% |

**Rendering bugs found (both now have regression tests):**

1. **The ground's collision plane was being drawn.** A body with no visual is drawn from its collision shape, and a PyBullet plane gets a default checker texture. It z-fought our textured quad **since Block 3**. Block 3's marker tests were unaffected (the pad is above it); every ground-appearance measurement would have been. Fix: a hidden dummy visual 50 m underground.
2. **EGL shares one mesh between visual shapes loaded from the same file**, so meshScale variants rendered blank (TinyRenderer was fine). Fix: one OBJ per scale. Visual-DR tests now run under both renderers.
3. `cv2.transform` with a one-row matrix already returns (h, w); an extra `[..., 0]` produced a column vector. Caught by the brightness and detection tests.

## 9. Initial conditions (Table I, p.5544)

| Item | Value | Status |
|---|---|---|
| Altitude offset Δz_0 | U(2, 8) m above the pad top | [paper] |
| Lateral offset (Δx_0, Δy_0) | U(−3, 3) m each | [paper] |
| Platform in FOV at t = 0 | required (p.5543 §II-A) | [paper]; enforced by D12 rejection sampling |
| Platform yaw misalignment | U(−60°, 60°) | [paper] |
| Estimator / LSTM state | zeros at reset | [unspecified] |

### Block 4 verified (30 Sep 2026), `envs/landing_sim.py`, `tests/test_block4_termination.py`, 12/12 pass

- **Termination:** checked after every physics substep (10 per policy step); the step ends at the first terminal substep.
- **Outcomes:**
  - `success`: any drone contact with the pad-top face (contact normal z ≥ 0.7).
  - `crash_platform`: side-face contact.
  - `crash_ground`.
  - `tilt`: > 80°.
  - `drift`: horizontal distance > 15 m, or > 12 m above the pad.
  - `timeout`: 300 steps; a truncation, not a termination.
- **Logged at every terminal step:** relative position (world and pad frame), relative velocity, tilt, and `com_over_pad`.
- **Hand-built checks, all pass:** centre drop, side crash, lateral and vertical drift, tilt, timeout, step-after-done, and a velocity-matched descent onto a **6 m/s** platform (success, pad-frame error −0.085 / −0.051 m).
- **Spawn (2000 resets, seed 20000):** 99.8% accepted on the first draw. KS passes for dx, dy, dz, ψ_0 and the D12 yaw offset. Pad-centre pixel row v spans 11–305 of 320 (median 190).
- **D15 (decided: keep the paper's definition + report strict), success region.** The LMF2 collision cube is 0.5 m, so pad-top contact is reachable with the CoM up to 0.75 + 0.25 = **1.0 m** from the pad centre, 0.25 m past the pad edge. Measured: CoM offset 0.97 m → `success` with `com_over_pad = False`; 1.03 m → `crash_ground`. The CoM sits 0.24 m above the pad at touchdown.

### Block 5 verified (30 Sep 2026), scripted oracle on the TRUE state

`policies/oracle.py` (privileged: velocity feedforward + P on the lagged relative position, descent gated by lateral error; it is **not** a baseline). `scripts/eval_oracle.py`, per-episode seeds 9000 + i, gains re-sampled per episode, 200 episodes each:

| c | success | strict (`com_over_pad`) | failures | steps to land (mean) | touchdown error p95 | impact v_z (mean) |
|---|---|---|---|---|---|---|
| 0.0 | 100% | 100% | none | 52 | 0.056 m | −0.47 m/s |
| 0.5 | 100% | 100% | none | 49 | 0.22 m | −0.61 m/s |
| 1.0 | **99%** | 99% | 2 timeouts | 63 | 0.40 m | −0.62 m/s |

**The task as built is solvable within the action limits at full difficulty.** The c = 1 timeouts come from the oracle's own gate logic: it holds altitude once more than 0.6 m off centre over a turning platform and never re-aligns. No crashes, tilts or drift.

Random-policy reachability (uniform actions within the limits, 100 episodes):

| hold | c | success | crash_ground | drift | tilt | median xy travel | median min height |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 6 | 90 | 0 | 2 | 2.7 m | −0.50 m (reaches ground) |
| 1 | 1 | 0 | 69 | 29 | 2 | 2.6 m | −0.41 m |
| 10 | 0 | 0 | 23 | 55 | 22 | 12.7 m | 2.1 m |
| 10 | 1 | 0 | 10 | 74 | 16 | 8.2 m | 3.7 m |

- **The V4 exploration failure does not recur.** Random velocity commands move the drone metres and reach pad height; V4's position offsets moved it 0.1 m.
- **Watch item, tilt.** Held full-scale random commands flip the drone in 16–22% of episodes. The velocity controller has no tilt limit (AerialGym faithful: `max_inclination_angle_rad` is defined but unused), and a 10 m/s step demands ~78° of tilt. Early PPO may crash this way; it costs −10 like any crash.
- **Bug fixed.** Touchdown relative velocity was read after the contact solver had zeroed it (−0.05 m/s at c = 0). It is now read pre-contact: −0.47 m/s, and −0.969 m/s for a −1 m/s descent (tested).

Tests: `tests/test_block5_oracle.py`, 4/4. The oracle subset is 50 episodes at c = 0 and c = 1.

## 10. Network (Figs. 3, 4, p.5544)

| Block | Paper | V5 (ArUco base) | Status |
|---|---|---|---|
| Visual encoder | frozen keypoint encoder + trainable CNN → l_t ∈ R^512 | `policies/encoder.py`: input 256×160 uint8 → x/255 − 0.5; 5 conv layers, stride 2, kernels (5, 3, 3, 3, 3), channels (32, 64, 64, 128, 128), ELU; 5×8×128 = 5120 → Linear → ELU → l_t ∈ R^512. **2,899,648 params**; no BatchNorm | [paper] dim; architecture [unspecified] |
| Memory layer | LSTM; input [l_t, u_t]; h_t ∈ R^512 | 1-layer LSTM, hidden 512 | [paper]; layers [unspecified] |
| State-estimation layer | MLP on [l_t, h_t, u_t] → y_t ∈ R^256 (N = 256) | MLP 1031 → 512 → 256, ELU, linear output | [paper] dims; hidden [unspecified] |
| Estimate | s̃_rel = y_t[0:6] | same | [paper] |
| Decision layer | MLP on [y_t, u_t] → a_t | MLP 263 → 256 → 128 → 4 (Gaussian mean); state-independent log σ, init −0.5 | [paper] inputs; sizes [unspecified] |
| Critic | MLP on [u_t, s_rel] (13-D), non-recurrent, discarded at deployment | MLP 13 → 256 → 256 → 1 | [paper]; sizes [unspecified] |

**Privilege boundary:** the actor forward pass receives only (I_t, u_t). A unit

### Block 11 verified (30 Sep 2026), `policies/shin_policy.py`, `tests/test_block11_network.py`, 10/10 pass

- **Parameters:** encoder 2,899,648 · LSTM 2,115,584 · estimation 659,712 · decision 100,996 · critic 69,633 · **total 5,845,577**.
- **Dimensions:** LSTM in 519 / hidden 512; estimation 1031 → 512 → 256 (y_t); s̃ = y[0:6] (checked by hand recomputation); decision 263 → 256 → 128 → 4; critic 13 → 256 → 256 → 1. Critic inputs are divided by a fixed scale [5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5] [unspecified].
- **Policy:** Gaussian with state-independent log σ (init −0.5); actions are clipped only in the env.
- **Variant P actor:** MLP [s_rel, u] → 256 → 128 → 4, with no CNN, LSTM or estimate. Same critic.
- **Privilege boundary (numerical):**
  - Vision actor: outputs (μ, s̃) are bit-identical when critic, target and s_rel change, and change when the image changes.
  - P actor: invariant to image, critic and target; changes with s_rel.
  - Critic: reads only its own input.
- **Recurrence:** step-by-step `act()` (rollouts) equals whole-sequence `evaluate()` (PPO update) in log-probs, values and estimates, across an episode boundary mid-sequence. The reset (h ← h·(1 − start)) isolates episodes exactly.
- **Gradient routing:**
  - L_est → encoder, LSTM and estimation only (not decision, critic or log σ).
  - Policy loss → the whole actor, not the critic.
  - Value loss → the critic only.
  - This is what the paper's joint objective requires. Nothing else is being quietly trained by the wrong loss.
- **Also tested:** no batch dependence; a real env step through the policy.

test asserts that the actor output is invariant to s_rel.

### Block 7 verified (30 Sep 2026), `tests/test_block7_encoder.py`, 5/5 pass

- **Downsampling:** INTER_AREA 512×320 → 256×160 is the exact 2×2 mean (±0.5). Done env-side; rollouts store uint8 256×160 (40 KB per frame).
- **Encoder:** output (B, 512), finite; 3-D and 4-D input identical; no batch dependence; gradients reach every parameter.
- **Overfit check:** 48 rendered frames, 120 Adam steps → train R² 0.999 / 0.999 / 0.999 for the body-frame pad position. Images and labels are consistent, and the network can learn.

**Probe (`scripts/probe_encoder.py`, supervised, single frame, pad centre in view, seeds from 20000; a diagnostic, not the method):**

| run | frames (train) | DR | grad steps | train R² x/y/z | test R² x/y/z | test median error |
|---|---|---|---|---|---|---|
| CPU, tiny | 600 (480) | off | 480 | 0.999 / 0.995 / 0.999 | 0.78 / 0.56 / 0.83 | 0.59 m |
| CPU, tiny | 1500 (1200) | on | 380 | — | 0.52 / 0.25 / −0.05 | 1.41 m |
| **Desktop GPU, EGL** | 3000 (2400) | on | 1520 | 0.992 / 0.981 / 0.983 | **0.73 / 0.64 / 0.81** | **0.57 m** (p90 2.15 m) |

Held-out accuracy is **data- and step-limited** (2.9M params), not a pipeline fault. The DR run underfits at 380 steps. A GPU run with more frames is logged when available.

**Finding for Blocks 12–13.** While the privileged oracle flies (it ignores the camera), the pad centre is **out of the image in ~50% of visited frames** (0.455 and 0.496 in two runs). Over or past the pad, the 60°-pitched camera looks ahead of it. A single frame cannot locate an unseen pad: this is the regime the LSTM, L_est and r_active exist for.

### Block 8 verified (30 Sep 2026), `envs/shin_env.py`, `tests/test_block8_env.py`, 10/10 pass

- **Env:** gymnasium `ShinLandingEnv(mode='vision' | 'privileged')`, passing gymnasium's `check_env`. Every observation key is always present; **the actor's view is chosen by `ACTOR_KEYS`**: vision → (image, u); privileged (variant P) → (u, s_rel). `target` and `critic` are never actor inputs in vision mode.
- **Observation:** image uint8 160×256; u = noisy [v_b, q] (7); critic = [u_clean, s_rel] (13); target = s_rel = [R^T(p_pad − p), R^T(v_pad − v)] (6), from the true state.
- **Action:** clip(a, −1, 1) × [10, 10, 3, π/3] in the heading frame. Measured: 10.1 m/s forward after 1.5 s, world yaw rate −1.044 rad/s for −π/3.
  - A full-rate turn at 10 m/s needs ~47° of bank, so the body-z rate is then cos(tilt) of the world yaw rate. That's physics, not a scaling bug.
- **Privilege boundary (tested):** two envs that differ only in platform velocity give **identical** actor views (image, u) but different targets and critic inputs. The actor can infer platform motion only from images over time.
- **Also tested:** noise is present in u and absent in the critic (σ ≈ 0.05 m/s); `terminated` vs `truncated` (timeout = truncated at 300 steps); one render per reset and per step in vision mode and none for P; seeded determinism; `AsyncVectorEnv` with spawn workers.
- **Reward:** Block 9, `reward_fn='paper'` = `ShinReward()` (literal D3) by default.

## 11. Learning

### 11.1 PPO (all [unspecified]; D8)

| Hyperparameter | Proposal |
|---|---|
| Implementation | own recurrent PPO (PyTorch), single optimiser, actor and critic separate |
| n_envs × rollout | 16 × 256 = 4096 steps per update |
| BPTT | sequences of 32 steps with stored initial (h, c); hidden state reset at episode start |
| Epochs / minibatches | 5 / 4 |
| γ / λ | 0.99 / 0.95 |
| Clip ε | 0.2 (no value clipping) |
| LR | 3e-4 Adam, linear decay |
| Entropy coefficient | 0.0 |
| Max grad norm | 1.0 |
| Value loss coefficient | 0.5 |
| Reward / obs normalisation | none (±10 terminals are designed to dominate) |

### Block 12 built (30 Sep 2026): `policies/ppo.py`, `scripts/train_ppo.py`, `tests/test_block12_ppo.py`, 10/10 pass (CPU; no training run yet)

- **Vector env:** gymnasium `AsyncVectorEnv` (spawn) with **SAME_STEP autoreset**. The final observation and final info are read from `info['final_obs'] / info['final_info']`.
- **Rollout:** the LSTM (h, c) is stored at every 32-step chunk start; `starts` masks reset it.
- **Timeouts are bootstrapped:** next_value = V(final_obs.critic). Terminal steps: 0. The GAE chain is cut at every done.
  - **Tested:** a hand-computed GAE with a truncation and a termination. With the horizon forced to 5, every truncated step's next_value is V(final obs), not the reset obs' value.
- **Update:** loss = L_clip + 0.5·mean((V − R)²) − 0·H + λ_est·L_est, where L_est = mean (s̃ − s_rel)² [paper Eq. 1].
  - Per-minibatch advantage normalisation [unspecified; cleanRL default], Adam eps 1e-5, linear LR decay, grad-norm clip 1.0.
  - Variant P: λ_est = 0 (no estimator).
- **Gradient accumulation:** a minibatch of 128 chunks × 32 steps = 4096/4 = 1024 frames can be split into micro-batches with one optimizer step. **The parameters after one step are identical to the unsplit update (tested).** This is needed on the 4 GB T400: the CNN activations for 1024 frames are about 4.5 GB. Default for vision on GPU: 8 chunks (256 frames) per micro-batch. `scripts/gpu_update_check.py` measures the peak memory.
- **Recurrence correctness (tested):** re-evaluating a real rollout in chunks, from the stored states, reproduces the rollout log-probs, values and estimates, so the first update starts at ratio = 1 (first-minibatch KL < 1e-6).
- **Estimator path (tested):** L_est alone drops 8.40 → 0.40 in 40 steps on a fixed batch.
- **Also tested:** checkpoint round-trip; blind-age bins cover every step; the CLI end-to-end for both modes (2 updates, async envs, files and columns).
- **Logging** (`runs/<name>/`):
  - `updates.csv`: success, outcome counts, strict success, impact v_z, level and window, losses, KL, clip fraction, explained variance, log σ, estimate error by blind age (vision), sps.
  - `episodes.csv`: one row per episode.
  - `curriculum.csv`: one row per window.
  - `config.json`.
  - Checkpoints `latest.pt`, `u<N>.pt`, `level_<L>.pt` (git-ignored).

### 11.2 Auxiliary estimation loss (Eq. 1, p.5545)

L_est_t = (1/6) Σ_i (s_rel_t,i − s̃_rel_t,i)², raw units (m, m/s), minimised
jointly with the PPO loss on the same minibatches [paper]. Total loss =
L_PPO + λ_est · L_est, with λ_est = 1.0 [unspecified] (D10).

### 11.3 Active-perception reward (p.5545 §III-C)

r_active_t = −α · [β (L_est_t+1 − τ)]_0^1, with α = 0.1, β = 1.0, τ = 0.01 [paper].

- **Computation:** after the rollout, using the estimate recorded at t+1 during collection (detached), added to r_t before GAE. On terminal or truncated steps there is no t+1, so r_active = 0 [unspecified].
- **Watch item (not pre-fixed):** V4 fell below τ by 50k steps and the term went inert for 92% of training (notes §9 #6). Log the fraction of steps with L_est_t+1 > τ on every update.
- **Scale note:** the paper's Table IV RMSEs (0.47 m, 0.59 m/s) imply L_est ≈ 0.28 ≫ τ, so the term should be active in this task's scale (2–8 m altitude). V4's estimator ran at a much smaller error scale.

## 12. Build and verification order (tests in `v5_shin/tests/`)

| # | Component | Pass criterion |
|---|---|---|
| 1 | Physics + quad + controller | **DONE 30 Sep: 30/30** (§2.2) |
| 2 | Platform motion | **DONE 30 Sep: 10/10** (§3) |
| 3 | Camera + ArUco pad + EGL; **throughput benchmark** | **DONE 30 Sep: 18/18** (§4). Benchmark on the lab desktop: *pending* |
| 4 | Termination | **DONE 30 Sep: 12/12** (§9). D15 decided |
| 5 | **Scripted oracle on true state** | **DONE 30 Sep: 99% at c = 1** (200 ep), random reachability OK (§9) |
| 6 | Sensor noise + domain randomisation | **DONE 30 Sep: 17/17** (§8). Oracle under full DR 99% at c = 1 |
| 7 | CNN perception (ArUco base) | **DONE 30 Sep: 5/5** (§10); probe data-limited as expected |
| 8 | Obs/action interfaces | **DONE 30 Sep: 10/10** (§10), privilege boundary tested |
| 9 | Reward | **DONE 30 Sep: 19/19** (§6); D3 → literal after the landscape |
| 10 | Curriculum | **DONE 30 Sep: 9/9** (§7) |
| 11 | Network | **DONE 30 Sep: 10/10** (§10), 5.85M params |
| 12 | PPO + L_est | **BUILT 30 Sep: 10/10 CPU tests**; GPU fit check and smoke run need approval |
| 13 | r_active | hand-computed on a recorded rollout |
| 14 | Evaluation script | **BUILT 30 Sep: `scripts/evaluate.py`, 3/3 tests**: per-episode seeds 9000 + i (gains re-drawn), deterministic mean actions, worker-count independent (tested), Wilson 95% CI, strict success, impact speed, vision est RMSE + blind-age bins |
| 15 | PACMAN keypoint pad | single change, after the above |

## 13. Evaluation protocol

| Item | Paper | V5 | Status |
|---|---|---|---|
| Training-scenario evaluation | 10,000 episodes, c = 1, random-walk platform (p.5546 §V-A) | 10,000 episodes, seed base **9000** | [paper] |
| Metrics | success rate, position RMSE (m), velocity RMSE (m/s) (Table IV) | same, **plus** failure breakdown (success / ground crash / platform-side crash / drift / tilt / timeout), touchdown speed, blind fraction, longest blind run | [paper] + ours |
| RMSE frames | not stated | reported over all frames **and** per blind-age bin (visible, 1–15, 16–60, 61–120, >120 steps). This is the notes §9 #2 lesson: a visible-only error cannot see where crashes originate. | [unspecified] |
| Diverse dynamics (Table V, Fig. 9) | 1,000 episodes each: 8 m/s straight (98.8%), sinusoidal linear acceleration (79.1%), circle (79.6%), zigzag (69.7%), U-turn (55.0%), boat heave (99.5%) | parameters read off Fig. 9; lower priority | [paper] rates; parameters [inferred] |
| Promotion evaluation | — | seed base **10000** | ours |
| Perception diagnostics | — | seed base **20000** | ours |

Every number is reported with its checkpoint file, timestep, and task (c, spawn ranges).

### Configurations and paper targets (Table IV, p.5546)

| ID | Configuration | Paper success | Paper pos / vel RMSE |
|---|---|---|---|
| **A** | ArUco + CNN + LSTM + L_est + r_active (**V5 base**) = paper "w/o keypoint encoder" | 91% | 0.953 m / 1.063 m/s |
| A-noAP | A without r_active | — (the paper's 91% was measured on the keypoint base) | — |
| A-noEst | A without L_est and r_active, LSTM kept = paper "w/o state estimation" | 73% (keypoint base) | — |
| H | EKF+RL: ArUco detector + solvePnP (**written fresh**) → constant-velocity EKF → [pos, vel, diag cov] + u_t → MLP policy | 59% | 1.331 / 1.501 |
| **P** | **Privileged actor:** actor gets [u_t, s_rel] (true); same PPO, reward (no r_active), curriculum | — (ours) | — |
| K | Full proposed: keypoint pad + frozen PACMAN encoder | 97% | 0.474 / 0.589 |

Only A (and later K) is directly comparable to a paper number. The other
ablations are run on the ArUco base, so they are compared by gap and direction,
then repeated on K.

### State vs control reading (supervisor's question)

- **P fails:** control/training problem. Perfect information does not land, in this task.
- **P lands, A does not:** state problem.
- **A vs A-noEst:** what the learned estimator adds.
- **A vs H:** learned vs model-based estimation.

## 14. Repository

```
v5_shin/
  SPEC.md
  envs/  policies/  scripts/  tests/  assets/   (URDFs, ArUco texture, generated ground textures)
  runs/<run_name>/   config.json, CSVs (tracked); *.zip *.pt *.pth *.pkl (ignored)
```

`.gitignore` additions (anchored at the repo root):

```
/v5_shin/runs/**/*.zip
/v5_shin/runs/**/*.pt
/v5_shin/runs/**/*.pth
/v5_shin/runs/**/*.pkl
```


## 15. Runs log

Every run: name, commit of the code, task, and outcome. Training-time numbers come from the **stochastic** policy; pinned evaluations (Block 14) are reported separately.

### P_smoke_s1 (30 Sep 2026): privileged actor (variant P), seed 1, 2M steps, v_z limit ±3 (pre-D16)

- **Config:** code at commit `3b3b19a`; 16 envs × 256 steps; SPEC §11.1 PPO; literal D3; full DR; curriculum from level 10. 489 updates, 93k episodes, ~1,550 steps/s (≈ 22 min on the lab desktop). CSVs committed in `43bcb64`.
- **Learning:** 4/151 successes at update 1. Level 80 (c = 1) reached at ~update 120 (~0.5M steps). **Zero timeouts in the whole run** (no hover stall), zero drift after update 60.

| updates | level | episodes | success | strict (CoM over pad) | impact v_z median (p10 / p90) | speed median | steps to land (median) |
|---|---|---|---|---|---|---|---|
| 0–60 | 10 | 12,412 | 0.438 | 0.301 | −4.10 (−5.68 / −2.93) | 4.51 m/s | 19 |
| 60–120 | 80 | 12,317 | 0.792 | 0.646 | −4.18 | 4.68 | 20 |
| 120–250 | 80 | 23,272 | 0.827 | 0.698 | −3.79 | 4.42 | 22 |
| 250–400 | 80 | 28,264 | 0.891 | 0.785 | −4.09 | 4.64 | 21 |
| 400–490 | 80 | 16,917 | **0.909** | **0.807** | −4.14 (−5.44 / −2.69) | 4.61 | 21 |

- **Reading, control:** with perfect information, PPO with this reward and curriculum learns the full task (≈ 91% at c = 1, training-time). V4's failure to descend does not recur.
- **Reading, landing style:** it **dives**. Median impact is −4.1 m/s throughout, landing ~2 s after spawn. This is the diving risk of the literal D3; it is inconsistent with the paper's Fig. 9 (≲ 1 m/s descents), which led to D16.
- **Reading, D15:** about 1 in 9 "successes" is an edge contact with the CoM off the pad (0.909 vs 0.807).
- **What this does NOT establish:** a pinned evaluation (seed 9000); more than one seed; behaviour under the D16 limit.
