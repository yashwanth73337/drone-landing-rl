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
| D17 | D3 re-test under the ±1 m/s limit (§6, §15) | **PROPOSED 1 Oct, pending owner confirmation: keep the literal D3.** One prose run (P_smoke_prose_vz1_s1) was tested against the literal run under D16. Under prose the curriculum never left L10 in 2M steps; the run lost the success-rate comparison (30.5% vs 81.5% pinned at c = 1), gained 337/1000 tilt crashes, and still touched down at −1.66 m/s median. The touchdown speed is set by the thrust-deficit sink, not by the reward sign (§15). |
| D18 | Renderer for vision training on the 4 GB T400 | **DECIDED 1 Oct (measured): TinyRenderer (CPU), 16 envs × 256 steps, PPO update on the GPU (micro 8).** Each EGL env holds ~250 MB of GPU memory (1/4/8/16 envs: +252/+1005/+2009/+3658 MiB of 3715), so 16 EGL envs + training does not fit; the first A smoke run crashed out of memory. 8 EGL × 512 steps would fit only with ~0.3 GB to spare and is no faster overall: the update dominates (33.5 s at micro 8, peak 1.82 GiB; 37.8 s at micro 4, 1.06 GiB). Rollout: tiny 16 envs 194 steps/s vs EGL 8 envs 266. Tiny images were validated in Block 3 (corner error ≤ 1.6 px) and the Block 6 visual-DR tests run under both renderers. Evaluation of vision runs must use the same renderer (`--renderer tiny`). |
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
| Motor integration | **Euler** with the discrete factor dt/(dt+τ). AerialGym's LMF2 config sets no `integration_scheme`, so AerialGym falls back to **RK4**. When the paper was submitted (10 Nov 2025), that RK4 had a sign bug, fixed 3 May 2026 in commit `8a15191`: it closed e^x − 1 of the gap instead of 1 − e^−x. Fraction of the gap closed per 10 ms step at τ_down: ours 0.667, AerialGym now 0.486, AerialGym paper-era 0.947 (τ_up 0.11–0.17 in all three). Measured jitter sink: −0.725 / −0.695 / −0.780 m/s, so the effect is small (1 Oct audit). | [deviation], small |
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

### Block 13 built (1 Oct 2026): `policies/active_perception.py`, `tests/test_block13_active.py`, 8/8 pass

- **Formula:** r_active_t = −0.1 · clip(1.0 · (L_est_{t+1} − 0.01), 0, 1), where L_est_{t+1} is the Eq. 1 loss of the estimate the policy produced during the rollout for the observation at t+1 (current parameters, detached).
  - **Last rollout step:** the estimate for the next observation is computed without advancing the LSTM. A test checks that it equals the estimate the next rollout actually produces.
  - **Done steps** (terminated or truncated): r_active = 0, because the next observation belongs to a new episode [unspecified].
- **Integration:**
  - Added to the env reward before GAE; the test recomputes GAE from rewards + r_active.
  - Recomputed independently from a recorded vision rollout across forced episode boundaries (horizon 5), with every (t, b) matching to 1e-6.
- **Switch:** `train_ppo.py --r-active {auto, off}`.
  - auto = on for vision with λ_est > 0 (proposed method A); off = A-noAP.
  - It is never used by variant P or A-noEst (λ_est = 0).
  - Recorded as `r_active_used` in config.json.
- **Logging:**
  - `updates.csv` gains `r_active_frac`: the fraction of live steps with L_est_{t+1} > τ (the V4 "inert term" watch item); `r_active_mean` was already logged.
  - The console line shows `ra <frac>/<mean>`.
- **Watch item, saturation early in training.** An untrained estimator has L_est ≈ 6.5 (test rollout). So r_active saturates at −0.1 on every live step, acting as a time penalty: the discounted sum over a 300-step horizon is ≈ −9.5, comparable to the −10 crash penalty. It stops being a constant once L_est < 1.01. The paper's Table IV RMSEs imply L_est ≈ 0.3 at convergence (r_active ≈ −0.03 per step). Log and watch; do not pre-fix.

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
| 13 | r_active | **BUILT 1 Oct: `policies/active_perception.py`, 8/8 tests** (§11.3); no training run yet |
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

### P_smoke_vz1_s1 (30 Sep 2026): variant P, seed 1, 2M steps, v_z limit ±1 (D16), literal D3

- **Config:** same as P_smoke_s1 except `--vz-max 1.0` (the new default). Code at `59b2389`. The training-time table can be regenerated from the committed `updates.csv`.

### P_smoke_prose_vz1_s1 (1 Oct 2026): variant P, seed 1, 2M steps, v_z limit ±1, PROSE D3 (D17 test)

- **Config:** same as P_smoke_vz1_s1 except `--vz-penalty prose`. 489 updates at ~1,450 steps/s.
- **Learning (training-time, stochastic policy):**
  - **The curriculum never left L10** (c = 0.125); the window success rate stayed around 0.4–0.55.
  - **Zero timeouts**, so the hover/follow stall predicted in Block 9 did not appear.
  - **Tilt (> 80°) crashes grew instead:** 1 per update at u1, ~150–250 of ~330–400 episodes per update after u130. Ground/platform crashes fell to single digits.
  - The episode count per update doubled (~150 → ~330), so episodes got shorter; tilt comes early.
- **Cause of the tilt: NOT established.** Unverified hypotheses:
  - (a) r2 = Δ|dz| / max(d, 1) does not telescope. Descending near the pad pays more than climbing far from it costs, and prose leaves climbing unpenalised, so an up/down pumping cycle can pay.
  - (b) With no pressure to descend, episodes are spent chasing at full lateral command; the controller has no tilt limit (Block 5 watch item).

### Pinned evaluation: all three smoke runs, `latest.pt`, c = 1.0, 1,000 episodes, seed base 9000, deterministic mean actions

| run | timestep | success [95% CI] | strict | ground / platform / tilt / drift / timeout | impact v_z median (p10 / p90) | steps (median) |
|---|---|---|---|---|---|---|
| P_smoke_s1 (±3, literal) | ~2.0M | 91.4% [89.5, 93.0] | 78.0% | 76 / 10 / 0 / 0 / 0 | −3.36 (−4.26 / −2.10) | 22 |
| P_smoke_vz1_s1 (±1, literal) | ~2.0M | 81.5% [79.0, 83.8] | 68.6% | 158 / 24 / 3 / 0 / 0 | −2.30 (−3.07 / −1.54) | 28 |
| P_smoke_prose_vz1_s1 (±1, prose) | 2,002,944 | 30.5% [27.7, 33.4] | 26.6% | 190 / 26 / 337 / 142 / 0 | −1.66 (−2.23 / −1.09) | 27 |

**Safe success.** Fraction of ALL 1,000 episodes that succeed with |impact v_z| ≤ threshold; the strict success fraction is in brackets.

| run | ≤ 0.5 m/s | ≤ 1.0 | ≤ 1.5 | ≤ 2.0 |
|---|---|---|---|---|
| P_smoke_s1 | 0.000 (0.000) | 0.004 (0.000) | 0.029 (0.013) | 0.080 (0.033) |
| P_smoke_vz1_s1 | 0.000 (0.000) | 0.012 (0.002) | 0.076 (0.027) | 0.249 (0.159) |
| P_smoke_prose_vz1_s1 | 0.003 (0.003) | 0.025 (0.020) | 0.097 (0.084) | 0.248 (0.218) |

**Readings:**

- **Neither D3 reading reproduces the paper's slow descents.** With the command limited to −1 m/s, both runs touch down faster than any commanded speed: median −2.30 (literal) and −1.66 (prose). The excess of 0.7–1.3 m/s matches the Block 1 thrust-deficit sink (−0.66 to −1.1 m/s under jittery commands, motor τ_up 50–80 ms vs τ_down 5 ms). **So touchdown speed is set by the actuation model plus policy jitter, not by the reward sign.**
- **The paper's success metric hides unsafe touchdowns.** At c = 1, safe success (≤ 1 m/s) is at most 2.5% in every run, including those at 81–91% on the paper metric.
- **The prose reading costs far more than it gains.** It gives 3× lower success and a stalled curriculum, and gains only +0.013 safe success (≤ 1 m/s) and ~0.06 strict safe success (≤ 2 m/s). This supports keeping the literal D3 (D17).
- **Caveats:**
  - one seed per configuration;
  - the prose policy was trained only at c = 0.125 and is evaluated at c = 1, which is out of its training distribution. Its drift count (142) partly reflects this.
- **Not yet done:** a direct test of the sink as the cause, e.g. evaluating the same checkpoints with symmetric motor time constants. That is an evaluation-only change, needs no training, and is not yet approved.

### Motor time-constant diagnostic (1 Oct 2026): evaluation only, no retraining

- **Question:** do fast touchdowns come from the motor asymmetry (spin-up 50–80 ms, spin-down 5 ms) making the drone sink faster than commanded?
- **Method:**
  - `evaluate.py --motor {asym, sym_slow, sym_fast}`:
    - `sym_slow`: τ_down := τ_up per motor;
    - `sym_fast`: τ_up := 5 ms.
  - Only the time constants change. RNG streams are unchanged (tested), so the episodes are the same.
  - Checkpoint `P_smoke_vz1_s1/latest.pt` (2,002,944 steps; literal D3; ±1 m/s), c = 1.0, 1,000 episodes, seed base 9000, deterministic.
  - Code: `envs/quad.py` (`MotorModel(mode=...)`), `envs/landing_sim.py`, `envs/shin_env.py`, `scripts/evaluate.py`; tests in `tests/test_diag_motor.py` (4); 164 tests pass.
  - The `asym` arm reproduced the earlier evaluation exactly (same seed, outcome, steps and impact v_z in all 1,000 episodes).

| motors | success [95% CI] | strict | ground / platform / tilt / drift / timeout | impact v_z median (p10 / p90) | commanded v_z, last 5 steps (median) |
|---|---|---|---|---|---|
| asym (as trained) | 81.5% [79.0, 83.8] | 68.6% | 158 / 24 / 3 / 0 / 0 | −2.30 (−3.07 / −1.54) | −0.31 |
| sym_slow | 68.7% [65.8, 71.5] | 49.3% | 217 / 72 / 24 / 0 / 0 | −1.52 (−2.20 / −0.92) | −0.33 |
| sym_fast | 55.8% [52.7, 58.9] | 33.6% | 138 / 105 / 63 / 0 / 136 | −0.78 (−1.30 / −0.37) | −0.33 |

Safe success. Fraction of all 1,000 episodes; strict version in brackets.

| motors | ≤ 0.5 m/s | ≤ 1.0 | ≤ 1.5 | ≤ 2.0 |
|---|---|---|---|---|
| asym | 0.000 (0.000) | 0.012 (0.002) | 0.076 (0.027) | 0.249 (0.159) |
| sym_slow | 0.019 (0.004) | 0.102 (0.037) | 0.331 (0.190) | 0.569 (0.386) |
| sym_fast | 0.101 (0.048) | 0.407 (0.239) | 0.533 (0.323) | 0.554 (0.332) |

**Readings:**

- **Hypothesis supported.** Touchdown speed falls as the motor lag is made symmetric and then fast: −2.30 → −1.52 → −0.78 m/s median. The commanded descent near touchdown is about the same in all three arms (−0.31 to −0.33 m/s).
  - The asymmetry alone accounts for ~0.8 m/s.
  - The remaining symmetric lag accounts for ~0.7 m/s more.
- **The policy uses the sink as its descent mechanism.**
  - It commands only ~−0.3 m/s over its last 0.5 s, yet hits at −2.3 m/s.
  - With fast symmetric motors, 136/1,000 episodes time out (0 with asym): without the thrust deficit, those episodes never come down within 30 s.
  - This is an exploit of the simulated actuators. Real ESCs/motors are not known to have a 10–16× faster spin-down; this should be checked against the lab hardware before Semester 2.
- **Caveats:**
  - The policy is out of its training distribution in the sym arms, so their success rates are NOT what a policy trained on symmetric motors would achieve.
  - The command over the last 0.5 s is not the steady-state command. With K_v,z 1.3–1.7, the vertical loop's time constant is ~0.6–0.75 s, so earlier commands still shape the impact speed. So "impact minus last command" is not a pure measure of the sink.
  - One seed; one checkpoint.
- **Relevance to the paper:** AerialGym's LMF2 config has the same constants. Its paper-era integrator spun down even faster (sink −0.78 vs our −0.73 m/s in the scratch study). So the paper's simulator very likely had the same sink. How the paper obtained ≲ 1 m/s descents (Fig. 9) is unknown.

### P_smoke_vz1_symslow_s1 (1 Oct 2026): variant P trained with symmetric motors (single change)

- **Config:** identical to P_smoke_vz1_s1 (seed 1, 2M steps, literal D3, ±1 m/s, full DR, tiny renderer) except `--motor sym_slow` (τ_down := τ_up, 50–80 ms). Code: `train_ppo.py --motor` (default asym); `evaluate.py` defaults to the trained motor. 165 tests pass.
- **Training:** reached L80 (c = 1); zero timeouts; at the end ~100 successes, ~20–30 crashes and 3–11 tilt per ~130 episodes per update.

Pinned evaluation: `latest.pt`, 2,002,944 steps, c = 1.0, 1,000 episodes, seed base 9000, deterministic.

| trained on | evaluated on | success [95% CI] | strict | ground / platform / tilt / drift / timeout | impact v_z median (p10 / p90) | cmd v_z last 5 | safe ≤ 1.0 (strict) | safe ≤ 2.0 (strict) |
|---|---|---|---|---|---|---|---|---|
| asym | asym | 81.5% [79.0, 83.8] | 68.6% | 158 / 24 / 3 / 0 / 0 | −2.30 (−3.07 / −1.54) | −0.31 | 0.012 (0.002) | 0.249 (0.159) |
| asym | sym_slow | 68.7% [65.8, 71.5] | 49.3% | 217 / 72 / 24 / 0 / 0 | −1.52 (−2.20 / −0.92) | −0.33 | 0.102 (0.037) | 0.569 (0.386) |
| **sym_slow** | **sym_slow** | **89.7% [87.7, 91.4]** | 78.8% | 71 / 22 / 7 / 3 / 0 | **−1.93 (−2.72 / −1.22)** | −0.58 | 0.042 (0.028) | 0.495 (0.408) |
| sym_slow | asym | 92.7% [90.9, 94.2] | 82.4% | 30 / 41 / 2 / 0 / 0 | −2.47 (−3.31 / −1.53) | −0.56 | 0.018 (0.013) | 0.238 (0.186) |

**Readings:**

- **Removing the motor asymmetry during training does NOT produce gentle landings.**
  - On its own motors the new policy lands at −1.93 m/s median, and 90% of its landings are faster than 1.22 m/s.
  - The command is capped at −1 m/s, and the sym_slow jitter sink is only ~0.2 m/s. So the policy found another way to come down faster than it can command, most likely tilt / attitude-lag thrust loss during aggressive lateral manoeuvres (unverified).
- **So the asym diagnostic identified the mechanism the first policy used, not the root cause.**
  - The root cause is the incentive. The literal D3 pays for descent, and the terminal +10 replaces the shaping on the contact step, so impact speed is never penalised, under either D3 reading.
  - Success has no speed condition. PPO therefore descends as fast as the dynamics allow, by whatever means the simulator offers.
  - Gentle landing is physically possible in this task: the oracle lands at −0.62 m/s at c = 1.
- **Success:** sym_slow-trained 89.7% vs asym-trained 81.5% (in-distribution), and 92.7% when the sym_slow policy is flown on asym motors. This is one seed each; D14 requires 3 seeds before any difference is claimed.
- **Implication for the reproduction:** touchdown speed is a property of the paper's task definition (reward + success criterion), not a simulator bug to fix. Keep the paper's motors (asym), and report safe success alongside the paper's metric. A safe-landing variant (an impact-speed condition or penalty) is a separate single-change experiment, after the reproduction.

### A_smoke_s1 (1 Oct 2026): configuration A (vision), plumbing smoke run, 100k steps

- **Config:**
  - Commit `6e18d4b`. Vision mode: ArUco + CNN + LSTM + L_est (λ 1) + r_active (auto → on).
  - asym motors, literal D3, ±1 m/s, full DR, curriculum from L10, seed 1.
  - TinyRenderer (D18), 16 × 256 steps, micro-batch 8, GPU.
  - 25 updates (102,400 steps).
- **Result: everything runs.**
  - No out-of-memory crash.
  - **57 steps/s overall** (~72 s per update), so 1M steps ≈ 4.9 h.
- **L_est (training minibatch mean)** falls steadily: 4.62 → 3.35 → 1.95 → 1.47 → 1.12 → 0.93 → 0.82 (u1, 2, 5, 8, 16, 18, 25).
- **r_active:**
  - Active on 100% of live steps throughout. τ = 0.01 is far below the current error, and at the paper's own error scale (L_est ≈ 0.3) it would stay ~100% too.
  - Mean −0.096 → −0.059 per step, falling with L_est. It saturated (−0.1) only while L_est > 1.01, i.e. the first ~17 updates.
- **Behaviour:**
  - Success per update 6/157 → 42/205 (4% → 20%).
  - Crashes dominate: ~145 per update.
  - Tilt crashes are rising slowly (2 → 17 per update).
  - Zero timeouts and zero drift. The curriculum stays at L10 (window success ≈ 0.2).
  - Explained variance 0.0 → 0.3–0.5.
- **What this does NOT establish:** whether A learns the task. 100k steps is ~5% of where variant P reached L80 (~0.5M steps).

### A_s1 (1–3 Oct 2026): configuration A (vision), seed 1, 4M steps

- **Config:**
  - Commit `e57b93c`. Vision: ArUco + CNN + LSTM + L_est (λ 1) + r_active (on).
  - asym motors, literal D3, v_z ±1 m/s (verified in config.json), full DR.
  - TinyRenderer (D18), 16 × 256, micro-batch 8, GPU.
  - 977 updates, 20.0 h, ~56 steps/s.
- **Curriculum:** L10 → L20 at ~1.1M steps, L30 ~2.6M, L40 ~3.2M, L50 ~3.9M. **It did not reach L80 (c = 1).** The final window was 0.83 at L50, so it was about to promote.
  - For comparison, variant P reached L80 at ~0.5M steps.
  - A used ~176k episodes, more than the paper's entire Fig. 5 span.
- **Training-time L_est** at each level: 0.07–0.12 (L10–L20), 0.15–0.22 (L30), 0.2–0.26 (L40), 0.35–0.5 (L50).
- **r_active:** active on 98–100% of live steps; mean −0.006 to −0.026 per step.
- **Tilt** stayed low (0–20 per update).
- **Impact v_z mean in training:** −3.6 to −4.7 m/s throughout.

Pinned evaluation: `latest.pt` at 4,001,792 steps, 1,000 episodes, seed base 9000, deterministic, renderer tiny, asym motors. The variant P rows use `P_smoke_vz1_s1/latest.pt` (2,002,944 steps, trained to L80) on the same seeds.

| c | P success / strict | A success / strict | A − P | A ground / platform / tilt / drift / timeout | A est RMSE pos / vel | P impact v_z median (cmd last 5) | A impact v_z median (cmd last 5) | A safe ≤ 1 m/s |
|---|---|---|---|---|---|---|---|---|
| 0.125 | 100.0% / 98.7% | 93.4% / 85.2% | −6.6 | 29 / 35 / 2 / 0 / 0 | 0.31 m / 0.86 m/s | −1.73 (−0.47) | −4.06 (−0.48) | 0.000 |
| 0.5 | 99.0% / 93.2% | 88.8% / 75.6% | −10.2 | 65 / 38 / 4 / 5 / 0 | 0.84 / 0.93 | −2.01 (−0.39) | −3.79 (−0.53) | 0.000 |
| 0.625 (A's last level) | 97.5% / 88.5% | 82.2% / 67.1% | −15.3 | 92 / 72 / 6 / 8 / 0 | 1.37 / 1.12 | −2.10 (−0.36) | −3.62 (−0.52) | 0.004 |
| 1.0 (A untrained) | 81.5% / 68.6% | 48.5% / 38.1% | −33.0 | 301 / 90 / 7 / 117 / 0 | 4.36 / 2.58 | −2.30 (−0.31) | −3.47 (−0.52) | 0.003 |

A's mean position-estimate error by blind age (visible / 1–15 / 16–60 / 61–120 steps):

| c | visible | 1–15 | 16–60 | 61–120 |
|---|---|---|---|---|
| 0.125 | 0.18 m | 0.36 m | 1.11 m (n 30) | — |
| 0.5 | 0.20 m | 0.51 m | 3.01 m (n 500) | — |
| 1.0 | 1.14 m | 2.64 m | 6.73 m (n 3,616) | 11.1 m (n 43) |

**Readings** (one seed each; preliminary):

- **Control is not the bottleneck at c ≤ 0.625.**
  - With the true state, the same PPO / reward / curriculum lands 97.5–100%.
  - With vision, A loses 7–15 points, and the gap grows with c in step with A's estimation error (0.31 → 0.84 → 1.37 m).
  - This points to the STATE side, especially estimation during blind stretches (pad out of view). It is not yet a clean separation: A's policy was also learned on its own estimates (see the next diagnostic).
- **At c = 0.5, A's estimation error (0.84 m / 0.93 m/s) is close to the paper's for this configuration** ("w/o keypoint encoder": 0.953 m / 1.063 m/s). The paper's 91% success, however, is at c = 1, which A never trained on.
- **Learning speed:** vision progressed ~8× slower through the curriculum than variant P, and had not finished after 4M steps.
- **A touches down at −3.5 to −4.1 m/s** while commanding about −0.5 m/s near touchdown, far faster than P (−1.7 to −2.3) on the same motors and cap. The mechanism is unverified; candidates are jittery vision actions amplifying the asym-motor sink, and tilt.
- **Safe landing (≤ 1 m/s) is ~0% at every c.**

### State-vs-control diagnostics on A_s1 (3 Oct 2026): evaluation only

- **Tool:** `evaluate.py --inject-true-state` (`policies/shin_policy.py actor_seq(inject_true_state=True)`).
  - The decision layer gets y with y[0:6] replaced by the TRUE s_rel; the image, LSTM and the other 250 latent dims are unchanged.
  - Tests: `tests/test_diag_inject.py` (2); 175 pass.
- **Run:** checkpoint `A_s1/latest.pt` (4,001,792 steps), 1,000 episodes, seed base 9000, deterministic, renderer tiny. P = `P_smoke_vz1_s1/latest.pt`.

| c | P (true state) | A (own estimate) | A + true state injected | gap recovered by injection | A on sym_fast motors |
|---|---|---|---|---|---|
| 0.5 | 99.0% [98.2, 99.5] | 88.8% [86.7, 90.6] | 89.1% [87.0, 90.9] | 0.3 of 10.2 pts (~0%) | 44.6% (395 drift, impact −1.58) |
| 0.625 | 97.5% [96.3, 98.3] | 82.2% [79.7, 84.4] | 86.4% [84.1, 88.4] | 4.2 of 15.3 pts (~27%) | — |
| 1.0 | 81.5% [79.0, 83.8] | 48.5% [45.4, 51.6] | 54.8% [51.7, 57.9] | 6.3 of 33.0 pts (~19%) | 15.1% (739 drift, impact −1.73) |

Injected-run failure counts (ground / platform / tilt / drift / timeout):

| c | A | A + inject |
|---|---|---|
| 0.5 | 65 / 38 / 4 / 5 / 0 | 52 / 50 / 4 / 3 / 0 |
| 0.625 | 92 / 72 / 6 / 8 / 0 | 68 / 60 / 2 / 6 / 0 |
| 1.0 | 301 / 90 / 7 / 117 / 0 | 300 / 80 / 6 / 66 / 0 |

**Readings** (one seed; preliminary):

- **Correcting the explicit estimate recovers little of the vision gap**: ~0% at c = 0.5, ~20–27% at c = 0.625–1.0.
  - Most of the gap to P remains even when A's decision layer is handed the true relative state.
  - So, in this test, A's shortfall is mainly in the controller learned under vision, not in the explicit estimate s̃. That points at the **control / training side**.
  - At c = 1.0 injection does halve the drift failures (117 → 66), so the estimate matters more where the pad is lost for long stretches.
- **Limits of the test:**
  - (1) The decision layer also reads 250 other latent dims computed from vision, which may carry (imperfect) state information. Injection removes error only in the explicit channel.
  - (2) The decision layer never saw true-state inputs in training. Near the estimate's accuracy, that is a small shift.
  - (3) P was trained to L80; A only to L50.
- **A's 4 m/s touchdowns are the actuator sink exploit, amplified.** With fast symmetric motors, A's impact drops to −1.6 / −1.7 m/s, but success collapses: 395 / 739 episodes drift away (vs 136 timeouts for P under the same motors). A's controller depends on the asymmetric motor sink to come down even more than P's.

### P seeds 2–3 and A_noEst_s1 (3–4 Oct 2026)

- **Runs:**
  - `P_smoke_vz1_s2`, `P_smoke_vz1_s3`: identical to `P_smoke_vz1_s1` except the seed. Both reached L80 in 2M steps (~0.4 h each).
  - `A_noEst_s1`: identical to A_s1 except `--lambda-est 0`. This disables r_active automatically; LSTM and estimation MLP are kept but never supervised. 4M steps, 19.9 h. **It never left L10.**
- **Pinned evaluation:** `latest.pt` of each run, 1,000 episodes, seed base 9000, deterministic; vision runs with renderer tiny; asym motors.

| c | P s1 | P s2 | P s3 | P mean (range) | A_s1 | A_noEst_s1 |
|---|---|---|---|---|---|---|
| 0.125 | 100.0% | 100.0% | 99.7% | 99.9% | 93.4% | **25.2%** [22.6, 28.0] |
| 0.5 | 99.0% | 99.2% | 99.3% | 99.2% (0.3) | 88.8% | **7.9%** [6.4, 9.7] |
| 0.625 | 97.5% | 97.6% | 97.1% | 97.4% (0.5) | 82.2% | **7.1%** [5.7, 8.9] |
| 1.0 | 81.5% | 89.9% | 83.8% | 85.1% (8.4) | 48.5% | **4.5%** [3.4, 6.0] |

Strict success at c = 1, by seed (s1 / s2 / s3):

| run | strict |
|---|---|
| P | 68.6 / 75.5 / 70.3% |
| A_s1 | 38.1% |
| A_noEst_s1 | 2.6% |

Impact v_z median at c = 1:

| run | impact v_z median |
|---|---|
| P | −2.30 / −2.08 / −2.16 |
| A_s1 | −3.47 |
| A_noEst_s1 | −5.02 |

A_noEst_s1 failure counts (ground / platform / tilt / drift / timeout):

| c | failures |
|---|---|
| 0.125 | 524 / 56 / 168 / 0 / 0 |
| 1.0 | 693 / 11 / 201 / 50 / 0 |

Its "est RMSE" (4.9–7.0 m) is meaningless: the head is unsupervised.

**Readings:**

- **The control ceiling is stable across seeds.**
  - P lands 99–100% up to c = 0.625 with < 1 pt spread.
  - At c = 1, P lands 81.5–89.9% (mean 85.1%): the full task is harder for control too, and seed variation is ~8 pts there.
- **Without the estimation loss, vision does not learn.**
  - A_noEst stays at L10 for 4M steps and lands only 25% even at the level it trained on. Ground crashes and tilt dominate.
  - With L_est, A reaches L50 and lands 93% at c = 0.125.
  - Same direction as the paper ("removing the learned state estimator prevents reliable curriculum progression"; 73.99% vs 91%), but much larger here: 4.5% vs 48.5% at c = 1. One plausible reason: on the ArUco base the CNN has no pretrained keypoint features, so the state supervision is the main signal shaping the representation. Untested.
- **Combined with the injection test, this refines the state-vs-control reading:**
  - (1) Learning to extract the relative state from images is the hard, essential part. Without explicit state supervision, PPO alone does not learn it here: a STATE / representation-learning problem during training.
  - (2) Once A is trained with L_est, correcting its explicit 6-number estimate at test time recovers little of its remaining gap to P (0–27%). The residual gap lies in what the vision-trained network does with its representation, not in the explicit estimate.
  - Both one-seed for vision.
