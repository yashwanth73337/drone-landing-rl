"""LMF2 quadrotor + Lee controller parameters (V5 Block 1).

Provenance: Shin et al. Table II gains are identical to AerialGym's
`lmf2_controller_config.py` (github.com/ntnu-arl/aerial_gym_simulator,
aerial_gym/config/controller_config/), so V5 simulates the LMF2 airframe
from `aerial_gym/config/robot_config/lmf2_config.py` and
`resources/robots/lmf2/model.urdf`. Values are copied, not imported.
"""
import numpy as np

G = 9.81  # m/s^2, AerialGym gravity [0, 0, -9.81]

# ---- Airframe (lmf2/model.urdf) --------------------------------------------
BASE_MASS = 1.2                                # kg, base_link
PROP_MASS = 0.01                               # kg, x4 prop links
PROP_XY = 0.10                                 # m, prop link offsets (+-0.1, +-0.1, 0)
MASS = BASE_MASS + 4 * PROP_MASS               # 1.24 kg total
BASE_INERTIA = np.array([0.013, 0.014, 0.013])  # kg m^2, base_link diag
# Merged inertia = base + 4 point masses at (+-0.1, +-0.1, 0)
INERTIA = BASE_INERTIA + 4 * PROP_MASS * np.array(
    [PROP_XY**2, PROP_XY**2, 2 * PROP_XY**2])
COLLISION_BOX = 0.5                            # m, cube edge (base_link collision)

# ---- Control allocation (lmf2_config.control_allocator_config) --------------
# Rows: [Fz, tau_x, tau_y, tau_z]; columns: motors 0..3. The Fx, Fy rows are zero.
ALLOCATION = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [-0.13, -0.13, 0.13, 0.13],
    [-0.13, 0.13, 0.13, -0.13],
    [-0.07, 0.07, -0.07, 0.07],
])
ALLOCATION_INV = np.linalg.pinv(ALLOCATION)

# ---- Motor model (lmf2_config.motor_model_config) ---------------------------
MOTOR_MIN_THRUST = 0.1       # N
MOTOR_MAX_THRUST = 10.0      # N  -> thrust/weight = 40 / (1.24*9.81) = 3.29
TAU_INC_RANGE = (0.05, 0.08)  # s, spin-up time constant, sampled per motor
TAU_DEC = 0.005              # s, spin-down time constant
# The thrust constant (9.26e-6..1.83e-5) cancels out of AerialGym's rpm-space
# update (T_new = (sqrt T + a (sqrt T_ref - sqrt T))^2), so it is not needed.

# ---- Rigid-body damping (lmf2_config.robot_asset) ---------------------------
LINEAR_DAMPING = 0.01
ANGULAR_DAMPING = 0.01

# ---- Lee velocity controller (lmf2_controller_config) -----------------------
# Each component is sampled independently from U(min, max).
K_VEL_RANGE = (np.array([2.7, 2.7, 1.3]), np.array([3.3, 3.3, 1.7]))    # Table II
K_ROT_RANGE = (np.array([1.6, 1.6, 0.25]), np.array([1.85, 1.85, 0.4]))  # Table II
K_ANGVEL_RANGE = (np.array([0.4, 0.4, 0.075]), np.array([0.5, 0.5, 0.09]))  # config only
MAX_YAW_RATE = np.pi / 3.0   # rad/s, clamp inside compute_body_torque

# ---- Timing ------------------------------------------------------------------
PHYSICS_DT = 0.01            # s, AerialGym base_sim_config dt (100 Hz)
POLICY_DT = 0.1              # s, paper Delta t
SUBSTEPS = int(round(POLICY_DT / PHYSICS_DT))  # 10


def sample_gains(rng, mode="random"):
    """mode: 'random' | 'nominal' | 'min' | 'max'."""
    out = {}
    for name, (lo, hi) in (("K_vel", K_VEL_RANGE), ("K_rot", K_ROT_RANGE),
                           ("K_angvel", K_ANGVEL_RANGE)):
        if mode == "random":
            out[name] = rng.uniform(lo, hi)
        elif mode == "nominal":
            out[name] = 0.5 * (lo + hi)
        elif mode == "min":
            out[name] = lo.copy()
        elif mode == "max":
            out[name] = hi.copy()
        else:
            raise ValueError(mode)
    return out
