"""
continue_v3b.py
===============

TRUE continuation of V3b seed 1 into the MOVING phase of the curriculum
(levels 8-12, platform speed 0.05 -> 0.25 m/s), to answer:

    Can the frozen-LSTM TD3 policy, which lands at 85% on the stationary
    level-5 task, be trained through the moving levels -- and if it stalls,
    does it stall the way V3a did (level 9, rising timeouts)?

V3b seed 1 reached level 5/12 (radius 0.11-0.23 m, stationary) in 600k fresh
transitions. Zero-shot on a moving pad at its own radius it scored 64% at
0.05 m/s and 36% at 0.10 m/s, failing mostly by timeout. It has never trained
on a moving platform. This script continues its training from the checkpoint
it already has.


WHY THIS IS A SEPARATE FILE
------------------------------------------------------------------------------
train_lstm_v3b.py stays a from-scratch trainer, so V3b's headline result can
still be described as "fresh TD3, no warm start". This script is an extension
experiment and must be reported as "600k fresh + N continued", never as one
uninterrupted budget -- exactly as continue_v3a.py is for V3a.


WHAT IS RESTORED, AND WHAT IS NOT
------------------------------------------------------------------------------
Restored exactly, from a PAIRED model zip + replay buffer pickle:

    actor, critic, actor_target, critic_target weights
    actor.optimizer and critic.optimizer state (Adam moments)
    replay buffer
    num_timesteps, via learn(reset_num_timesteps=False)
    action noise configuration

Only PAIRED checkpoints are accepted (--resume-from):

    final          final_model.zip          + final_replay_buffer.pkl
    best_success   best_success_model.zip   + best_success_replay_buffer.pkl

Each pair was written at the same timestep, so the buffer holds exactly the
transitions the model had trained on. That is checked after loading: buffer
rows x n_envs must equal num_timesteps (final_*) or num_timesteps - 4
(best_success_*, saved inside the callback, which SB3 runs before storing the
current step's transition).

best_model.zip is NOT accepted. In these run directories it is written by
SB3's EvalCallback -- best MEAN REWARD over 5 episodes at whatever level was
current -- not by the curriculum callback, and no replay buffer was saved with
it. Resuming from it with either buffer would pair a policy with transitions
it did not generate.

NOT restored -- the same honest gap as continue_v3a.py:

    numpy / torch RNG state. SB3 does not checkpoint the exploration stream.


START LEVEL IS REQUIRED, ON PURPOSE
------------------------------------------------------------------------------
--start-level has no default. Seed 1 last trained at level 5. Starting at
level 8 skips levels 6-7 (radius 0.26, 0.30 m) that it never trained on, so
the first continued level changes TWO things at once: spawn radius
0.23 -> 0.30 m AND platform speed 0.00 -> 0.05 m/s. Starting at 6 lets the
normal promotion rule carry it into the moving phase. Both are legitimate;
they are different experiments. The script prints what changes and warns on
skipped levels; the choice is made on the command line, not here.


CHECKPOINTING -- THE 19 SEP FIX, WHICH THE V3b RUNS DID NOT HAVE
------------------------------------------------------------------------------
Uses train_aruco_v3a.CurriculumAndCheckpointCallback as it is NOW, which
includes the per-level best checkpoints (best_LNN.zip + best_per_level.csv)
and the "new global best is below the best seen at any level" warning. The
original V3b runs predate that fix.

Scope, stated plainly:

  * best_success_model.zip keeps the level-dominant rule, and its key is
    seeded by replaying the source run's CSV (as continue_v3a.py does), so an
    easier level cannot overwrite a harder-level save.
  * best_LNN.zip, best_per_level.csv and the regression warning are
    CONTINUATION-SCOPED. They start empty in the new directory: the source
    run has no per-level files to point at, and seeding them would make the
    manifest reference checkpoints that do not exist.


CSV HISTORY
------------------------------------------------------------------------------
The source run directory is NEVER modified. The new directory's
success_evaluations.csv and perception_evaluations.csv start as a copy of the
source rows up to and including the resume timestep, then the continuation
appends to them in the identical column format. Rows AFTER the resume
timestep (possible with --resume-from best_success) belong to a discarded
trajectory and are not copied. Headers are checked against the live code
before copying; a mismatch aborts rather than producing a mixed file.

continuation.json records the source run, checkpoint pair, resume timestep,
start level, estimator path and SHA-256, git commit, and command line.


WHAT IS DELIBERATELY NOT CHANGED
------------------------------------------------------------------------------
Estimator (frozen, temporal_estimator_v3b_r2.pt), actor_obs_source='lstm',
15-D observation with no validity bit, reward, camera, marker, TD3
architecture and hyperparameters, curriculum levels, promotion threshold and
cadence, 4 parallel envs, evaluation seeds (promotion 10000, perception
20000). Env construction goes through train_lstm_v3b.make_kwargs and the
evaluators are routed through train_lstm_v3b._swap_env_class, so training,
promotion and perception evaluation all see the LSTM observation.


HAZARDS NAMED IN ADVANCE
------------------------------------------------------------------------------
1. Buffer composition. The restored buffer is entirely STATIONARY data
   (levels 1-5). Oldest data starts evicting only past 1M total transitions.
   If progress stalls as speed ramps in, buffer composition is a hypothesis
   to test -- the same one pre-registered in continue_v3a.py.
2. Distributional feedback on the estimator. The LSTM was validated
   PASSIVELY (train_lstm_v3b.py docstring). Once the policy adapts to a
   moving pad it visits new states. Check LSTM error, not just success.
   CAUTION: pos_err_m / vel_err_ms in perception_evaluations.csv do NOT
   measure the LSTM. v3a.evaluate_policy_full averages info['est_err_*'] --
   the parent's constant-velocity estimate -- and only on frames where the
   marker was DETECTED. It is visible-frame measurement error. The LSTM's
   blind-frame error (info['lstm_err_pos_norm']) is not in that CSV. The
   format is kept unchanged here so rows stay comparable with the source
   runs; LSTM blind-frame error needs a separate diagnostic.
3. V3a comparison. V3a entered levels 8-9 and regressed at 9 (40.2% ->
   21.2%), driven by timeouts. The pre-stated comparison is: does V3b stall
   at the same level, by the same failure mode, and does below_platform
   stay at zero?
4. Promotion noise. One 20-episode eval at >=90%. "Level reached" is partly
   luck; conclusions come from pinned, matched-task evaluations afterwards.


USAGE (from src/scripts)
------------------------------------------------------------------------------
Inspect first -- prints everything it would restore and exits, loads nothing:

    python continue_v3b.py \\
        --run-dir results_lstm_v3b/run_20260914_195837 \\
        --resume-from final --start-level 8 \\
        --estimator ../models/temporal_estimator_v3b_r2.pt --verify

Then continue:

    python continue_v3b.py \\
        --run-dir results_lstm_v3b/run_20260914_195837 \\
        --resume-from final --start-level 8 \\
        --estimator ../models/temporal_estimator_v3b_r2.pt \\
        --additional-steps 600000
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import types
import zipfile

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.LSTMLanderAviary import LSTMLanderAviary          # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402
# Imported AFTER v3a, so train_lstm_v3b captures the ORIGINAL
# v3a.make_env_kwargs before anything is rebound.
import train_lstm_v3b as v3b                                # noqa: E402


EXPECTED_ESTIMATOR = 'temporal_estimator_v3b_r2.pt'
FIRST_MOVING_LEVEL = 8                                      # 1-based

CHECKPOINT_PAIRS = {
    'final': ('final_model.zip', 'final_replay_buffer.pkl'),
    'best_success': ('best_success_model.zip',
                     'best_success_replay_buffer.pkl'),
}

# Must match the row dict in v3a.PerceptionDiagnosticCallback._on_step.
PERCEPTION_COLUMNS = [
    'timestep', 'level', 'platform_speed_range', 'success_rate',
    'blind_fraction', 'longest_blind_run', 'steps_before_first_detection',
    'pos_err_m', 'vel_err_ms',
]


# ===========================================================================
# HELPERS -- nothing here loads a model or touches an RNG
# ===========================================================================

def success_columns():
    """Column order the live callback will write, derived from its own code
    rather than hardcoded, so a later change to _build_record is caught."""
    snapshot = {'level': 0, 'radius_min': 0.0, 'radius_max': 0.0,
                'platform_speed': 0.0}
    res = {'reasons': {}, 'successes': 0, 'episodes': 0,
           'success_rate': 0.0, 'precision_mean_cm': float('inf'),
           'mean_return': 0.0, 'near_miss': 0}
    dummy = types.SimpleNamespace(num_timesteps=0)
    rec = v3a.CurriculumAndCheckpointCallback._build_record(
        dummy, snapshot, res)
    return list(rec.keys())


def read_csv(path):
    """(header, rows) or (None, []) if the file does not exist."""
    if not os.path.exists(path):
        return None, []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def rows_upto(rows, timestep):
    return [r for r in rows if int(r['timestep']) <= timestep]


def peek_num_timesteps(model_zip):
    """Read num_timesteps from the SB3 zip's JSON 'data' entry without
    loading any weights, so --verify can show the resume point cheaply."""
    try:
        with zipfile.ZipFile(model_zip) as z:
            data = json.loads(z.read('data').decode('utf-8'))
        return int(data['num_timesteps'])
    except Exception:                                       # noqa: BLE001
        return None


def replay_best_key(rows):
    """Identical rule to continue_v3a.replay_best_key:
    lexicographic (level, success_rate, -precision), successes > 0."""
    best_level, best_rate, best_prec = -1, -1.0, float('inf')
    for r in rows:
        if int(r['successes']) <= 0:
            continue
        level = int(r['level']) - 1
        rate = float(r['success_rate'])
        prec_raw = r['mean_success_distance_cm']
        prec = float(prec_raw) if prec_raw not in ('', None) else float('inf')
        if (level, rate, -prec) > (best_level, best_rate, -best_prec):
            best_level, best_rate, best_prec = level, rate, prec
    return best_level, best_rate, best_prec


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here,
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'],
                                    cwd=here, capture_output=True, text=True,
                                    check=True).stdout.strip())
        return {'commit': commit, 'dirty': dirty}
    except Exception:                                       # noqa: BLE001
        return {'commit': None, 'dirty': None}


def level_desc(level_1based):
    hi, speed = v3a.CURRICULUM_LEVELS[level_1based - 1]
    kind = 'moving' if speed > 0 else 'stationary'
    return (f'level {level_1based:2d}  radius {v3a.RADIUS_MIN:.2f}-{hi:.2f} m'
            f'  speed {speed:.2f} m/s  ({kind})')


# ===========================================================================
# PLAN
# ===========================================================================

def build_plan(args):
    """Everything --verify prints. Loads no weights."""
    problems = []

    est = os.path.abspath(args.estimator)
    if not os.path.exists(est):
        problems.append(f'estimator not found: {est}')
    elif os.path.basename(est) != EXPECTED_ESTIMATOR:
        problems.append(
            f'estimator is {os.path.basename(est)}, but V3b seed 1 was '
            f'trained on {EXPECTED_ESTIMATOR}. A different estimator changes '
            f'the observation -- that is a new experiment, not a '
            f'continuation.')

    model_name, buffer_name = CHECKPOINT_PAIRS[args.resume_from]
    model_path = os.path.join(args.run_dir, model_name)
    buffer_path = os.path.join(args.run_dir, buffer_name)
    for p in (model_path, buffer_path):
        if not os.path.exists(p):
            problems.append(f'{p} not found. A true continuation needs the '
                            f'model AND its paired replay buffer.')

    n_levels = len(v3a.CURRICULUM_LEVELS)
    if not 1 <= args.start_level <= n_levels:
        problems.append(f'--start-level must be 1..{n_levels}, '
                        f'got {args.start_level}')

    succ_path = os.path.join(args.run_dir, 'success_evaluations.csv')
    perc_path = os.path.join(args.run_dir, 'perception_evaluations.csv')
    succ_header, succ_rows = read_csv(succ_path)
    perc_header, perc_rows = read_csv(perc_path)
    if succ_header is None or not succ_rows:
        problems.append(f'{succ_path} missing or empty -- cannot recover '
                        f'the curriculum history.')

    resume_ts = (peek_num_timesteps(model_path)
                 if os.path.exists(model_path) else None)
    if resume_ts is None and os.path.exists(model_path):
        problems.append(f'could not read num_timesteps from {model_path}')

    expected_succ = success_columns()
    if succ_header is not None and succ_header != expected_succ:
        problems.append(
            'success_evaluations.csv header differs from the live callback '
            f'columns.\n      source: {succ_header}\n      live  : '
            f'{expected_succ}')
    if perc_header is not None and perc_header != PERCEPTION_COLUMNS:
        problems.append(
            'perception_evaluations.csv header differs from the live '
            f'callback columns.\n      source: {perc_header}\n      live  : '
            f'{PERCEPTION_COLUMNS}')

    plan = {
        'problems': problems,
        'estimator': est,
        'model_path': model_path,
        'buffer_path': buffer_path,
        'resume_ts': resume_ts,
        'succ_header': succ_header,
        'perc_header': perc_header,
        'succ_rows': [],
        'perc_rows': [],
        'reached_level': None,
        'resume_row': None,
    }
    if resume_ts is not None and succ_rows:
        kept = rows_upto(succ_rows, resume_ts)
        plan['succ_rows'] = kept
        plan['perc_rows'] = rows_upto(perc_rows, resume_ts)
        plan['dropped_succ'] = len(succ_rows) - len(kept)
        if kept:
            plan['reached_level'] = int(kept[-1]['level'])
            # The callback promotes AFTER writing the row, so a >=90% last
            # row means the policy was already training one level higher.
            plan['promoted_after_last'] = (
                float(kept[-1]['success_rate'])
                >= v3a.PROMOTE_THRESHOLD * 100.0
                and plan['reached_level'] < len(v3a.CURRICULUM_LEVELS))
        match = [r for r in succ_rows if int(r['timestep']) == resume_ts]
        plan['resume_row'] = match[-1] if match else None
        plan['best_key'] = replay_best_key(kept)
    return plan


def describe(args, plan):
    print('=' * 74)
    print('  V3b CONTINUATION -- RESTORE PLAN')
    print('=' * 74)
    print(f'  source run    : {args.run_dir}')
    print(f'  resume from   : {args.resume_from}')
    for label, p in (('model', plan['model_path']),
                     ('replay buffer', plan['buffer_path'])):
        size = (f'{os.path.getsize(p) / 1e6:.1f} MB'
                if os.path.exists(p) else 'MISSING')
        print(f'  {label:<14}: {os.path.basename(p)} ({size})')
    print(f'  estimator     : {os.path.basename(plan["estimator"])} (FROZEN)')
    print('  actor obs     : LSTM (15-D, no validity bit)')
    print()

    ts = plan['resume_ts']
    print('  RESUME POINT')
    print(f'    num_timesteps (from model zip)      {ts}')
    row = plan['resume_row']
    if row is not None:
        print(f'    CSV row at that timestep            level {row["level"]}'
              f', {row["success_rate"]}% '
              f'({row["successes"]}/{row["episodes"]})')
    elif ts is not None:
        print('    CSV row at that timestep            NONE -- expected for '
              "'final' if the run ended between evaluations; suspicious for "
              "'best_success'.")
    if plan.get('dropped_succ'):
        print(f'    rows after resume point             '
              f'{plan["dropped_succ"]} (discarded trajectory, NOT copied)')
    if 'best_key' in plan:
        bl, br, bp = plan['best_key']
        print(f'    seeded best_success key             level {bl + 1}, '
              f'{br:.1f}%, {bp:.3f} cm')
    print()

    reached = plan['reached_level']
    print('  CURRICULUM')
    if reached is not None:
        print(f'    last evaluated at   {level_desc(reached)}')
        if plan.get('promoted_after_last'):
            print(f'    NOTE: that evaluation passed the promotion threshold, '
                  f'so the policy was')
            print(f'          already training at level {reached + 1} when '
                  f'this checkpoint was saved.')
    if 1 <= args.start_level <= len(v3a.CURRICULUM_LEVELS):
        print(f'    will START at       {level_desc(args.start_level)}')
        if reached is not None:
            r_hi, r_sp = v3a.CURRICULUM_LEVELS[reached - 1]
            s_hi, s_sp = v3a.CURRICULUM_LEVELS[args.start_level - 1]
            changes = []
            if s_hi != r_hi:
                changes.append(f'radius_max {r_hi:.2f} -> {s_hi:.2f} m')
            if s_sp != r_sp:
                changes.append(f'speed {r_sp:.2f} -> {s_sp:.2f} m/s')
            print(f'    first-level change  '
                  f'{", ".join(changes) if changes else "none"}')
            if len(changes) > 1:
                print('    NOTE: more than one task variable changes at the '
                      'first continued level.')
            if args.start_level > reached + 1:
                skipped = list(range(reached + 1, args.start_level))
                print(f'    WARNING: skips level(s) {skipped}, never trained '
                      f'on by this policy:')
                for lv in skipped:
                    print(f'               {level_desc(lv)}')
        if args.start_level < FIRST_MOVING_LEVEL:
            print(f'    moving phase begins at level {FIRST_MOVING_LEVEL} '
                  f'via the normal promotion rule.')
    print()

    print('  NOT RESTORED')
    print('    numpy / torch RNG state (SB3 does not checkpoint it).')
    print('    Report as "600k fresh + N continued", never as one budget.')
    print()
    print('  CONTINUATION-SCOPED (start empty in the new directory)')
    print('    best_LNN.zip, best_per_level.csv, below-best-at-any-level '
          'warning')
    if args.additional_steps is not None and ts is not None:
        print()
        print(f'  additional steps : {args.additional_steps}')
        print(f'  total after run  : {ts + args.additional_steps}')

    if plan['problems']:
        print()
        print('  PROBLEMS -- will refuse to train:')
        for p in plan['problems']:
            print(f'    - {p}')
    print('=' * 74)


# ===========================================================================
# CONTINUE
# ===========================================================================

def write_history(out_dir, plan):
    """Seed the new directory's CSVs with the source history up to the
    resume timestep. The callbacks then append (they only write a header
    when the file does not exist)."""
    for name, header, rows in (
            ('success_evaluations.csv', plan['succ_header'],
             plan['succ_rows']),
            ('perception_evaluations.csv', plan['perc_header'],
             plan['perc_rows'])):
        if header is None:
            continue
        with open(os.path.join(out_dir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)


def continue_run(args):
    plan = build_plan(args)
    describe(args, plan)

    if args.verify:
        print('\n  --verify: nothing was loaded or trained. Exiting.')
        return
    if plan['problems']:
        print('\nRefusing to continue.')
        sys.exit(1)
    if args.additional_steps is None or args.additional_steps <= 0:
        print('\n--additional-steps is required (a positive integer). There '
              'is no default on purpose.')
        sys.exit(1)
    if not v3a.verify_curriculum_port():
        print('\nRefusing to train on a diverged curriculum port.')
        sys.exit(1)

    out_dir = os.path.join(args.outdir, time.strftime('cont_%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n  output: {out_dir}')
    print('  the source run directory is NEVER modified.\n')

    start_idx = args.start_level - 1
    radius_max, speed = v3a.CURRICULUM_LEVELS[start_idx]
    radius_min = v3a.RADIUS_MIN

    env_kwargs = v3b.make_kwargs(radius_min, radius_max, speed,
                                 plan['estimator'])
    train_env = make_vec_env(LSTMLanderAviary, n_envs=v3a.N_TRAIN_ENVS,
                             seed=args.seed, env_kwargs=env_kwargs)
    eval_env = LSTMLanderAviary(**env_kwargs)

    print('  loading model...')
    model = TD3.load(plan['model_path'], env=train_env,
                     tensorboard_log=os.path.join(out_dir, 'tb'))
    print(f'  loaded. num_timesteps = {model.num_timesteps}')
    if model.num_timesteps != plan['resume_ts']:
        print(f'\nERROR: loaded num_timesteps {model.num_timesteps} != '
              f'peeked {plan["resume_ts"]}.')
        sys.exit(1)

    if model.action_noise is None:
        print('\nERROR: no action noise restored. Training without it would '
              'change exploration -- an experimental parameter. Refusing.')
        sys.exit(1)
    net_arch = (model.policy_kwargs or {}).get('net_arch')
    if net_arch is not None and list(net_arch) != list(v3a.NET_ARCH):
        print(f'\nERROR: net_arch {net_arch} != {v3a.NET_ARCH}. Refusing.')
        sys.exit(1)
    print(f'  hyperparameters: lr={model.learning_rate} '
          f'batch={model.batch_size} buffer={model.buffer_size} '
          f'gamma={model.gamma} tau={model.tau} '
          f'policy_delay={model.policy_delay} net_arch={net_arch}')
    print(f'  action noise   : {model.action_noise}')

    print('  loading replay buffer...')
    model.load_replay_buffer(plan['buffer_path'])
    rb = model.replay_buffer
    rb_envs = getattr(rb, 'n_envs', 1)
    transitions = rb.size() * rb_envs
    print(f'  loaded. {rb.size()} rows x {rb_envs} envs = '
          f'{transitions} transitions (full={rb.full})')

    if rb.size() == 0:
        print('\nERROR: replay buffer loaded empty. Refusing -- this is the '
              'destructive fine-tuning condition.')
        sys.exit(1)
    if rb_envs != v3a.N_TRAIN_ENVS:
        print(f'\nERROR: buffer has {rb_envs} envs, training uses '
              f'{v3a.N_TRAIN_ENVS}. Refusing.')
        sys.exit(1)
    # SB3's collect_rollouts increments num_timesteps and calls the callback
    # BEFORE _store_transition. So a pair saved from inside the callback
    # (best_success_*) holds num_timesteps - n_envs transitions; a pair saved
    # after learn() returns (final_*) holds exactly num_timesteps.
    expected = {model.num_timesteps, model.num_timesteps - rb_envs}
    if not rb.full and transitions not in expected:
        msg = (f'buffer holds {transitions} transitions but the model '
               f'trained for {model.num_timesteps} (expected '
               f'{sorted(expected)}). These files may not be a pair.')
        if not args.allow_buffer_mismatch:
            print(f'\nERROR: {msg} Refusing. (--allow-buffer-mismatch '
                  f'overrides, and must then be reported.)')
            sys.exit(1)
        print(f'\n  WARNING: {msg} Continuing because '
              f'--allow-buffer-mismatch was passed.\n')

    # ---- history + manifest ------------------------------------------
    write_history(out_dir, plan)
    manifest = {
        'experiment': 'V3b continuation into moving curriculum',
        'source_run': os.path.abspath(args.run_dir),
        'resume_from': args.resume_from,
        'model': os.path.abspath(plan['model_path']),
        'replay_buffer': os.path.abspath(plan['buffer_path']),
        'resume_timestep': model.num_timesteps,
        'buffer_transitions': transitions,
        'last_evaluated_level': plan['reached_level'],
        'start_level': args.start_level,
        'start_task': {'radius_min': radius_min, 'radius_max': radius_max,
                       'platform_speed': speed},
        'additional_steps': args.additional_steps,
        'seed': args.seed,
        'estimator': plan['estimator'],
        'estimator_sha256': sha256(plan['estimator']),
        'actor_obs_source': 'lstm',
        'rng_state_restored': False,
        'allow_buffer_mismatch': bool(args.allow_buffer_mismatch),
        'history_rows_copied': {'success': len(plan['succ_rows']),
                                'perception': len(plan['perc_rows'])},
        'git': git_state(),
        'argv': sys.argv,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out_dir, 'continuation.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # ---- curriculum + callbacks --------------------------------------
    cur = v3a.CurriculumState(radius_min=radius_min, radius_max=radius_max,
                              platform_speed=speed)
    cur.level = start_idx

    ckpt_cb = v3a.CurriculumAndCheckpointCallback(
        out_dir, cur, eval_envs=[eval_env])
    ckpt_cb.level = start_idx
    bl, br, bp = plan['best_key']
    ckpt_cb.best_level = bl
    ckpt_cb.best_success_rate = br
    ckpt_cb.best_precision = bp
    # best_by_level / best_any_* deliberately left empty: continuation-scoped.
    ckpt_cb.last_eval_timestep = model.num_timesteps

    perception = v3a.PerceptionDiagnosticCallback(out_dir, cur)
    perception.last_eval_timestep = model.num_timesteps

    callbacks = CallbackList([
        EvalCallback(
            eval_env, best_model_save_path=out_dir, log_path=out_dir,
            eval_freq=max(v3a.REWARD_EVAL_EVERY_TIMESTEPS
                          // v3a.N_TRAIN_ENVS, 1),
            n_eval_episodes=5, deterministic=True, render=False),
        ckpt_cb,
        perception,
    ])

    total_after = model.num_timesteps + args.additional_steps
    print(f'\n  resuming at {level_desc(args.start_level)}')
    print(f'  training {args.additional_steps} additional transitions -> '
          f'{total_after} total\n')

    # Route every evaluator through LSTMLanderAviary, exactly as
    # train_lstm_v3b.train() does. Without this, promotion and perception
    # evaluations would silently score the constant-velocity observation.
    token = v3b._swap_env_class(plan['estimator'])
    try:
        t0 = time.time()
        model.learn(total_timesteps=args.additional_steps,
                    callback=callbacks, reset_num_timesteps=False)
        elapsed = time.time() - t0
    finally:
        v3b._restore_env_class(token)

    model.save(os.path.join(out_dir, 'final_model'))
    model.save_replay_buffer(os.path.join(out_dir, 'final_replay_buffer.pkl'))

    manifest['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest['final_timestep'] = int(model.num_timesteps)
    manifest['elapsed_hours'] = round(elapsed / 3600, 3)
    with open(os.path.join(out_dir, 'continuation.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\n  done in {elapsed / 3600:.2f} h. saved to {out_dir}')
    print(f'  total transitions now: {model.num_timesteps}')
    print('\n  Levels reached, and the best checkpoint at EACH level:')
    print(f'    tail -5 {out_dir}/success_evaluations.csv')
    print(f'    cat {out_dir}/best_per_level.csv')
    print('\n  best_success_model.zip is level-dominant. Pick checkpoints '
          'from best_per_level.csv,')
    print('  and evaluate each with --radius-min/--radius-max/--speed '
          'pinned to its level.')


def main():
    ap = argparse.ArgumentParser(
        description='Continue V3b seed 1 into the moving curriculum.')
    ap.add_argument('--run-dir', required=True,
                    help='source V3b run directory')
    ap.add_argument('--resume-from', required=True,
                    choices=sorted(CHECKPOINT_PAIRS),
                    help="'final' = final_model + final_replay_buffer; "
                         "'best_success' = best_success_model + "
                         "best_success_replay_buffer")
    ap.add_argument('--start-level', type=int, required=True,
                    help='1-based curriculum level to resume training at '
                         '(8 = first moving level)')
    ap.add_argument('--estimator', required=True,
                    help=f'frozen estimator, must be {EXPECTED_ESTIMATOR}')
    ap.add_argument('--additional-steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--outdir', default='results_lstm_v3b_cont')
    ap.add_argument('--verify', action='store_true',
                    help='print the restore plan and exit; loads nothing')
    ap.add_argument('--allow-buffer-mismatch', action='store_true',
                    help='continue even if buffer size != model timesteps '
                         '(recorded in continuation.json)')
    continue_run(ap.parse_args())


if __name__ == '__main__':
    main()
