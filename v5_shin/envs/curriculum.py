"""Curriculum on platform motion (V5 Block 10; Shin et al. Sec. III-D2, Fig. 3; SPEC §7, D5).

  levels 10, 20, ..., 80 (Fig. 3: "Level 10 -> ... -> Level 80"), c = level / 80
  "Level updated every 512 episodes" (Fig. 3). Criterion [unspecified] -> D5:
  after every 512 COMPLETED episodes at the current level (pooled over all envs,
  non-overlapping windows), promote one level if success >= 80%. No demotion.

Episodes that STARTED at an older c (in flight when a promotion happened) are not
counted toward the new level's window: every result is tagged with the c it was
started at.
"""
import numpy as np

LEVELS = tuple(range(10, 81, 10))
WINDOW = 512
THRESHOLD = 0.80


def level_to_c(level):
    return level / 80.0


class Curriculum:
    def __init__(self, start_level=10, window=WINDOW, threshold=THRESHOLD):
        assert start_level in LEVELS
        self.level = start_level
        self.window = window
        self.threshold = threshold
        self.n = 0                  # episodes counted in the current window
        self.k = 0                  # successes in the current window
        self.total_episodes = 0     # all completed episodes seen (incl. stale ones)
        self.stale = 0              # episodes ignored because they started at an older c
        self.history = []           # one row per closed window

    @property
    def c(self):
        return level_to_c(self.level)

    def record(self, success, c_episode):
        """Report one finished episode. Returns True if this closed a window and promoted."""
        self.total_episodes += 1
        if not np.isclose(c_episode, self.c):
            self.stale += 1
            return False
        self.n += 1
        self.k += int(bool(success))
        if self.n < self.window:
            return False
        rate = self.k / self.n
        promoted = rate >= self.threshold and self.level < LEVELS[-1]
        self.history.append(dict(total_episodes=self.total_episodes, level=self.level,
                                 window_success=rate, promoted=promoted))
        if promoted:
            self.level = LEVELS[LEVELS.index(self.level) + 1]
        self.n = self.k = 0
        return promoted

    # ---- checkpointing ---------------------------------------------------------
    def state_dict(self):
        return dict(level=self.level, window=self.window, threshold=self.threshold, n=self.n,
                    k=self.k, total_episodes=self.total_episodes, stale=self.stale,
                    history=list(self.history))

    @classmethod
    def from_state_dict(cls, d):
        cur = cls(start_level=d["level"], window=d["window"], threshold=d["threshold"])
        for key in ("n", "k", "total_episodes", "stale"):
            setattr(cur, key, d[key])
        cur.history = list(d["history"])
        return cur


def false_promotion_probability(p_true, window=WINDOW, threshold=THRESHOLD):
    """P(one window promotes | true success rate p_true), exact binomial tail."""
    from math import comb, ceil
    k_min = ceil(threshold * window)
    return float(sum(comb(window, k) * p_true ** k * (1 - p_true) ** (window - k)
                     for k in range(k_min, window + 1)))
