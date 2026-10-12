"""Read-only paired comparison of pinned evaluations (episode i = seed 9000+i, same episodes in every run).
Usage (from ~/mtp/drone-landing-rl):  python /tmp/paired_eval.py
Reads v5_shin/runs/*/eval_*/episodes.csv only; writes nothing."""
import csv, math, os, sys

RUNS = "v5_shin/runs"

def path(run, ckpt, c, inj=False):
    return f"{RUNS}/{run}/eval_{ckpt}_c{c}_s9000_det{'_injtrue' if inj else ''}/episodes.csv"

def load(p):
    with open(p, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        cols = rd.fieldnames
    if "outcome" in cols:
        ok = [r["outcome"].strip().lower() == "success" for r in rows]
        how = "outcome == 'success'"
    elif "success" in cols:
        ok = [r["success"].strip().lower() in ("1", "1.0", "true", "yes") for r in rows]
        how = "success column"
    else:
        sys.exit(f"cannot find an outcome/success column in {p}; columns are: {cols}")
    key = next((k for k in ("seed", "episode", "episode_id", "ep", "idx", "index") if k in cols), None)
    keys = [r[key] for r in rows] if key else list(range(len(rows)))
    return dict(zip(keys, ok)), how, key

def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

def compare(label, pa, pb):
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"{label}: skipped (missing {pa if not os.path.exists(pa) else pb})")
        return
    A, how, key = load(pa)
    B, _, _ = load(pb)
    common = sorted(set(A) & set(B), key=str)
    if len(common) != len(A) or len(common) != len(B):
        print(f"{label}: WARNING only {len(common)} episodes in common (A {len(A)}, B {len(B)})")
    n = len(common)
    b = sum(1 for k in common if A[k] and not B[k])   # success in A only
    c = sum(1 for k in common if B[k] and not A[k])   # success in B only
    sa = sum(1 for k in common if A[k]); sb = sum(1 for k in common if B[k])
    d = (c - b) / n
    se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0)) / n
    print(f"{label}\n   A {sa}/{n} = {100*sa/n:.1f}%   B {sb}/{n} = {100*sb/n:.1f}%   "
          f"B-A = {100*d:+.1f} pts  (paired 95% CI {100*(d-1.96*se):+.1f} to {100*(d+1.96*se):+.1f})\n"
          f"   success in A only: {b}   in B only: {c}   exact McNemar p = {mcnemar_p(b, c):.4f}"
          f"   [matched on {key or 'row order'}; success = {how}]")

print("== Training progress, c = 1.0, latest.pt (A = earlier, B = later)")
chain = [("A_s2", "4M"), ("A_s2_cont", "8M"), ("A_s2_cont2", "12M"), ("A_s2_cont3", "16M"), ("A_s2_cont4", "20M")]
for (r1, s1), (r2, s2) in zip(chain, chain[1:]):
    compare(f"{s1} -> {s2}", path(r1, "latest", "1.0"), path(r2, "latest", "1.0"))

print("\n== True-state injection (A = normal, B = injected), latest.pt")
for run, s in chain[1:]:
    for c in ("1.0", "0.5"):
        compare(f"{s}, c = {c}", path(run, "latest", c), path(run, "latest", c, inj=True))

print("\n== Easy difficulty, c = 0.125, latest.pt")
compare("8M -> 20M", path("A_s2_cont", "latest", "0.125"), path("A_s2_cont4", "latest", "0.125"))
compare("16M -> 20M", path("A_s2_cont3", "latest", "0.125"), path("A_s2_cont4", "latest", "0.125"))
