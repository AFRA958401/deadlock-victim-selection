"""
simulation.py - deadlock victim-selection simulator (minimal version).

Model (see Section 3 of the paper):
  * discrete time, horizon 500, arrivals in the first 80 time units
  * R resource types, 2 identical instances each
  * every process asks for 1-2 units of a first resource type and, with
    probability C, for 1-2 units of a second (different) type while still
    holding the first
  * when every active process is blocked, a multi-instance detector finds the
    deadlocked processes; the victim is chosen ONLY among processes that lie
    on a cycle of the wait-for graph
  * the victim loses its current-attempt work, releases its resources and
    restarts; after its 5th victimisation it is terminated
"""
import random
import numpy as np

HORIZON = 500
ARRIVAL_WINDOW = 80
MAX_VICTIMS = 5          # terminated after this many victimisations
RESTART_PENALTY = 2
TERMINATION_PENALTY = 5

POLICIES = ["Adaptive", "Fixed Weight", "Lowest Priority", "Minimum Cost", "Minimum History"]
SUPPLEMENTARY = ["Adaptive", "Ablation - cost", "Ablation - history", "Ablation - priority",
                 "Ablation - resource", "AdaptiveHighWeights", "AdaptiveLowWeights"]


class Proc:
    def __init__(self, pid, arrival, priority, r1, u1, r2, u2, d1, d2):
        self.pid, self.arrival, self.priority = pid, arrival, priority
        self.r1, self.u1, self.r2, self.u2, self.d1, self.d2 = r1, u1, r2, u2, d1, d2
        self.planned = d1 + (d2 if r2 is not None else 0)
        self.victims = 0              # how many times it was chosen as a victim
        self.reset()
        self.finish_time = None

    def reset(self):
        self.state = "future"         # future, wait1, run1, wait2, run2, done, dead
        self.work_done = 0            # work finished in the current attempt
        self.left = 0                 # time left in the current phase
        self.wait_since = 0
        self.alloc = {}               # resource type -> units held

    def request(self):                # (type, units) of the current request
        return (self.r1, self.u1) if self.state == "wait1" else (self.r2, self.u2)

    def held_units(self):
        return sum(self.alloc.values())


def make_workload(n, R, C, seed):
    """Same (n, R, C, seed) always gives the same workload, for every policy."""
    rng = random.Random(f"{n}-{R}-{C}-{seed}")
    procs = []
    for pid in range(n):
        r1 = rng.randrange(R)
        r2, u2, d2 = None, 0, 0
        if rng.random() < C:
            r2 = rng.choice([r for r in range(R) if r != r1])
            u2, d2 = rng.randint(1, 2), rng.randint(5, 15)
        procs.append(Proc(pid, rng.randint(0, ARRIVAL_WINDOW - 1), rng.randint(1, 5),
                          r1, rng.randint(1, 2), r2, u2, rng.randint(2, 5), d2))
    return procs


# ------------------------------------------------------------------ detection
def detect_deadlock(blocked, avail):
    """Multi-instance finish/work procedure. Returns (deadlocked processes, final Work)."""
    work = list(avail)
    finish = {p.pid: p.held_units() == 0 for p in blocked}
    changed = True
    while changed:
        changed = False
        for p in blocked:
            rtype, units = p.request()
            if not finish[p.pid] and units <= work[rtype]:
                for t, u in p.alloc.items():
                    work[t] += u
                finish[p.pid] = True
                changed = True
    return [p for p in blocked if not finish[p.pid]], work


def cycle_members(dead, work):
    """Keep only the deadlocked processes that lie on a cycle of the wait-for graph."""
    waits_for = {}
    for p in dead:
        rtype, units = p.request()
        waits_for[p.pid] = [q.pid for q in dead
                            if q.pid != p.pid and units > work[rtype] and q.alloc.get(rtype, 0) > 0]
    members = []
    for p in dead:
        seen, stack = set(), list(waits_for[p.pid])
        while stack:                              # can p reach itself again?
            x = stack.pop()
            if x == p.pid:
                members.append(p)
                break
            if x not in seen:
                seen.add(x)
                stack.extend(waits_for[x])
    return members


# ------------------------------------------------------------- victim policies
def expected_cost(p):
    penalty = TERMINATION_PENALTY if p.victims + 1 >= MAX_VICTIMS else RESTART_PENALTY
    return p.work_done + penalty


def score_choice(cands, weights_fn, C):
    """Four-factor score S = lc*c + lh*h + lp*p + lr*r ; smallest score is the victim."""
    max_cost = max(expected_cost(p) for p in cands)
    max_held = max(p.held_units() for p in cands)
    max_req = max(p.request()[1] for p in cands)
    h = {p.pid: min(p.victims / MAX_VICTIMS, 1) for p in cands}
    rho = sum(h.values()) / len(cands)
    lc, lh, lp, lr = weights_fn(C, rho)
    best = None
    for p in sorted(cands, key=lambda q: q.pid):  # ties -> lower process id
        c = expected_cost(p) / max_cost
        pr = (p.priority - 1) / 4
        r = ((1 - p.held_units() / max_held) + p.request()[1] / max_req) / 2
        s = lc * c + lh * h[p.pid] + lp * pr + lr * r
        if best is None or s < best[0] - 1e-12:
            best = (s, p)
    return best[1]


def adaptive_weights(lam0=0.20, alpha=0.20, beta=0.20, omit=None):
    def f(C, rho):
        lh = min(max(lam0 + alpha * C + beta * rho, 0.20), 0.60)
        rest = (1 - lh) / 0.70
        w = [0.40 * rest, lh, 0.15 * rest, 0.15 * rest]      # cost, history, priority, resource
        if omit is not None:                                  # ablation: drop one factor, renormalise
            w[omit] = 0.0
            total = sum(w)
            w = [x / total for x in w]
        return w
    return f


def choose_victim(policy, cands, C):
    if policy == "Minimum Cost":
        return min(cands, key=lambda p: (expected_cost(p), p.pid))
    if policy == "Lowest Priority":
        return min(cands, key=lambda p: (p.priority, p.pid))
    if policy == "Minimum History":
        return min(cands, key=lambda p: (p.victims, p.pid))
    if policy == "Youngest First":                            # age-based (wound-wait style)
        return max(cands, key=lambda p: (p.arrival, -p.pid))
    if policy == "Fixed Weight":
        return score_choice(cands, lambda C, rho: [0.40, 0.30, 0.15, 0.15], C)
    if policy == "Adaptive":
        return score_choice(cands, adaptive_weights(), C)
    if policy.startswith("Ablation - "):
        omit = ["cost", "history", "priority", "resource"].index(policy.split(" - ")[1])
        return score_choice(cands, adaptive_weights(omit=omit), C)
    if policy == "AdaptiveHighWeights":
        return score_choice(cands, adaptive_weights(0.30, 0.30, 0.30), C)
    if policy == "AdaptiveLowWeights":
        return score_choice(cands, adaptive_weights(0.10, 0.10, 0.10), C)
    raise ValueError(policy)


# ------------------------------------------------------------------ one run
def run(policy, n, R, C, seed):
    procs = make_workload(n, R, C, seed)
    avail = [2] * R
    stats = dict(work_lost=0, restarts=0, terminated=0, events=0)

    def release(p):
        for t, u in p.alloc.items():
            avail[t] += u
        p.alloc = {}

    def grant_waiting(now):
        waiting = sorted((p for p in procs if p.state in ("wait1", "wait2")),
                         key=lambda p: (p.wait_since, p.pid))
        for p in waiting:
            rtype, units = p.request()
            if avail[rtype] >= units:
                avail[rtype] -= units
                p.alloc[rtype] = units
                p.state = "run1" if p.state == "wait1" else "run2"
                p.left = p.d1 if p.state == "run1" else p.d2

    def settle(now):
        """Grant resources; if everybody is blocked, detect and recover (repeat)."""
        while True:
            grant_waiting(now)
            if any(p.state in ("run1", "run2") for p in procs):
                return
            blocked = [p for p in procs if p.state in ("wait1", "wait2")]
            if not blocked:
                return
            dead, work = detect_deadlock(blocked, avail)
            cands = cycle_members(dead, work) if dead else []
            if not cands:
                return
            v = choose_victim(policy, cands, C)
            stats["events"] += 1
            stats["work_lost"] += v.work_done
            v.victims += 1
            release(v)
            if v.victims >= MAX_VICTIMS:
                v.state = "dead"
                stats["terminated"] += 1
            else:
                stats["restarts"] += 1
                v.reset()
                v.state, v.wait_since = "wait1", now   # restart at the back of the queue

    for now in range(HORIZON):
        for p in procs:
            if p.state == "future" and p.arrival <= now:
                p.state, p.wait_since = "wait1", now
        settle(now)
        for p in procs:                                  # advance one time unit
            if p.state in ("run1", "run2"):
                p.left -= 1
                p.work_done += 1
                if p.left == 0:
                    if p.state == "run1" and p.r2 is not None:
                        p.state, p.wait_since = "wait2", now + 1
                    else:
                        release(p)
                        p.state, p.finish_time = "done", now + 1

    # ---- metrics
    done = [p for p in procs if p.state == "done"]
    unfinished = [p for p in procs if p.state not in ("done", "dead")]
    progress = [1.0 if p.state == "done" else 0.0 if p.state == "dead"
                else p.work_done / p.planned for p in procs]
    jain = sum(progress) ** 2 / (n * sum(x * x for x in progress)) if any(progress) else 1.0
    turn = [p.finish_time - p.arrival for p in done]
    pct = (lambda q: float(np.percentile(turn, q))) if turn else (lambda q: float("nan"))
    return dict(
        policy=policy, n_processes=n, n_resources=R, contention=C, seed=seed,
        completed=len(done), terminated=stats["terminated"], unfinished=len(unfinished),
        throughput=len(done) / HORIZON,
        recovery_cost=stats["work_lost"] + RESTART_PENALTY * stats["restarts"]
                      + TERMINATION_PENALTY * stats["terminated"],
        work_lost=stats["work_lost"], restarts=stats["restarts"],
        recovery_time=stats["events"] * 1,
        repeat_proxy=sum(1 for p in unfinished if p.victims >= 3),
        max_victimisations=max(p.victims for p in procs), jain=jain,
        mean_turnaround=float(np.mean(turn)) if turn else float("nan"),
        p95_turnaround=pct(95), p99_turnaround=pct(99))


if __name__ == "__main__":
    for pol in POLICIES:
        print(run(pol, 50, 5, 0.5, 0))
