#!/usr/bin/env python3
"""
experiment_runner.py - experiment driver for simulation.py (deadlock victim selection).

Runs the experiment design described in Section 3.4 / Appendix B of
"An Adaptive Four-Factor Victim-Selection Policy for Operating-System Deadlock
Recovery" using the UNMODIFIED simulation.py, and writes one CSV row per run.

Place this file in the same folder as simulation.py.

Usage (see the bottom of this docstring for the full list):
    python experiment_runner.py                      # test mode (default): 1 config x 2 seeds x 5 policies
    python experiment_runner.py --mode main          # 3x3x3 configs x 30 seeds x 5 policies = 4,050 runs
    python experiment_runner.py --mode supplementary # 3x3x3 configs x 10 seeds x 7 variants = 1,890 runs
    python experiment_runner.py --mode all           # main + supplementary
    add --workers 4 to use 4 processes; add --outdir DIR to choose the output folder.

What is NOT done here (not requested, and not available in simulation.py):
  * no statistical analysis (Wilcoxon / Holm / bootstrap) - this script only produces run-level data;
  * no new metrics: every metric column is exactly a key returned by simulation.run().
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np          # noqa: E402  (simulation.py itself requires numpy)
import simulation as sim    # noqa: E402  (imported, never modified)

# --------------------------------------------------------------------------
# Experiment design, taken from the manuscript (Section 3.4 and Appendix B)
# --------------------------------------------------------------------------
PROCESS_COUNTS = (20, 50, 100)
RESOURCE_COUNTS = (3, 5, 8)
CONTENTION_LEVELS = (0.2, 0.5, 0.8)     # keep as these exact floats: the workload RNG is seeded
                                        # with a string built from C, so 0.2 != 0.20000000000000001
MAIN_SEEDS = tuple(range(0, 30))        # seeds 0-29
SUPPLEMENTARY_SEEDS = tuple(range(0, 10))   # seeds 0-9

# Policy lists are read from simulation.py (not retyped) and checked against the manuscript.
MAIN_POLICIES = tuple(sim.POLICIES)
SUPPLEMENTARY_VARIANTS = tuple(sim.SUPPLEMENTARY)

# Test mode: ONE configuration, TWO seeds (same configuration as simulation.py's __main__ block).
TEST_CONFIG = (50, 5, 0.5)              # (n_processes, n_resources, contention)
TEST_SEEDS = (0, 1)

# Manuscript values that simulation.py is checked against at start-up.
MANUSCRIPT_CONSTANTS = dict(HORIZON=500, ARRIVAL_WINDOW=80, MAX_VICTIMS=5,
                            RESTART_PENALTY=2, TERMINATION_PENALTY=5)
MANUSCRIPT_MAIN_POLICIES = ("Adaptive", "Fixed Weight", "Lowest Priority",
                            "Minimum Cost", "Minimum History")
MANUSCRIPT_SUPP_VARIANTS = 7            # 4 ablations + 2 coefficient variants + default Adaptive
EXPECTED_RUNS = dict(main=4050, supplementary=1890)


def check_against_manuscript():
    """Stop early if simulation.py and the manuscript disagree on a fixed setting."""
    problems = []
    for name, expected in MANUSCRIPT_CONSTANTS.items():
        actual = getattr(sim, name, None)
        if actual != expected:
            problems.append(f"{name}: simulation.py={actual!r}, manuscript={expected!r}")
    if set(MAIN_POLICIES) != set(MANUSCRIPT_MAIN_POLICIES):
        problems.append(f"POLICIES: simulation.py={list(MAIN_POLICIES)}, "
                        f"manuscript={list(MANUSCRIPT_MAIN_POLICIES)}")
    if len(SUPPLEMENTARY_VARIANTS) != MANUSCRIPT_SUPP_VARIANTS or "Adaptive" not in SUPPLEMENTARY_VARIANTS:
        problems.append(f"SUPPLEMENTARY: expected 7 variants including 'Adaptive', "
                        f"got {list(SUPPLEMENTARY_VARIANTS)}")
    if problems:
        raise SystemExit("simulation.py disagrees with the manuscript:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------
# Task construction (fixed, deterministic order)
# --------------------------------------------------------------------------
def build_tasks(mode):
    """Return a list of (experiment, policy, n, R, C, seed) tuples for one experiment."""
    if mode == "test":
        n, R, C = TEST_CONFIG
        return [("test", pol, n, R, C, seed) for seed in TEST_SEEDS for pol in MAIN_POLICIES]
    if mode == "main":
        policies, seeds = MAIN_POLICIES, MAIN_SEEDS
    elif mode == "supplementary":
        policies, seeds = SUPPLEMENTARY_VARIANTS, SUPPLEMENTARY_SEEDS
    else:
        raise ValueError(mode)
    tasks = []
    for n, R, C in itertools.product(PROCESS_COUNTS, RESOURCE_COUNTS, CONTENTION_LEVELS):
        for seed in seeds:
            for pol in policies:        # all policies see the same (n, R, C, seed) -> matched workload
                tasks.append((mode, pol, n, R, C, seed))
    return tasks


def execute(task):
    """Run one simulation. Top-level function so multiprocessing can pickle it."""
    experiment, policy, n, R, C, seed = task
    result = sim.run(policy, n, R, C, seed)
    return experiment, result


# --------------------------------------------------------------------------
# Checks on the returned values (reported, never altered)
# --------------------------------------------------------------------------
def sanity_check(rows):
    problems = []
    for r in rows:
        tag = f"{r['policy']} n={r['n_processes']} R={r['n_resources']} C={r['contention']} seed={r['seed']}"
        if r["completed"] + r["terminated"] + r["unfinished"] != r["n_processes"]:
            problems.append(f"{tag}: completed+terminated+unfinished != n_processes")
        expected_cost = (r["work_lost"] + sim.RESTART_PENALTY * r["restarts"]
                         + sim.TERMINATION_PENALTY * r["terminated"])
        if r["recovery_cost"] != expected_cost:
            problems.append(f"{tag}: recovery_cost != work_lost + 2*restarts + 5*terminated")
        if not math.isclose(r["throughput"], r["completed"] / sim.HORIZON):
            problems.append(f"{tag}: throughput != completed/HORIZON")
        if r["recovery_time"] != r["restarts"] + r["terminated"]:
            problems.append(f"{tag}: recovery_time != restarts + terminated (events)")
    return problems


def determinism_check(task):
    """Run one task twice; the simulator has no hidden randomness, so results must be identical."""
    a = repr(execute(task)[1])
    b = repr(execute(task)[1])
    return a == b


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fieldnames):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def run_experiment(mode, workers, outdir, quiet=False):
    tasks = build_tasks(mode)
    if mode in EXPECTED_RUNS and len(tasks) != EXPECTED_RUNS[mode]:
        raise SystemExit(f"{mode}: built {len(tasks)} runs, manuscript says {EXPECTED_RUNS[mode]}")
    if not quiet:
        print(f"[{mode}] {len(tasks)} runs, workers={workers}")

    started = time.time()
    rows, step = [], max(1, len(tasks) // 20)
    if workers > 1:
        with Pool(workers) as pool:
            iterator = pool.imap(execute, tasks, chunksize=8)      # imap keeps task order
            for i, (experiment, result) in enumerate(iterator, 1):
                rows.append((experiment, result))
                if not quiet and i % step == 0:
                    print(f"  {i}/{len(tasks)}")
    else:
        for i, task in enumerate(tasks, 1):
            rows.append(execute(task))
            if not quiet and i % step == 0:
                print(f"  {i}/{len(tasks)}")

    # Columns: runner label, then every key returned by sim.run() in its own order,
    # then the simulation horizon (a constant of simulation.py, not a measured metric).
    metric_keys = list(rows[0][1].keys())
    fieldnames = ["experiment"] + metric_keys + ["horizon"]
    flat = []
    for experiment, result in rows:
        if list(result.keys()) != metric_keys:
            raise SystemExit("sim.run() returned inconsistent keys between runs")
        flat.append({"experiment": experiment, **result, "horizon": sim.HORIZON})

    # Keep generated files in the same folder layout consumed by the analysis scripts.
    mode_dir = os.path.join(outdir, mode) if mode in ("main", "supplementary") else os.path.join(outdir, "test")
    os.makedirs(mode_dir, exist_ok=True)
    csv_path = os.path.join(mode_dir, f"results_{mode}.csv")
    write_csv(csv_path, flat, fieldnames)

    problems = sanity_check(flat)
    meta = dict(
        mode=mode, rows=len(flat), csv=os.path.basename(csv_path), csv_sha256=sha256_of(csv_path),
        columns=fieldnames,
        design=dict(
            process_counts=PROCESS_COUNTS, resource_counts=RESOURCE_COUNTS,
            contention_levels=CONTENTION_LEVELS,
            seeds=(list(TEST_SEEDS) if mode == "test"
                   else list(MAIN_SEEDS if mode == "main" else SUPPLEMENTARY_SEEDS)),
            test_config=TEST_CONFIG if mode == "test" else None,
            policies=list(MAIN_POLICIES if mode in ("main", "test") else SUPPLEMENTARY_VARIANTS)),
        simulation_constants={k: getattr(sim, k) for k in MANUSCRIPT_CONSTANTS},
        simulation_py_sha256=sha256_of(sim.__file__),
        python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
        sanity_check_problems=problems,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        command=" ".join(sys.argv))
    with open(os.path.join(mode_dir, f"results_{mode}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not quiet:
        print(f"  wrote {csv_path} ({len(flat)} rows, {time.time() - started:.1f}s)")
        print("  sanity checks:", "OK" if not problems else f"{len(problems)} PROBLEM(S)")
        for p in problems[:10]:
            print("   -", p)
    return flat, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["test", "main", "supplementary", "all"], default="test",
                    help="test (default): 1 config x 2 seeds; main: 4,050 runs; "
                         "supplementary: 1,890 runs; all: main + supplementary")
    ap.add_argument("--outdir", default=SCRIPT_DIR,
                    help="base output folder; main/ and supplementary/ are created inside it (default: script folder)")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of processes (default 1). Output is identical for any value.")
    args = ap.parse_args()

    check_against_manuscript()

    modes = ["main", "supplementary"] if args.mode == "all" else [args.mode]
    if "test" in modes:
        n, R, C = TEST_CONFIG
        ok = determinism_check(("test", "Adaptive", n, R, C, TEST_SEEDS[0]))
        print("determinism check (same run twice gives identical output):", "OK" if ok else "FAILED")
    exit_code = 0
    for mode in modes:
        _, problems = run_experiment(mode, max(1, args.workers), args.outdir)
        if problems:
            exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
