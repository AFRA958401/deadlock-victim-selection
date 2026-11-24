"""
run_age_baseline.py - supplementary age-based baseline ("Youngest First", wound-wait style).

Runs the Youngest First policy on the same 27 configurations and the same 30 seeds as the
main experiment (27 x 30 = 810 runs) and writes results_age_baseline.csv.
Usage:  python run_age_baseline.py
"""
import csv, itertools
from pathlib import Path
from multiprocessing import Pool
import simulation as sim

def task(args):
    n, R, C, seed = args
    row = sim.run("Youngest First", n, R, C, seed)
    row["experiment"], row["horizon"] = "age_baseline", sim.HORIZON
    return row

if __name__ == "__main__":
    jobs = list(itertools.product([20, 50, 100], [3, 5, 8], [0.2, 0.5, 0.8], range(30)))
    with Pool() as pool:
        rows = pool.map(task, jobs)
    outdir = Path(__file__).resolve().parent / "age_baseline"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "results_age_baseline.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(len(rows), "runs written to", output)
