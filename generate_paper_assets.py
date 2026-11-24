"""Regenerate manuscript Table 5 and Figures 4-5 from the included run-level CSV files."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
main = pd.read_csv(ROOT / "main" / "results_main.csv")

# Table 5: means by policy and process count.
table5 = main.groupby(["policy", "n_processes"])[["recovery_cost", "terminated"]].mean().reset_index()
table5.to_csv(ROOT / "table5_process_count.csv", index=False, float_format="%.6f")

# Figure 4: mean recovery cost by contention and policy.
fig4 = main.groupby(["contention", "policy"], as_index=False)["recovery_cost"].mean()
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for policy, group in fig4.groupby("policy"):
    group = group.sort_values("contention")
    ax.plot(group["contention"], group["recovery_cost"], marker="o", label=policy)
ax.set_title("Mean recovery cost across contention levels")
ax.set_xlabel("Contention probability")
ax.set_ylabel("Mean recovery cost (model units)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "Figure4_Recovery_Cost_by_Contention.png", dpi=300)
plt.close(fig)

# Figure 5: mean run-level P95 turnaround by policy.
fig5 = main.groupby("policy", as_index=False)["p95_turnaround"].mean()
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.bar(fig5["policy"], fig5["p95_turnaround"])
ax.set_title("Mean 95th-percentile turnaround by policy")
ax.set_ylabel("Mean run-level 95th-percentile turnaround")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(ROOT / "Figure5_P95_Turnaround_by_Policy.png", dpi=300)
plt.close(fig)

print("Wrote table5_process_count.csv, Figure4_Recovery_Cost_by_Contention.png, and Figure5_P95_Turnaround_by_Policy.png")
