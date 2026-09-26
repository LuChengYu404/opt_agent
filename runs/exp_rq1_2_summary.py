"""EXP-RQ1.2 oracle replay summary across 3 10-seed datasets."""
import json
from pathlib import Path

datasets = [
    ("h1",     "runs/exp_rq1_2_h1_report.json"),
    ("h3",     "runs/exp_rq1_2_h3_report.json"),
    ("hfull",  "runs/exp_rq1_2_hfull_report.json"),
]
families = ["branin", "hartmann", "ackley"]
baselines = ["behavior", "uniform", "stop"]

rows = []
for tag, path in datasets:
    r = json.loads(Path(path).read_text())
    row = {
        "dataset": tag,
        "n_records": r["n_records"],
        "gate2_pass": r["gate2"]["pass"],
        "gate3_pass": r["gate3"]["pass"],
        "gate4_pass": r["gate4"]["pass"],
        "g4_by_family": {},
    }
    for fam in families:
        if fam in r["gate4"]["summary"]:
            row["g4_by_family"][fam] = {
                b: {
                    "mean": r["gate4"]["summary"][fam]["comparisons"][b]["mean_gain"],
                    "ci_lo": r["gate4"]["summary"][fam]["comparisons"][b]["ci95"][0],
                    "ci_hi": r["gate4"]["summary"][fam]["comparisons"][b]["ci95"][1],
                    "pass_lo>0": r["gate4"]["summary"][fam]["comparisons"][b]["ci95"][0] > 0,
                }
                for b in baselines
            }
    rows.append(row)

print(json.dumps(rows, indent=2))
Path("runs/exp_rq1_2_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
