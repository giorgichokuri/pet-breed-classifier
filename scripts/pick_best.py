"""Print the name of the run with the highest validation accuracy."""
import json
from pathlib import Path

runs = []
for f in Path("experiments").glob("*/metrics.json"):
    if f.parent.name.startswith("smoke"):
        continue
    m = json.loads(f.read_text())
    runs.append((m["best_val_acc"], f.parent.name))
if not runs:
    raise SystemExit("no finished runs in experiments/")
print(max(runs)[1])
