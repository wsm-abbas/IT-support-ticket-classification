"""Prints the two result files side by side."""
import json

import pandas as pd

rows = []
for name in ["baseline", "transformer"]:
    try:
        with open(f"results/{name}.json") as f:
            rows.append(json.load(f))
    except FileNotFoundError:
        print(f"results/{name}.json not found, run that script first")

if rows:
    print(pd.DataFrame(rows).to_string(index=False))
