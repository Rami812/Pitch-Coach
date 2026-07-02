"""
sync_model.py — copy the freshly trained model into the Vercel function.

The Vercel function (api/index.py) is fully self-contained: the XGBoost tree
dump is embedded as a Python string literal so the deployed function depends on
zero external files (this is what fixed the recurring bundling crashes).

After retraining (python train_xgboost.py regenerates model_trees.json), run:

    python sync_model.py

It replaces the embedded `_MODEL = _json.loads(...)` literal in api/index.py
with the current contents of model_trees.json, then verifies the file still
parses. Nothing else in api/index.py is touched.
"""

import json
import pathlib
import re

TREES = pathlib.Path("model_trees.json")
INDEX = pathlib.Path("api/index.py")


def main() -> None:
    if not TREES.exists():
        raise SystemExit("model_trees.json not found. Run `python train_xgboost.py` first.")

    trees_json = TREES.read_text().strip()
    json.loads(trees_json)  # validate it is well-formed JSON

    src = INDEX.read_text()
    pattern = re.compile(r"^_MODEL = _json\.loads\(.*\)$", re.M)
    if not pattern.search(src):
        raise SystemExit("Could not find the `_MODEL = _json.loads(...)` line in api/index.py")

    # repr() produces a correctly-escaped Python string literal on one line.
    new_line = "_MODEL = _json.loads(" + repr(trees_json) + ")"
    src = pattern.sub(lambda _m: new_line, src, count=1)

    # Make sure we did not break the module.
    compile(src, str(INDEX), "exec")

    INDEX.write_text(src)
    n_trees = len(json.loads(trees_json)["trees"])
    print(f"Synced {n_trees} trees into {INDEX}. Commit api/index.py and redeploy.")


if __name__ == "__main__":
    main()
