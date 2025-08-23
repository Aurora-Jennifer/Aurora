import json
from pathlib import Path


def test_search_history_has_fold_arrays():
    path = Path("reports/experiments/search_history.json")
    if not path.exists():
        # Skip if not generated in this run
        return
    h = json.loads(path.read_text())
    b = h.get("baseline", {}).get("ic_per_fold", [])
    s = h.get("best", {}).get("ic_per_fold", [])
    assert isinstance(b, list) and isinstance(s, list)
    assert len(b) == len(s) and len(b) > 0


def test_es_policy_train_tail():
    path = Path("reports/experiments/search_history.json")
    if not path.exists():
        return
    h = json.loads(path.read_text())
    policy = h.get("meta", {}).get("early_stopping", {}).get("policy")
    assert policy == "train_tail"


