import copy

import scripts.multi_walkforward_report as mwr


def test_smoke_determinism(monkeypatch):
    monkeypatch.setenv("CI", "true")
    r1 = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    r2 = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    a, b = copy.deepcopy(r1), copy.deepcopy(r2)
    for k in ("timestamp", "duration_s"):
        a.pop(k, None)
        b.pop(k, None)
    assert a == b
