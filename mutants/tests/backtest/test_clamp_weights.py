from utils.risk_clamp import clamp_weights


def test_clamp_weights_caps_per_pos_and_gross():
    w = {"A": 0.9, "B": 0.6, "C": -0.8}
    clamped, stats = clamp_weights(w, max_pos=0.5, max_gross=1.0)
    assert max(abs(v) for v in clamped.values()) <= 0.5 + 1e-9
    assert sum(abs(v) for v in clamped.values()) <= 1.0 + 1e-9
    assert stats["clamped"] is True
