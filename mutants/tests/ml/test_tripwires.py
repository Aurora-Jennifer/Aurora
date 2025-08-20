from ml.runtime import compute_turnover, detect_weight_spikes


def test_spike_tripwire():
    prev, now = {"SPY": 0.0}, {"SPY": 0.4}
    spikes = detect_weight_spikes(prev, now, max_delta=0.25)
    assert "SPY" in spikes


def test_turnover_cap():
    prev, now = {"A": 0.5, "B": 0.5}, {"A": -0.5, "B": -0.5}
    assert compute_turnover(prev, now) == 2.0
