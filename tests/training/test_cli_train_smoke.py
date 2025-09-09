"""
CLI smoke tests for actual training scripts
"""
import re
import subprocess


def run_train(max_steps: int = 50):
    """Run training with minimal steps"""
    cmd = [
        "python",
        "scripts/training/train_reward_based.py",
        "--symbol",
        "SPY",
        "--model-type",
        "ensemble",
        "--lookback-days",
        "100",
        "--min-trades",
        "10",
        "--verbose",
    ]
    out = subprocess.check_output(cmd, text=True)
    # assert required training completion indicators exist
    assert "TRAINING COMPLETED SUCCESSFULLY!" in out
    assert "Training Time:" in out
    assert "Validation Reward:" in out
    # Extract training time for validation
    m = re.search(r"Training Time: ([0-9.]+) seconds", out)
    assert m
    return float(m.group(1)), out


def test_cli_smoke():
    """Test that CLI training runs without errors"""
    training_time, out = run_train(max_steps=60)
    assert training_time > 0  # Training should take some time
    assert training_time < 300  # But not too long for smoke test
