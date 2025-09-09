"""
LR scheduler order and values tests
"""
from torch import nn, optim


def test_scheduler_after_optimizer_step():
    """Test that scheduler is called after optimizer step"""
    m = nn.Linear(4, 1)
    opt = optim.SGD(m.parameters(), lr=0.1)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    lrs = []
    for _ in range(3):
        opt.step()
        sched.step()  # correct order
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs == [0.05, 0.025, 0.0125]
