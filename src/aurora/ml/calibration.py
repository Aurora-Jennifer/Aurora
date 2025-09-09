"""
Temperature scaling and decision thresholding for trading models
"""
from typing import Any

import torch
import torch.nn.functional as F

HOLD_ID = 2


class TempScaler(torch.nn.Module):
    """Temperature scaling module"""
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Parameter(torch.ones(()))
    
    def forward(self, logits):
        return logits / self.t.clamp_min(1e-3)


@torch.no_grad()
def fit_temperature(val_logits: torch.Tensor, val_y: torch.Tensor, 
                   iters: int = 100, lr: float = 0.01) -> float:
    """Fit temperature scaling on validation data"""
    T = torch.tensor([1.0], device=val_logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=iters, line_search_fn="strong_wolfe")
    
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / T.clamp_min(1e-3), val_y)
        loss.backward()
        return loss
    
    opt.step(closure)
    return float(T.detach().clamp(0.25, 4.0).item())


@torch.no_grad()
def grid_tau(val_logits: torch.Tensor, val_y: torch.Tensor, T: float = 1.0) -> float:
    """Find optimal tau threshold on validation data"""
    p = torch.softmax(val_logits / T, dim=-1)
    maxp, _ = p.max(dim=-1)
    taus = torch.linspace(0.33, 0.60, 28, device=p.device)
    best_tau, best_acc = 0.5, -1
    
    for tau in taus:
        pred = torch.where(maxp >= tau, p.argmax(-1), torch.full_like(val_y, HOLD_ID))
        acc = (pred == val_y).float().mean().item()
        if acc > best_acc:
            best_acc, best_tau = acc, float(tau.item())
    
    return best_tau


@torch.no_grad()
def decide_actions(logits: torch.Tensor, T: float, tau: float, 
                  costs_bps: float = 4.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make trading decisions with temperature scaling and thresholding"""
    p = torch.softmax(logits / T, dim=-1)
    maxp, arg = p.max(dim=-1)
    
    # Cost-aware margin: encourage HOLD when advantage is small
    adv = (p[:, 1] - p[:, 0]).abs()  # symmetric margin around BUY/SELL
    hold = (maxp < tau) | (adv < (costs_bps * 1e-4))
    
    a = arg.clone()
    a[hold] = HOLD_ID
    
    return a, p, maxp, adv


@torch.no_grad()
def selective_trading(actions: torch.Tensor, maxp: torch.Tensor, 
                     top_quantile: float = 0.80) -> torch.Tensor:
    """Apply selective trading - only trade on top confidence days"""
    q = torch.quantile(maxp, top_quantile)
    a_sel = actions.clone()
    a_sel[maxp < q] = HOLD_ID
    return a_sel


def log_decision_stats(actions: torch.Tensor, maxp: torch.Tensor, 
                      T: float, tau: float) -> dict[str, Any]:
    """Log decision statistics for debugging"""
    hold_rate = float((actions == HOLD_ID).float().mean())
    action_hist = {int(i): int((actions == i).sum()) for i in range(3)}
    
    return {
        "T": T,
        "tau": tau,
        "test_maxp_mean": float(maxp.mean()),
        "test_maxp_p50": float(maxp.median()),
        "test_maxp_p90": float(maxp.kthvalue(int(0.9 * len(maxp)))[0]),
        "hold_rate": hold_rate,
        "action_hist": action_hist
    }


@torch.no_grad()
def advantage_based_decisions(q_net, v_net, states: torch.Tensor, 
                            costs_bps: float = 4.0, tau_margin: float = 0.001,
                            trade_topk: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Make decisions based on Q-V advantage with cost awareness"""
    q = q_net(states)              # [B, 3]
    v = v_net(states).squeeze(-1)  # [B]
    adv = q - v[:, None]           # [B, 3]
    
    # Subtract explicit trading costs
    cost = costs_bps * 1e-4
    adv_adj = adv.clone()
    adv_adj[:, 0] -= cost  # BUY cost
    adv_adj[:, 1] -= cost  # SELL cost
    # HOLD (index 2) stays as-is (no cost)
    
    # Find best action and margin vs HOLD
    best = adv_adj.argmax(dim=1)
    margin = adv_adj.gather(1, best[:, None]).squeeze(1) - adv_adj[:, 2]  # vs HOLD
    
    # Apply threshold
    action = torch.where(margin > tau_margin, best, torch.full_like(best, HOLD_ID))
    
    # Apply top-k selection if specified
    if trade_topk is not None and trade_topk > 0:
        # For multi-symbol, apply top-k per timestep
        if states.shape[0] > trade_topk:
            # Get top-k margins across the batch
            topk_indices = torch.topk(margin, k=min(trade_topk, states.shape[0])).indices
            action = torch.full_like(action, HOLD_ID)
            action[topk_indices] = best[topk_indices]
    
    # Calculate statistics
    hold_rate = float((action == HOLD_ID).float().mean())
    action_hist = {int(i): int((action == i).sum()) for i in range(3)}
    
    stats = {
        "tau_margin": tau_margin,
        "costs_bps": costs_bps,
        "trade_topk": trade_topk,
        "hold_rate": hold_rate,
        "action_hist": action_hist,
        "margin_mean": float(margin.mean()),
        "margin_p50": float(margin.median()),
        "margin_p90": float(torch.quantile(margin, 0.90)),
        "margin_std": float(margin.std()),
    }
    
    return action, margin, stats


def tune_tau_margin(val_states: torch.Tensor, val_prices: torch.Tensor,
                   q_net, v_net, costs_bps: float = 4.0) -> float:
    """Tune tau margin on validation to maximize Sharpe after costs"""
    # Generate candidate tau values
    with torch.no_grad():
        q = q_net(val_states)
        v = v_net(val_states).squeeze(-1)
        adv = q - v[:, None]
        cost = costs_bps * 1e-4
        adv_adj = adv.clone()
        adv_adj[:, 0] -= cost
        adv_adj[:, 1] -= cost
        best = adv_adj.argmax(dim=1)
        margins = adv_adj.gather(1, best[:, None]).squeeze(1) - adv_adj[:, 2]
    
    # Use percentiles of margin distribution as candidate thresholds
    taus = torch.quantile(torch.abs(margins), torch.linspace(0.5, 0.95, 20))
    
    best_tau, best_sharpe = 0.001, -1e9
    
    for tau in taus:
        # Simple Sharpe proxy: mean margin / std margin
        actions = torch.where(margins > tau, best, torch.full_like(best, HOLD_ID))
        trade_mask = actions != HOLD_ID
        
        if trade_mask.sum() > 0:
            trade_margins = margins[trade_mask]
            sharpe_proxy = float(trade_margins.mean() / (trade_margins.std() + 1e-8))
            
            if sharpe_proxy > best_sharpe:
                best_sharpe = sharpe_proxy
                best_tau = float(tau)
    
    return best_tau


class CalibratedPredictor:
    """Calibrated predictor with temperature scaling and thresholding"""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.T = 1.0
        self.tau = 0.5
        self.calibrated = False
    
    def calibrate(self, val_logits: torch.Tensor, val_y: torch.Tensor):
        """Calibrate temperature and threshold on validation data"""
        self.T = fit_temperature(val_logits, val_y)
        self.tau = grid_tau(val_logits, val_y, self.T)
        self.calibrated = True
        print(f"Calibration complete: T={self.T:.3f}, tau={self.tau:.3f}")
    
    def predict(self, logits: torch.Tensor, costs_bps: float = 4.0, 
                selective: bool = False, top_quantile: float = 0.80) -> tuple[torch.Tensor, dict[str, Any]]:
        """Make calibrated predictions"""
        if not self.calibrated:
            raise ValueError("Must calibrate before predicting")
        
        actions, p, maxp, adv = decide_actions(logits, self.T, self.tau, costs_bps)
        
        if selective:
            actions = selective_trading(actions, maxp, top_quantile)
        
        stats = log_decision_stats(actions, maxp, self.T, self.tau)
        return actions, stats
