# Conventions — Return Types & Invariance

Default: percent returns (C_t / C_{t-1} - 1)
- Scale-invariant (multiplying prices by a constant does not change returns)
- Not additive-invariant (adding a constant shifts returns slightly)

Alternatives:
- log: log(C_t / C_{t-1}) — scale-invariant; not additive-invariant
- diff: C_t - C_{t-1} — additive-invariant; not scale-invariant

Testing policy:
- For percent/log: assert strict equality under scale; assert high correlation under additive shifts
- For diff: assert strict equality under additive shifts

Configuration:
```yaml
data:
  returns:
    type: percent  # percent | log | diff
```

When to use which return type:

| Type    | Use when                                     | Guarantees                          |
|---------|-----------------------------------------------|-------------------------------------|
| percent | Standard modeling, scale-invariant features   | Scale-invariant; not additive       |
| log     | Multiplicative models, small-return regimes   | Scale-invariant; not additive       |
| diff    | Spread/price-change strategies                | Additive-invariant; not scale       |

TODO: Expand diff-return property tests and add gating when configured.
