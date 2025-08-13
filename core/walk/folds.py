from dataclasses import dataclass
from typing import Iterator


@dataclass
class Fold:
    train_lo: int
    train_hi: int
    test_lo: int
    test_hi: int
    fold_id: int


def gen_walkforward(
    n: int,
    train_len: int,
    test_len: int,
    stride: int,
    warmup: int = 0,
    anchored: bool = False,
) -> Iterator[Fold]:
    fold_id = 0
    t0 = warmup + (0 if anchored else train_len)
    while True:
        train_hi = t0 - 1
        train_lo = warmup if anchored else train_hi - train_len + 1
        test_lo = t0
        test_hi = min(test_lo + test_len - 1, n - 1)
        if test_lo >= n or train_lo < warmup:
            break
        yield Fold(train_lo, train_hi, test_lo, test_hi, fold_id)
        fold_id += 1
        t0 += stride
