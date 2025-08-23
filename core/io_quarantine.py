from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.schemas.io_contracts import PriceFrame

if TYPE_CHECKING:
    import pandas as pd


def validate_or_quarantine(
    df: pd.DataFrame, name: str, path: str = "quarantine/"
) -> tuple[bool, Path | None]:
    try:
        PriceFrame.validate_df(df)
        return True, None
    except Exception as e:
        outdir = Path(path) / datetime.utcnow().strftime("%Y-%m-%d")
        outdir.mkdir(parents=True, exist_ok=True)
        base = outdir / f"{datetime.utcnow().strftime('%H%M%S')}_{name}"
        df.head(50).to_csv(str(base) + "_head.csv")
        df.tail(50).to_csv(str(base) + "_tail.csv")
        (base.with_suffix(".json")).write_text(json.dumps({"error": str(e)}, indent=2))
        return False, base
