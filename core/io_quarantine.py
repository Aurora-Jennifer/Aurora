from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from core.schemas.io_contracts import PriceFrame

def validate_or_quarantine(df: pd.DataFrame, name: str, path: str = "quarantine/") -> tuple[bool, Path | None]:
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


