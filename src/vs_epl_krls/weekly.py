"""Weekly calendar alignment without fabricated observations."""

from __future__ import annotations

import pandas as pd


def weekly_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert missing weeks as NaN; reject duplicates and off-calendar dates."""
    data = frame.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(data["date"], errors="raise"))
    if dates.empty or dates.hasnans or not dates.is_unique:
        raise ValueError("weekly dates must be nonempty, valid and unique")
    grid = pd.date_range(dates.min(), dates.max(), freq="7D")
    if not dates.isin(grid).all():
        raise ValueError("dates must belong to the same weekly calendar")
    data["date"] = dates
    return data.set_index("date").reindex(grid.rename("date")).reset_index()
