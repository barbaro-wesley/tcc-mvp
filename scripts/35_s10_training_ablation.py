"""Compare only the two input scalers on frozen development folds."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.selection import (
    S10Candidate, S10_HOLDOUT_START, build_s10_supervised,
    evaluate_temporal_fold, pinned_validation_folds,
)


def run(args: argparse.Namespace) -> dict:
    frame = pd.read_csv(args.panel, parse_dates=["date"])
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frozen = S10Candidate(**manifest["champion_selected_without_holdout"])
    data = build_s10_supervised(frame, horizon=1, feature_set=frozen.feature_set)
    folds = pinned_validation_folds(data.target_dates).folds
    assert all(data.target_dates[f.validation_end-1] < np.datetime64(S10_HOLDOUT_START) for f in folds)
    rows = []
    summary = []
    for scaling in ("minmax", "robust_bounded"):
        candidate = replace(frozen, candidate_id=f"frozen_parameters_{scaling}", feature_scaling=scaling)
        results = [evaluate_temporal_fold(candidate, data, fold) for fold in folds]
        rows.extend(r.summary_row() for r in results)
        actual = np.concatenate([r.actual for r in results])
        prediction = np.concatenate([r.predictions for r in results])
        naive = np.concatenate([r.naive for r in results])
        summary.append({
            "scaling": scaling,
            "candidate": asdict(candidate),
            "mae": float(np.abs(actual-prediction).mean()),
            "persistence_mae": float(np.abs(actual-naive).mean()),
            "rmse": float(np.sqrt(np.mean((actual-prediction)**2))),
            "n": len(actual),
            "worst_fold_mae_ratio": max(r.metrics["mae"] / r.naive_metrics["mae"] for r in results),
            "max_feature_clip_fraction": max(r.feature_clip_fraction for r in results),
        })
    payload = {
        "experiment": "scaler_ablation_calendar_v2", "holdout_evaluated": False,
        "promotion": False, "n_hyperparameter_searches": 0,
        "development_start": str(data.target_dates[folds[0].validation_start])[:10],
        "development_end": str(data.target_dates[folds[-1].validation_end-1])[:10],
        "development_fingerprint": hashlib.sha256(
            frame.loc[frame.date < pd.Timestamp(S10_HOLDOUT_START)].to_csv(index=False).encode()
        ).hexdigest(),
        "summary": summary, "folds": rows,
        "claim_boundary": "paired development comparison with identical parameters; not independent evidence of future accuracy",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "folds.csv", index=False)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data/processed/s10_causal_panel.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/vs_epl_krls/s10_selection/selection_manifest_h1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/vs_epl_krls/s10_training_v2")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
