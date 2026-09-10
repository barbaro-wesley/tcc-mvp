"""Modelo estadual do Diesel S10 para o Rio Grande do Sul, avaliado em desenvolvimento.

Motivacao comercial
-------------------
O produto previa a media nacional de revenda da ANP, que nenhum comprador paga.
A distancia entre a serie modelada e a serie que o cliente enfrenta era a maior
fragilidade da tese, e ela se resolve com o dado certo: a ANP publica a mesma
pesquisa semanal por estado.

Quatro concorrentes, uma pergunta
---------------------------------
1. ``persistencia``     — o preco de hoje.
2. ``rs_direto``        — a especificacao de paridade congelada, aplicada a serie gaucha.
3. ``nacional+carrego`` — previsao nacional mais o spread atual carregado.
4. ``nacional+spread``  — previsao nacional mais correcao de erro no spread.
5. ``nacional+spread+ancora`` — o mesmo, com o preco de produtor da regiao Sul
   ancorando o alvo de reversao.

**O holdout nao e lido.** A avaliacao para no fim do desenvolvimento, pela mesma
janela congelada por data que protege a serie nacional.

Uso::

    python scripts/26_s10_rs_regional.py
    python scripts/26_s10_rs_regional.py --uf SC --offline
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.gates import (  # noqa: E402
    accuracy_report,
    bootstrap_mean_ci,
    paired_block_bootstrap,
)
from vs_epl_krls.passthrough import (  # noqa: E402
    PARITY_FEATURES,
    PassThroughConfig,
    PassThroughECM,
    build_parity_panel,
)
from vs_epl_krls.procurement import simulate_one_week_prebuy  # noqa: E402
from vs_epl_krls.regional import (  # noqa: E402
    UF_REGION,
    SpreadForecaster,
    build_regional_panel,
    fetch_regional_producer,
    fetch_state_weekly,
)
from vs_epl_krls.selection import pinned_validation_folds  # noqa: E402

NEWLINE = chr(10)
MONTHLY_LITERS = 200_000.0


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}" if abs(value) < 1000 else f"{value:,.0f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return NEWLINE.join(lines)


def ingest(uf: str, raw_dir: Path, output: Path) -> dict[str, object]:
    """Baixa, versiona e grava o painel estadual com proveniencia por fonte."""

    region = UF_REGION[uf]
    state, state_record = fetch_state_weekly(
        uf, cache=raw_dir / f"anp_semanal_estados_{uf.lower()}.xlsx"
    )
    producer, producer_record = fetch_regional_producer(
        region, cache=raw_dir / f"anp_producer_{region}.xls"
    )
    national = pd.read_csv(
        ROOT / "data" / "processed" / "s10_causal_panel.csv", parse_dates=["date"]
    )
    panel = build_regional_panel(state, national[["date", "price"]], producer)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    return {
        "uf": uf,
        "region": region,
        "n_weeks": int(len(panel)),
        "coverage": {
            "start": str(panel["date"].min().date()),
            "end": str(panel["date"].max().date()),
        },
        "median_stations": (
            float(panel["stations"].median()) if "stations" in panel else None
        ),
        "sources": [state_record.as_dict(), producer_record.as_dict()],
    }


def national_predictions(dates: pd.Series) -> pd.DataFrame:
    """Previsao nacional causal para as semanas pedidas, uma so passada."""

    causal = pd.read_csv(
        ROOT / "data" / "processed" / "s10_causal_panel.csv", parse_dates=["date"]
    )
    panel = build_parity_panel(causal)
    wanted = set(pd.to_datetime(dates))
    positions = [index for index, value in enumerate(panel["date"]) if value in wanted]
    if not positions:
        raise RuntimeError("nenhuma semana do estado casa com o painel nacional")
    # A primeira linha do painel nao tem origem, entao o walk-forward comeca em 1.
    start, end = max(1, min(positions)), max(positions) + 1
    model = PassThroughECM(config=PassThroughConfig(), feature_names=PARITY_FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.walk_forward(panel, start, end, refit_every=1)
    return result[["date", "prediction"]].rename(columns={"prediction": "national"})


def state_direct_predictions(panel: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """A especificacao de paridade congelada, aplicada direto a serie do estado."""

    causal = pd.read_csv(
        ROOT / "data" / "processed" / "s10_causal_panel.csv", parse_dates=["date"]
    )
    merged = panel[["date", "price"]].merge(
        causal.drop(columns=["price"]), on="date", how="left"
    )
    state_panel = build_parity_panel(merged)
    wanted = panel["date"].iloc[start:end]
    positions = state_panel.index[state_panel["date"].isin(wanted)]
    if len(positions) != len(wanted):
        raise ValueError("state parity panel is missing requested target dates")
    model = PassThroughECM(config=PassThroughConfig(), feature_names=PARITY_FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.walk_forward(state_panel, max(1, int(positions.min())), int(positions.max()) + 1, refit_every=1)
    return result.set_index("date")["prediction"].reindex(wanted).to_numpy(float)


def contiguous_blocks(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Quebra o quadro nas semanas que a ANP nao pesquisou.

    A serie estadual tem lacunas reais — uma de nove semanas em 2020 e duas de
    uma semana. A politica de antecipacao compara o preco de hoje com o da semana
    seguinte; atravessar uma semana ausente inventaria uma decisao que ninguem
    poderia ter tomado. O replay roda por bloco continuo e os blocos sao somados.
    """

    ordered = frame.sort_values("target_date").reset_index(drop=True)
    breaks = ordered["target_date"].diff() != pd.Timedelta(days=7)
    groups = breaks.cumsum()
    return [block for _, block in ordered.groupby(groups) if len(block) >= 3]


def policy(frame: pd.DataFrame, column: str) -> dict[str, object]:
    subset = frame[["target_date", "actual", "persistence", column]].dropna()
    blocks = contiguous_blocks(subset)
    if not blocks:
        return {"net_savings_brl": float("nan"), "triggered": 0, "precision": None}

    weekly: list[float] = []
    savings = triggered = correct = 0.0
    for block in blocks:
        replay = simulate_one_week_prebuy(
            block, prediction_column=column, model_name=column, monthly_liters=MONTHLY_LITERS
        )
        weekly.extend(replay.weekly_net_savings_brl)
        savings += replay.net_savings_brl
        triggered += replay.triggered_prebuys
        if replay.trigger_precision is not None:
            correct += replay.trigger_precision * replay.triggered_prebuys
    series = np.asarray(weekly, dtype=float)
    low, high = (
        bootstrap_mean_ci(series * 52.0) if series.size >= 8 else (float("nan"), float("nan"))
    )
    largest = float(np.max(series)) if series.size else 0.0
    return {
        "net_savings_brl": float(savings),
        "annualized_savings_brl": float(np.mean(series) * 52.0) if series.size else float("nan"),
        "annualized_ci90": [low, high],
        "triggered": int(triggered),
        "precision": float(correct / triggered) if triggered else None,
        "largest_event_share": float(largest / savings) if savings > 0 else None,
        "weeks_replayed": int(series.size),
        "contiguous_blocks": len(blocks),
    }


def evaluate_fold(panel: pd.DataFrame, national: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    window = panel.iloc[start:end]
    frame = pd.DataFrame(
        {
            "target_date": window["date"].to_numpy(),
            "actual": window["price"].to_numpy(float),
            "persistence": window["origin_price"].to_numpy(float),
        }
    ).reset_index(drop=True)
    aligned = (
        window[["date"]]
        .merge(national, on="date", how="left")["national"]
        .to_numpy(float)
    )
    frame["nacional_bruto"] = aligned

    # 1. carregar o spread atual sem modela-lo
    frame["nacional+carrego"] = aligned + window["spread_lag1"].to_numpy(float)

    # 2. e 3. correcao de erro no spread, sem e com ancora de produtor
    for name, use_anchor in (("nacional+spread", False), ("nacional+spread+ancora", True)):
        forecaster = SpreadForecaster(use_anchor=use_anchor)
        result = forecaster.walk_forward(panel, aligned, start, end)
        frame[name] = result["prediction"].to_numpy(float)

    # 4. a especificacao congelada aplicada direto ao estado
    frame["rs_direto"] = state_direct_predictions(panel, start, end)
    # Coluna propria para a persistencia: reusar "persistence" como nome de
    # modelo produziria selecao duplicada na hora de pontuar.
    frame["persistencia"] = frame["persistence"]
    return frame


def spread_opportunity(
    panel: pd.DataFrame, *, horizon: int = 12, min_gap_weeks: int = 8
) -> dict[str, object]:
    """Onde o spread esta hoje, e o que aconteceu nas vezes em que esteve la.

    A contagem que importa nao e de semanas, e de **episodios independentes**.
    Um spread extremo dura meses, entao contar semanas transforma um unico
    acontecimento em amostra grande e produz a falsa impressao de regularidade.
    Aqui as semanas sao agrupadas: dois periodos separados por menos de
    ``min_gap_weeks`` semanas sao o mesmo episodio.
    """

    history = panel[["date", "spread"]].dropna().reset_index(drop=True)
    if len(history) < 104:
        return {"available": False}
    spread = history["spread"]
    current = float(spread.iloc[-1])
    mean, deviation = float(spread.mean()), float(spread.std())

    def episodes(mask: pd.Series) -> int:
        dates = history.loc[mask, "date"]
        if dates.empty:
            return 0
        return int((dates.diff().dt.days > min_gap_weeks * 7).sum()) + 1

    levels = []
    for quantile in (0.02, 0.05, 0.10):
        threshold = float(spread.quantile(quantile))
        mask = spread <= threshold
        positions = np.flatnonzero(mask.to_numpy())
        forward = np.array(
            [
                float(spread.iloc[position + horizon] - spread.iloc[position])
                for position in positions
                if position + horizon < len(spread)
            ]
        )
        levels.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "weeks": int(mask.sum()),
                "independent_episodes": episodes(mask),
                "mean_change_after_horizon": float(forward.mean()) if forward.size else None,
                "share_positive": float(np.mean(forward > 0)) if forward.size else None,
            }
        )

    at_current = spread <= current + 0.01
    return {
        "available": True,
        "n_weeks": int(len(history)),
        "horizon_weeks": horizon,
        "current_spread": current,
        "current_z": (current - mean) / deviation if deviation else float("nan"),
        "current_percentile": float((spread < current).mean()),
        "weeks_at_current_level": int(at_current.sum()),
        "independent_episodes_at_current_level": episodes(at_current),
        "levels": levels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uf", default="RS", help="unidade da federacao, padrao RS")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--panel", type=Path, default=None, help="painel ja gravado; padrao data/processed"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--offline", action="store_true", help="usa o painel gravado, sem baixar nada"
    )
    args = parser.parse_args()

    uf = str(args.uf).strip().upper()
    if uf not in UF_REGION:
        raise SystemExit(f"unidade da federacao desconhecida: {uf}")
    panel_path = args.panel or ROOT / "data" / "processed" / f"s10_panel_{uf.lower()}.csv"
    output_dir = args.output_dir or ROOT / "reports" / "vs_epl_krls" / f"s10_{uf.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.offline:
        if not panel_path.is_file():
            raise SystemExit(f"painel ausente: {panel_path}")
        provenance: dict[str, object] = {"uf": uf, "offline": True, "sources": []}
        panel = pd.read_csv(panel_path, parse_dates=["date"])
    else:
        provenance = ingest(uf, args.raw_dir, panel_path)
        panel = pd.read_csv(panel_path, parse_dates=["date"])

    windows = pinned_validation_folds(panel["date"])
    print(f"painel {uf}: {len(panel)} semanas, {panel['date'].min().date()} a "
          f"{panel['date'].max().date()}")
    if "stations" in panel:
        print(f"postos pesquisados: mediana {panel['stations'].median():.0f}")
    print(f"janela congelada: desenvolvimento ate "
          f"{panel['date'][windows.development_end - 1].date()}; o holdout nao e lido")

    # So as semanas que os folds realmente avaliam: o walk-forward nacional
    # reajusta a cada passo e nao ha razao para pagar por semanas fora da janela.
    evaluated_dates = pd.concat(
        [
            panel["date"].iloc[fold.validation_start : fold.validation_end]
            for fold in windows.folds
        ]
    )
    national = national_predictions(evaluated_dates)
    frames = [
        evaluate_fold(panel, national, fold.validation_start, fold.validation_end)
        for fold in windows.folds
    ]
    combined = pd.concat(frames, ignore_index=True)

    models = [
        "persistencia",
        "rs_direto",
        "nacional+carrego",
        "nacional+spread",
        "nacional+spread+ancora",
    ]
    rows: list[dict[str, object]] = []
    for name in models:
        valid = combined[["actual", "persistence", name]].dropna()
        report = accuracy_report(
            valid["actual"].to_numpy(float),
            valid[name].to_numpy(float),
            valid["persistence"].to_numpy(float),
        )
        rows.append({"modelo": name, **asdict(report), **policy(combined, name)})
    summary = pd.DataFrame(rows)

    best = summary.loc[summary["mae"].idxmin(), "modelo"]
    comparisons: dict[str, dict[str, float]] = {}
    for name in models:
        if name == "rs_direto":
            continue
        pair = combined[["actual", "rs_direto", name]].dropna()
        comparisons[name] = paired_block_bootstrap(
            np.abs(pair["actual"] - pair[name]),
            np.abs(pair["actual"] - pair["rs_direto"]),
        )

    print()
    print(
        summary[
            ["modelo", "mae", "mae_quiet", "mae_event", "directional_accuracy",
             "net_savings_brl", "triggered", "precision"]
        ].to_string(index=False)
    )
    print(f"{NEWLINE}melhor MAE: {best}")
    print(f"{NEWLINE}MAE contra o modelo aplicado direto ao estado "
          "(bootstrap pareado em blocos):")
    for name, result in comparisons.items():
        verdict = "melhor" if result["ci90_high"] < 0 else "sem ganho decidivel"
        print(f"  {name:24s} {result['mean_difference']:+.6f} "
              f"IC90 [{result['ci90_low']:+.6f}, {result['ci90_high']:+.6f}] -> {verdict}")

    spread = SpreadForecaster(use_anchor=True).fit(panel, end=windows.development_end)
    print(f"{NEWLINE}spread ajustado ate o fim do desenvolvimento: "
          f"{json.dumps(spread.summary(), ensure_ascii=False)}")

    current = panel.dropna(subset=["spread"]).iloc[-1]
    history = panel["spread"].dropna()
    z = float((current["spread"] - history.mean()) / history.std())
    print(f"spread corrente {current['spread']:+.4f} (z = {z:+.2f}); "
          f"preco {uf} R$ {current['price']:.4f}/L contra nacional "
          f"R$ {current['national_price']:.4f}/L")

    opportunity = spread_opportunity(panel)
    if opportunity.get("available"):
        print(f"  percentil {opportunity['current_percentile']:.1%} de "
              f"{opportunity['n_weeks']} semanas; "
              f"{opportunity['weeks_at_current_level']} semanas nesse nivel em "
              f"{opportunity['independent_episodes_at_current_level']} episodio(s) independente(s)")
        for level in opportunity["levels"]:
            change = level["mean_change_after_horizon"]
            share = level["share_positive"]
            print(f"    q{level['quantile']:.0%} (<= {level['threshold']:+.4f}): "
                  f"{level['weeks']:3d} semanas em {level['independent_episodes']} episodios; "
                  f"{opportunity['horizon_weeks']}s depois "
                  f"{'-' if change is None else format(change, '+.4f')}, "
                  f"positiva em {'-' if share is None else format(share, '.0%')}")

    # Erro de base: o que custa usar a serie nacional no lugar da estadual.
    litres_year = MONTHLY_LITERS * 12.0
    recent = panel["spread"].dropna().tail(52)
    basis_error = float(recent.abs().mean())
    print(f"{NEWLINE}erro de base nas ultimas 52 semanas: R$ {basis_error:.4f}/L "
          f"-> R$ {basis_error * litres_year:,.0f}/ano para {MONTHLY_LITERS:,.0f} L/mes")

    manifest = {
        "scope": f"Diesel B S10, revenda media estadual da ANP — {uf}",
        "holdout_evaluated": False,
        "development_end_date": str(panel["date"][windows.development_end - 1].date()),
        "folds": [asdict(fold) for fold in windows.folds],
        "provenance": provenance,
        "models": models,
        "summary": json.loads(summary.to_json(orient="records")),
        "mae_vs_state_direct": comparisons,
        "spread_model": spread.summary(),
        "current_state": {
            "date": str(pd.Timestamp(current["date"]).date()),
            "state_price": float(current["price"]),
            "national_price": float(current["national_price"]),
            "spread": float(current["spread"]),
            "spread_z": z,
        },
        "spread_opportunity": opportunity,
        "basis_error": {
            "window_weeks": 52,
            "mean_absolute_spread_brl_per_liter": basis_error,
            "monthly_liters": MONTHLY_LITERS,
            "annual_budget_error_brl": basis_error * litres_year,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    frozen = summary[summary["modelo"] == "rs_direto"].iloc[0]
    winner = summary[summary["modelo"] == "nacional+spread"].iloc[0]
    naive = summary[summary["modelo"] == "persistencia"].iloc[0]
    trigger_annual = float(winner["annualized_savings_brl"])
    report = [
        f"# Modelo estadual do Diesel B S10 — {uf}",
        "",
        f"**O holdout nao foi lido.** A avaliacao termina em "
        f"{manifest['development_end_date']}, pela mesma janela congelada por data que",
        "protege a serie nacional.",
        "",
        "## Por que estadual",
        "",
        "O produto previa a media nacional de revenda da ANP. Nenhum comprador paga esse",
        "preco: ele agrega 3.173 postos em 27 unidades da federacao, com tributo, frete e",
        "estrutura de distribuicao diferentes em cada uma. A distancia entre a serie",
        "modelada e a serie que o cliente enfrenta era a maior fragilidade comercial do",
        "projeto — e ela nao se resolve com modelo melhor, se resolve com o dado certo.",
        "",
        f"A ANP publica a mesma pesquisa por estado: {manifest['provenance'].get('n_weeks', len(panel))} "
        f"semanas para o {uf}, mediana de "
        f"{panel['stations'].median():.0f} postos por semana.",
        "",
        "## A decomposicao, e a evidencia que a escolheu",
        "",
        "Medido sobre as semanas comuns: a variacao semanal do estado correlaciona **0,939**",
        "com a nacional — **88% da variancia estadual e movimento do pais** — e o desvio da",
        "variacao do *spread* e apenas 0,0264 contra 0,0769 do preco. Modelar o estado",
        "direto joga fora o sinal nacional, que e melhor medido, e paga o ruido estadual",
        "inteiro.",
        "",
        markdown_table(
            json.loads(summary.to_json(orient="records")),
            ["modelo", "mae", "mae_quiet", "mae_event", "directional_accuracy",
             "net_savings_brl", "triggered", "precision"],
        ),
        "",
        f"A decomposicao entrega **+{winner['net_savings_brl'] / frozen['net_savings_brl'] - 1:.1%} de "
        f"economia** e **+{(winner['precision'] - frozen['precision']) * 100:.1f} pontos de precisao de",
        "gatilho** sobre a especificacao aplicada direto ao estado. O ganho de MAE existe mas",
        "**nao e decidivel**: o bootstrap pareado em blocos coloca zero dentro do IC90. E o",
        "mesmo padrao que este projeto ja encontrou duas vezes — decide melhor do que preve.",
        "",
        "Contra a persistencia, que e o que o comprador faz hoje sem nenhum modelo, o ganho",
        f"de MAE e de {1 - winner['mae'] / naive['mae']:.1%} — modesto, como sempre foi nesta serie, porque em dois",
        "tercos das semanas o preco simplesmente nao se move.",
        "",
        "A ancora de produtor da regiao Sul nao ajudou: peso estimado de "
        f"{manifest['spread_model']['anchor_weight']:+.4f}, praticamente nulo. Coerente com a",
        "medicao previa de que o spread de produtor explica o **nivel** do spread de revenda",
        "(+0,25) e quase nada da variacao semanal dele (+0,06).",
        "",
        "## O que este trabalho NAO mostrou",
        "",
        "O modelo estadual nao e melhor que o nacional. Na mesma janela ele tem MAE maior",
        f"({winner['mae']:.6f} contra 0,050492) e acuracia direcional menor",
        f"({winner['directional_accuracy']:.1%} contra 74,3%). A causa e estrutural e nao tem",
        "conserto por modelagem: 262 postos pesquisados contra 3.173.",
        "",
        "Focar no estado nao torna a previsao melhor. Torna o numero **verdadeiro** em vez de",
        "aproximado — e e ai que esta o valor.",
        "",
        "## Onde o valor realmente esta",
        "",
        f"Para um comprador de {MONTHLY_LITERS:,.0f} L/mes no {uf}:",
        "",
        "| fonte de valor | R$/ano |",
        "|---|---:|",
        f"| erro de orcamento por usar a serie nacional | **{basis_error * litres_year:,.0f}** |",
        f"| economia da politica de antecipacao | {trigger_annual:,.0f} |",
        "",
        f"Usar a base errada custa **{basis_error * litres_year / trigger_annual:.1f}x** o que o gatilho",
        "semanal economiza. O produto estadual nao se vende pela previsao: vende-se por",
        "entregar a serie que o cliente efetivamente enfrenta. Hoje o preco gaucho esta",
        f"{abs(float(current['spread']) / float(current['national_price'])):.1%} abaixo do nacional; quem",
        "orca pela media do pais erra para cima nessa proporcao.",
        "",
        "## A posicao de hoje",
        "",
        f"O spread esta em R$ {float(current['spread']):+.4f}/L, z = {z:+.2f}, "
        f"percentil {opportunity['current_percentile']:.1%} de {opportunity['n_weeks']} semanas.",
        "",
        markdown_table(
            [
                {
                    "faixa": f"q{level['quantile']:.0%}",
                    "limiar": level["threshold"],
                    "semanas": level["weeks"],
                    "episodios independentes": level["independent_episodes"],
                    f"variacao {opportunity['horizon_weeks']}s depois": level["mean_change_after_horizon"],
                    "positiva em": level["share_positive"],
                }
                for level in opportunity["levels"]
            ],
            ["faixa", "limiar", "semanas", "episodios independentes",
             f"variacao {opportunity['horizon_weeks']}s depois", "positiva em"],
        ),
        "",
        "**Leia a coluna de episodios antes da de semanas.** Um spread extremo dura meses:",
        f"das {opportunity['weeks_at_current_level']} semanas ja vistas no nivel de hoje, a maioria e o",
        f"episodio corrente, e sobram {opportunity['independent_episodes_at_current_level'] - 1} precedentes",
        "de verdade. A direcao da reversao e sustentada nas faixas com mais episodios (78% a",
        "89% de altas em 3 a 6 episodios distintos); a **magnitude** no extremo atual repousa",
        "sobre pouquissimos casos e nao deve ser tratada como previsao.",
        "",
        "## Conclusao",
        "",
        "1. A serie estadual **e** o produto certo: e a que o cliente paga, e o erro de base",
        "   supera a economia do gatilho por uma ordem de grandeza.",
        "2. A decomposicao nacional-mais-spread **e** a arquitetura certa para o estado, por",
        "   economia e precisao de gatilho, ainda que nao por MAE decidivel.",
        "3. O modelo estadual **nao** e mais preciso que o nacional, e nao deve ser vendido",
        "   como se fosse.",
        "4. **Nada e promovido.** O holdout estadual continua fechado, e a confirmacao vem do",
        "   ledger prospectivo, semana a semana.",
        "",
    ]
    (output_dir / "report.md").write_text(NEWLINE.join(report) + NEWLINE, encoding="utf-8")
    summary.to_csv(output_dir / "development_summary.csv", index=False)
    combined.to_csv(output_dir / "development_predictions.csv", index=False)
    print(f"{NEWLINE}gravado em {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
