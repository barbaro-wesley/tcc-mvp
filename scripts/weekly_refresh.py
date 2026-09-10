"""Atualizacao semanal automatica do S10: fontes, painel, previsoes e ledgers.

Encadeia os scripts do fluxo semanal na ordem em que dependem uns dos outros e
para no primeiro que falhar.  Escrito em Python puro, sem dependencia de shell,
para rodar igual em Linux (cron, systemd, container) e no Windows.

Cada passo e idempotente: reexecutar a mesma semana nao duplica registro no
ledger, entao o agendador pode tentar de novo sem sujar a evidencia.

O que este runner deliberadamente NAO faz:

  * Nao move o holdout.  ``S10_HOLDOUT_END`` esta fixo em ``selection.py`` e a
    janela de 104 semanas sustenta a evidencia ja publicada.  Reajustar o corte
    a cada semana nova tornaria os numeros divulgados irreproduziveis.
  * Nao retreina nem promove a release nacional servida pela API.  Trocar o
    primario passa pelos gates de selecao, que sao uma decisao com revisao
    humana, nao um efeito colateral de cron.

O efeito pratico e manter os ledgers prospectivos avancando semana a semana --
que e justamente a evidencia que decide promocoes daqui pra frente.

Uso::

    python scripts/weekly_refresh.py
    python scripts/weekly_refresh.py --skip-download
    python scripts/weekly_refresh.py --json          # uma linha por execucao
    python scripts/weekly_refresh.py --dry-run

Sai com codigo diferente de zero quando qualquer passo falha, para que o
agendador reclame em vez de escrever num log que ninguem le.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

#: (script, rotulo, exigido).  A ordem e a das dependencias entre eles: o painel
#: causal precisa do CSV semanal escrito pelo download, e as previsoes precisam
#: do painel.
STEPS: tuple[tuple[str, str, bool], ...] = (
    # Revenda ANP, Brent e cambio.  Revalida o gate de reproducao da Tabela 1;
    # se a ANP mudar o layout da planilha, quebra aqui e nao mais adiante.
    ("01_download.py", "download das fontes ANP/IPEA/BCB", True),
    # ULSD, preco de produtor e paridade -> data/processed/s10_causal_panel.csv
    ("21_s10_ingest_causal.py", "ingestao causal e painel", True),
    # Previsao nacional + liquidacao da semana vencida no ledger.
    ("23_s10_parity_production.py", "previsao nacional (paridade)", True),
    # Serie estadual e avaliacao do spread.
    ("26_s10_rs_regional.py", "serie estadual RS", True),
    # Previsao do RS + ledger estadual.
    ("27_s10_rs_production.py", "previsao estadual (RS)", True),
    # Alertas dos ledgers.  Sai != 0 quando algum degrada, entao e ele que
    # transforma "rodou" em "rodou e continua saudavel".
    ("29_s10_ledger_review.py", "revisao dos ledgers", True),
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class Runner:
    def __init__(self, root: Path, log_path: Path, as_json: bool) -> None:
        self.root = root
        self.log_path = log_path
        self.as_json = as_json
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{_now()}  {message}"
        # Em modo --json o stdout carrega o resumo legivel por maquina; as
        # linhas humanas vao so para o arquivo.
        if not self.as_json:
            print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def run_step(self, script: str, label: str) -> dict[str, object]:
        self.log(f"==> {label} ({script})")
        started = time.monotonic()
        # sys.executable mantem o job no mesmo interpretador/venv que o invocou,
        # sem depender de PATH nem de "activate" -- o que importa no cron.
        completed = subprocess.run(
            [sys.executable, str(self.root / "scripts" / script)],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        seconds = round(time.monotonic() - started, 1)
        output = (completed.stdout or "") + (completed.stderr or "")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(output.rstrip() + "\n")

        record: dict[str, object] = {
            "script": script,
            "label": label,
            "exit_code": completed.returncode,
            "seconds": seconds,
        }
        if completed.returncode != 0:
            self.log(f"FALHOU {label} (exit {completed.returncode}, {seconds}s)")
            tail = [ln for ln in output.strip().splitlines() if ln.strip()][-3:]
            for line in tail:
                self.log(f"    {line}")
            record["tail"] = tail
        else:
            self.log(f"ok {label} ({seconds}s)")
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="raiz do repositorio")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="pula 01_download.py; reprocessa sem bater nas fontes de novo",
    )
    parser.add_argument("--json", action="store_true", help="resumo JSON no stdout")
    parser.add_argument(
        "--dry-run", action="store_true", help="lista os passos sem executar nada"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="diretorio dos logs (padrao: <root>/reports/automation)",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    log_dir = args.log_dir or (root / "reports" / "automation")
    log_path = log_dir / f"weekly_{datetime.now().strftime('%Y-%m-%d')}.log"

    steps = [s for s in STEPS if not (args.skip_download and s[0] == "01_download.py")]

    if args.dry_run:
        for index, (script, label, _) in enumerate(steps, start=1):
            print(f"{index}. {label}  ({script})")
        return 0

    runner = Runner(root, log_path, args.json)
    runner.log("inicio da atualizacao semanal")
    if args.skip_download:
        runner.log("download pulado (--skip-download)")

    started = time.monotonic()
    results: list[dict[str, object]] = []
    failed: dict[str, object] | None = None

    for script, label, _required in steps:
        record = runner.run_step(script, label)
        results.append(record)
        if record["exit_code"] != 0:
            failed = record
            break

    total = round(time.monotonic() - started, 1)
    ok = failed is None
    runner.log(
        "atualizacao semanal concluida com sucesso"
        if ok
        else f"atualizacao semanal ABORTADA em '{failed['label']}'"
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "finished_at": _now(),
                    "seconds": total,
                    "steps": results,
                    "failed_step": failed["script"] if failed else None,
                    "log": str(log_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
