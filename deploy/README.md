# Automacao semanal em VPS Linux

O job e [`scripts/weekly_refresh.py`](../scripts/weekly_refresh.py) — Python puro, sem
dependencia de shell. Ele encadeia os seis passos do fluxo semanal, para no primeiro que
falhar e sai com codigo diferente de zero, para que o agendador reclame.

```
01_download.py             fontes ANP / IPEA / BCB  (+ gate da Tabela 1)
21_s10_ingest_causal.py    ULSD, produtor, paridade -> painel causal
23_s10_parity_production.py  previsao nacional + liquidacao do ledger
26_s10_rs_regional.py      serie estadual e spread
27_s10_rs_production.py    previsao do RS + ledger estadual
29_s10_ledger_review.py    alertas dos ledgers (falha se algum degradar)
```

Leva ~2 min por execucao. Todos os passos sao idempotentes: reexecutar a mesma semana nao
duplica registro no ledger, entao repetir uma tentativa que falhou e seguro.

## Instalacao na VPS

Use **Python 3.11 ou 3.12**, nao 3.13+. Os pins de `requirements-service.lock`
(numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.2) nao tem wheel para as versoes mais novas,
e a release em `artifacts/releases/` foi serializada com essas versoes — ler o pickle com
majors diferentes dispara `InconsistentVersionWarning` e o servico marca `runtime_mismatch`.

```bash
sudo useradd --system --home /opt/s10 --shell /usr/sbin/nologin s10
sudo mkdir -p /opt/s10 /var/log/s10
sudo chown -R s10:s10 /opt/s10 /var/log/s10

sudo -u s10 git clone <seu-repo> /opt/s10
cd /opt/s10
sudo -u s10 python3.12 -m venv .venv
sudo -u s10 .venv/bin/pip install -e ".[production,ingest,service]"
```

O extra `ingest` e obrigatorio para o job: e ele que traz `requests`, `xlrd` (planilha .xls
de precos de produtor) e `openpyxl` (.xlsx de revenda).

Valide antes de agendar:

```bash
sudo -u s10 .venv/bin/python scripts/weekly_refresh.py --dry-run   # lista os passos
sudo -u s10 .venv/bin/python scripts/weekly_refresh.py --json      # execucao real
```

## Opcao A — systemd (recomendada)

Log centralizado no journal, timeout, e `Persistent=true` para a VPS nao perder a semana
se estiver desligada no horario.

```bash
sudo cp deploy/s10-weekly.service deploy/s10-weekly.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now s10-weekly.timer

systemctl list-timers s10-weekly.timer     # confere o proximo disparo
sudo systemctl start s10-weekly.service    # dispara agora, para testar
journalctl -u s10-weekly.service -n 50     # ve o resultado
```

Ajuste `User`, `WorkingDirectory` e o caminho do venv no `.service` se nao usar `/opt/s10`.

## Opcao B — cron

```bash
sudo -u s10 crontab deploy/crontab.example   # revise os caminhos antes
sudo -u s10 crontab -l
```

O `flock` no crontab evita que duas execucoes se sobreponham e escrevam no mesmo ledger.

## Opcao C — container

O `Dockerfile` da raiz constroi **a API**, nao o job: ele copia so `15_s10_service.py`,
a release e os manifestos — `.dockerignore` exclui `data/`, e os scripts de ingestao nao
entram na imagem. Para rodar o job em container voce precisa de uma imagem propria que
inclua `scripts/`, o extra `ingest`, e volumes graváveis para `data/` e `reports/`; o
agendamento continua vindo de fora (systemd/cron no host, ou o scheduler do orquestrador).

Nao testei esse caminho aqui — o daemon do Docker nao estava rodando nesta maquina.

## O que a automacao deliberadamente nao faz

**Nao move o holdout.** `S10_HOLDOUT_END` esta fixo em `selection.py:311`, com um
comentario explicito de que as datas nao devem ser alteradas. O corte e uma data, nao um
offset: se cada semana nova empurrasse a janela, os numeros ja publicados deixariam de ser
reproduziveis.

**Nao promove a release servida pela API.** Trocar o primario passa pelos gates de selecao,
que sao decisao com revisao humana. O job mantem os **ledgers prospectivos** avancando —
que e o mecanismo que decide promocoes daqui pra frente.

Consequencia pratica: `/v1/forecast` continua respondendo 503 enquanto a release
`2026-08-16` estiver vencida. Isso e a protecao de previsao vencida funcionando, nao um
efeito colateral do agendamento.

## Monitoramento

O job ja falha alto (exit != 0), entao qualquer alerta que voce ja tenha para unidade
systemd ou cron funciona. Com `--json` o stdout vira uma linha por execucao, com o passo
que falhou e as ultimas linhas do erro — facil de mandar para um webhook.

```bash
journalctl -u s10-weekly.service --since "30 days ago" | grep -E "ABORTADA|FALHOU"
```

Logs por execucao ficam em `reports/automation/weekly_<data>.log`.

Vale ligar um alerta tambem para `29_s10_ledger_review.py`: ele falha quando um ledger
degrada (cobertura caindo, MAE pior que a persistencia), que e o sinal de que o modelo
parou de valer — nao de que o download quebrou.
