FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    S10_ENVIRONMENT=production \
    S10_ALLOWED_HOSTS=localhost,127.0.0.1

WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install ".[production,service]"

#: Data da release servida (semana de competencia da ultima observacao).  Fica
#: como build-arg porque a release e imutavel: uma imagem serve exatamente uma
#: release, e trocar de release e reconstruir a imagem -- nao editar arquivo
#: dentro de um container em producao.
ARG RELEASE_DATE=2026-08-30

COPY scripts/15_s10_service.py ./scripts/15_s10_service.py
COPY artifacts/releases/s10_production_${RELEASE_DATE}.joblib ./artifacts/releases/
COPY reports/vs_epl_krls/s10_product ./reports/vs_epl_krls/s10_product
COPY reports/vs_epl_krls/s10_selection/selection_manifest_h1.json ./reports/vs_epl_krls/s10_selection/selection_manifest_h1.json

# Os quatro caminhos acima bastam para servir previsao, catalogo, evidencia e
# cenario de custo.  As previsoes challenger, os ledgers e a revisao de gates
# sao opcionais: sem eles o servico sobe igual, e apenas /v1/decision,
# /v1/basis e /v1/governance ficam indisponiveis.  Monte-os como volume quando
# quiser a camada de decisao:
#   -v $PWD/reports/vs_epl_krls/s10_parity:/app/reports/vs_epl_krls/s10_parity:ro
#   -v $PWD/reports/vs_epl_krls/s10_rs:/app/reports/vs_epl_krls/s10_rs:ro
#   -v $PWD/reports/vs_epl_krls/s10_gates:/app/reports/vs_epl_krls/s10_gates:ro

# O CMD nao pode expandir ARG, entao a release vai pelo ambiente e o shell
# resolve na hora de subir.
ENV S10_RELEASE_DATE=${RELEASE_DATE}

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health/live', timeout=2)"]
CMD ["sh", "-c", "exec python scripts/15_s10_service.py --host 0.0.0.0 --port 8000 \
  --artifact artifacts/releases/s10_production_${S10_RELEASE_DATE}.joblib \
  --manifest reports/vs_epl_krls/s10_product/releases/${S10_RELEASE_DATE}.json"]
