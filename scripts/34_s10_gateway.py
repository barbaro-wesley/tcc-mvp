"""Sobe o gateway multi-cliente da plataforma S10.

O gateway cuida de clientes, chaves, cotas e consumo, e repassa a previsao para
a API S10 -- que continua read-only, sem banco, servindo a release imutavel.

Suba os dois processos, nesta ordem::

    python scripts/15_s10_service.py --state RS            # previsao, porta 8000
    python scripts/34_s10_gateway.py                       # gateway,  porta 8080

Variaveis de ambiente relevantes::

    S10_ADMIN_TOKEN        obrigatorio em producao; protege /admin
    S10_UPSTREAM_URL       onde a API de previsao responde (padrao :8000)
    S10_UPSTREAM_API_KEY   chave da API de previsao, se ela exigir uma
    S10_TENANCY_DB         arquivo sqlite dos clientes (padrao data/tenancy.db)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s10_tenancy.gateway import GatewaySettings, create_gateway  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--upstream", default=None, help="URL da API de previsao")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("porta invalida")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = GatewaySettings.from_environment()
    if args.database is not None:
        settings = GatewaySettings(**{**vars(settings), "database": str(args.database)})
    if args.upstream is not None:
        settings = GatewaySettings(**{**vars(settings), "upstream_url": args.upstream.rstrip("/")})

    if settings.admin_token is None:
        logging.warning(
            "S10_ADMIN_TOKEN nao definido: as rotas /admin estao abertas. "
            "Aceitavel em desenvolvimento, nunca em producao."
        )

    application = create_gateway(settings)
    logging.info("gateway em http://%s:%d  -> upstream %s", args.host, args.port, settings.upstream_url)
    logging.info("banco de clientes: %s", settings.database)

    import uvicorn

    uvicorn.run(application, host=args.host, port=args.port, server_header=False, date_header=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
