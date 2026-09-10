"""Gateway multi-cliente da plataforma S10.

O que este servico faz
----------------------
1. **CRUD de clientes** -- cadastro das transportadoras, plano, cota e limite.
2. **Chaves por cliente** -- emissao e revogacao; o segredo aparece uma vez.
3. **Cota e rate limit por cliente** -- por identidade, nao por IP: varias
   transportadoras atras do mesmo NAT corporativo nao competem pelo mesmo balde,
   e uma nao derruba a outra.
4. **Log de uso** -- cada requisicao vira uma linha, base para cobranca e para
   o cliente ver o proprio consumo.
5. **Repasse da previsao** -- encaminha para a API S10, que continua read-only.

O que ele deliberadamente NAO faz
---------------------------------
Nao carrega release, nao le artefato ``.joblib`` e nao calcula previsao.  A API
de previsao permanece ``read_only`` e sem banco; toda a mutabilidade -- clientes,
chaves, consumo -- vive aqui.  Se este servico cair, a previsao continua sendo
servida do outro lado.

Administracao
-------------
As rotas ``/admin`` exigem ``X-Admin-Token`` (``S10_ADMIN_TOKEN``).  Sem a
variavel definida o servico recusa subir em producao, para nao existir um painel
administrativo aberto por esquecimento.
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import os
import threading
import time
from typing import Any, Deque

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .store import ResolvedKey, TenancyStore

#: Rotas da API de previsao que o gateway repassa.  Lista fechada: uma rota nova
#: la nao fica automaticamente exposta aqui.
PROXIED_PATHS = {
    "/v1/forecast",
    "/v1/models",
    "/v1/evidence",
    "/v1/governance",
    "/v1/decision",
    "/v1/basis",
    "/v1/scenarios/cost",
}


class TenantCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    cnpj: str = Field(min_length=11, max_length=20)
    contact_email: str = Field(min_length=3, max_length=200)
    plan: str = Field(default="piloto", max_length=40)
    monthly_quota: int = Field(default=10_000, ge=0, le=10_000_000)
    rate_limit: int = Field(default=120, ge=1, le=10_000)


class TenantUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    contact_email: str | None = Field(default=None, min_length=3, max_length=200)
    plan: str | None = Field(default=None, max_length=40)
    monthly_quota: int | None = Field(default=None, ge=0, le=10_000_000)
    rate_limit: int | None = Field(default=None, ge=1, le=10_000)
    active: bool | None = None


class KeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(default="", max_length=120)


@dataclass(frozen=True)
class GatewaySettings:
    environment: str = "development"
    admin_token: str | None = None
    upstream_url: str = "http://127.0.0.1:8000"
    upstream_api_key: str | None = None
    database: str = "data/tenancy.db"
    request_timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls) -> "GatewaySettings":
        environment = os.getenv("S10_ENVIRONMENT", "development").strip().lower()
        if environment not in {"development", "test", "production"}:
            raise ValueError("S10_ENVIRONMENT must be development, test, or production")
        admin_token = os.getenv("S10_ADMIN_TOKEN") or None
        if environment == "production" and not admin_token:
            raise ValueError(
                "S10_ADMIN_TOKEN e obrigatorio em producao: sem ele as rotas /admin "
                "ficariam abertas"
            )
        return cls(
            environment=environment,
            admin_token=admin_token,
            upstream_url=os.getenv("S10_UPSTREAM_URL", "http://127.0.0.1:8000").rstrip("/"),
            upstream_api_key=os.getenv("S10_UPSTREAM_API_KEY") or None,
            database=os.getenv("S10_TENANCY_DB", "data/tenancy.db"),
            request_timeout_seconds=float(os.getenv("S10_UPSTREAM_TIMEOUT", "10")),
        )


class _TenantRateLimiter:
    """Janela deslizante de um minuto, chaveada por cliente."""

    def __init__(self) -> None:
        self._calls: dict[int, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, tenant_id: int, limit: int, now: float) -> bool:
        with self._lock:
            window = self._calls[tenant_id]
            while window and window[0] <= now - 60:
                window.popleft()
            if len(window) >= limit:
                return False
            window.append(now)
            return True


def create_gateway(
    settings: GatewaySettings | None = None,
    *,
    store: TenancyStore | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    configuration = settings or GatewaySettings.from_environment()
    database = store or TenancyStore(configuration.database)
    limiter = _TenantRateLimiter()
    owns_client = client is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = client or httpx.AsyncClient(
            base_url=configuration.upstream_url,
            timeout=configuration.request_timeout_seconds,
        )
        yield
        if owns_client:
            await app.state.http.aclose()

    application = FastAPI(
        title="S10 Platform Gateway",
        version="1.0.0",
        description="Clientes, chaves, cotas e consumo da plataforma S10.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------- auth

    def require_admin(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> None:
        if configuration.admin_token is None:
            if configuration.environment == "production":  # pragma: no cover - barrado no settings
                raise HTTPException(status_code=500, detail="admin token nao configurado")
            return  # desenvolvimento: painel aberto, como o .env.example ja assume
        if x_admin_token is None or not hmac.compare_digest(x_admin_token, configuration.admin_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="admin token invalido")

    def require_tenant(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> ResolvedKey:
        if not x_api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key ausente")
        resolved = database.resolve_key(x_api_key)
        if resolved is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="chave invalida ou revogada")

        tenant = resolved.tenant
        if not limiter.allow(tenant.id, tenant.rate_limit, time.monotonic()):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"rate limit de {tenant.rate_limit}/min excedido",
                headers={"Retry-After": "60"},
            )
        used = database.usage_this_period(tenant.id)
        if tenant.monthly_quota and used >= tenant.monthly_quota:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"cota mensal de {tenant.monthly_quota} chamadas atingida "
                    f"({used} usadas); fale com o comercial para ampliar o plano"
                ),
            )
        return resolved

    # ------------------------------------------------------------ admin: CRUD

    @application.post("/admin/tenants", status_code=201, dependencies=[Depends(require_admin)], tags=["admin"])
    def create_tenant(payload: TenantCreate) -> dict[str, object]:
        try:
            tenant = database.create_tenant(**payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return tenant.as_dict()

    @application.get("/admin/tenants", dependencies=[Depends(require_admin)], tags=["admin"])
    def list_tenants(include_inactive: bool = False) -> dict[str, object]:
        tenants = database.list_tenants(include_inactive=include_inactive)
        return {"count": len(tenants), "tenants": [t.as_dict() for t in tenants]}

    @application.get("/admin/tenants/{tenant_id}", dependencies=[Depends(require_admin)], tags=["admin"])
    def read_tenant(tenant_id: int) -> dict[str, object]:
        tenant = database.get_tenant(tenant_id)
        if tenant is None:
            raise HTTPException(status_code=404, detail="cliente nao encontrado")
        return tenant.as_dict()

    @application.patch("/admin/tenants/{tenant_id}", dependencies=[Depends(require_admin)], tags=["admin"])
    def update_tenant(tenant_id: int, payload: TenantUpdate) -> dict[str, object]:
        if database.get_tenant(tenant_id) is None:
            raise HTTPException(status_code=404, detail="cliente nao encontrado")
        try:
            tenant = database.update_tenant(tenant_id, **payload.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        assert tenant is not None
        return tenant.as_dict()

    @application.delete("/admin/tenants/{tenant_id}", dependencies=[Depends(require_admin)], tags=["admin"])
    def deactivate_tenant(tenant_id: int) -> dict[str, object]:
        if not database.deactivate_tenant(tenant_id):
            raise HTTPException(status_code=404, detail="cliente nao encontrado")
        # Desativa, nao apaga: o consumo registrado sustenta cobranca e auditoria.
        return {"tenant_id": tenant_id, "active": False, "keys_revoked": True}

    # ------------------------------------------------------------- admin: keys

    @application.post("/admin/tenants/{tenant_id}/keys", status_code=201,
                      dependencies=[Depends(require_admin)], tags=["admin"])
    def issue_key(tenant_id: int, payload: KeyCreate) -> dict[str, object]:
        try:
            raw, record = database.issue_key(tenant_id, label=payload.label)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            **record,
            "api_key": raw,
            "warning": "guarde agora: a chave nao pode ser recuperada depois",
        }

    @application.get("/admin/tenants/{tenant_id}/keys", dependencies=[Depends(require_admin)], tags=["admin"])
    def list_keys(tenant_id: int) -> dict[str, object]:
        if database.get_tenant(tenant_id) is None:
            raise HTTPException(status_code=404, detail="cliente nao encontrado")
        return {"tenant_id": tenant_id, "keys": database.list_keys(tenant_id)}

    @application.delete("/admin/keys/{key_id}", dependencies=[Depends(require_admin)], tags=["admin"])
    def revoke_key(key_id: int) -> dict[str, object]:
        if not database.revoke_key(key_id):
            raise HTTPException(status_code=404, detail="chave nao encontrada ou ja revogada")
        return {"key_id": key_id, "active": False}

    @application.get("/admin/tenants/{tenant_id}/usage", dependencies=[Depends(require_admin)], tags=["admin"])
    def tenant_usage(tenant_id: int, period: str | None = None) -> dict[str, object]:
        try:
            return database.usage_summary(tenant_id, period=period)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ----------------------------------------------------------- cliente final

    @application.get("/v1/me", tags=["cliente"])
    def whoami(identity: ResolvedKey = Depends(require_tenant)) -> dict[str, object]:
        """O cliente ve quem ele e e quanto da cota ja gastou."""

        summary = database.usage_summary(identity.tenant.id)
        return {
            "tenant": {
                "id": identity.tenant.id,
                "name": identity.tenant.name,
                "plan": identity.tenant.plan,
                "rate_limit_per_minute": identity.tenant.rate_limit,
            },
            "key_label": identity.key_label,
            "usage": summary,
        }

    @application.api_route("/v1/{resource:path}", methods=["GET", "POST"], tags=["previsao"])
    async def proxy(
        resource: str,
        request: Request,
        identity: ResolvedKey = Depends(require_tenant),
    ) -> JSONResponse:
        """Repassa para a API de previsao e contabiliza a chamada."""

        path = f"/v1/{resource}"
        if path not in PROXIED_PATHS:
            raise HTTPException(status_code=404, detail=f"recurso {path} nao e exposto pelo gateway")

        headers: dict[str, str] = {}
        if configuration.upstream_api_key:
            headers["X-API-Key"] = configuration.upstream_api_key
        body = await request.body()
        started = time.perf_counter()
        try:
            upstream = await request.app.state.http.request(
                request.method,
                path,
                content=body or None,
                headers={**headers, "Content-Type": "application/json"} if body else headers,
            )
            payload: Any = upstream.json()
            code = upstream.status_code
        except httpx.HTTPError as exc:
            payload, code = {"detail": f"servico de previsao indisponivel: {exc}"}, 503

        elapsed = (time.perf_counter() - started) * 1000.0
        # Contabiliza inclusive erro: uma chamada que falhou consumiu recurso, e
        # esconde-la do log faria a fatura divergir do que o cliente observou.
        database.record_usage(
            tenant_id=identity.tenant.id,
            api_key_id=identity.api_key_id,
            path=path,
            method=request.method,
            status_code=code,
            latency_ms=elapsed,
        )
        return JSONResponse(status_code=code, content=payload)

    @application.get("/health", tags=["health"])
    def health() -> dict[str, object]:
        return {
            "status": "alive",
            "service": "s10-gateway",
            "upstream": configuration.upstream_url,
            "environment": configuration.environment,
        }

    application.state.store = database
    application.state.settings = configuration
    return application
