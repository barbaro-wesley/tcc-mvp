from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from s10_tenancy.gateway import GatewaySettings, create_gateway
from s10_tenancy.store import TenancyStore, hash_key

ADMIN = {"X-Admin-Token": "token-de-teste"}


@pytest.fixture()
def store() -> TenancyStore:
    database = TenancyStore(":memory:")
    yield database
    database.close()


def _upstream(handler=None) -> httpx.AsyncClient:
    """Cliente falso no lugar da API de previsao: o gateway nao deve depender dela."""

    def _default(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"forecast": {"target_date": "2026-09-06", "point": 6.88}})

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler or _default),
        base_url="http://upstream.test",
    )


@pytest.fixture()
def client(store: TenancyStore):
    settings = GatewaySettings(environment="test", admin_token="token-de-teste", database=":memory:")
    app = create_gateway(settings, store=store, client=_upstream())
    with TestClient(app) as test_client:
        yield test_client


def _tenant(client: TestClient, *, cnpj: str = "12345678000199", **extra) -> dict:
    body = {"name": "Transportadora Alfa", "cnpj": cnpj, "contact_email": "ops@alfa.com.br", **extra}
    response = client.post("/admin/tenants", json=body, headers=ADMIN)
    assert response.status_code == 201, response.text
    return response.json()


def _key(client: TestClient, tenant_id: int) -> str:
    response = client.post(f"/admin/tenants/{tenant_id}/keys", json={"label": "erp"}, headers=ADMIN)
    assert response.status_code == 201, response.text
    return response.json()["api_key"]


# --------------------------------------------------------------------- store


def test_raw_key_is_never_stored(store: TenancyStore):
    tenant = store.create_tenant(name="Alfa", cnpj="1", contact_email="a@b.c")
    raw, record = store.issue_key(tenant.id)
    rows = store.list_keys(tenant.id)
    assert raw not in str(rows)
    assert record["key_prefix"] == raw[:12]
    assert store.resolve_key(raw) is not None
    # O banco guarda o hash, e so ele.
    assert store._connection.execute(  # noqa: SLF001
        "SELECT key_hash FROM api_keys WHERE id = ?", (record["id"],)
    ).fetchone()["key_hash"] == hash_key(raw)


def test_duplicate_cnpj_is_refused(store: TenancyStore):
    store.create_tenant(name="Alfa", cnpj="99", contact_email="a@b.c")
    with pytest.raises(ValueError, match="ja existe cliente"):
        store.create_tenant(name="Beta", cnpj="99", contact_email="b@b.c")


def test_deactivating_a_tenant_revokes_its_keys_and_keeps_usage(store: TenancyStore):
    tenant = store.create_tenant(name="Alfa", cnpj="1", contact_email="a@b.c")
    raw, _ = store.issue_key(tenant.id)
    store.record_usage(
        tenant_id=tenant.id, api_key_id=None, path="/v1/forecast",
        method="GET", status_code=200, latency_ms=1.0,
    )
    assert store.deactivate_tenant(tenant.id) is True
    assert store.resolve_key(raw) is None
    # O consumo sobrevive: e a base da cobranca e da auditoria.
    assert store.usage_this_period(tenant.id) == 1


def test_revoked_key_stops_resolving(store: TenancyStore):
    tenant = store.create_tenant(name="Alfa", cnpj="1", contact_email="a@b.c")
    raw, record = store.issue_key(tenant.id)
    assert store.revoke_key(int(record["id"])) is True
    assert store.resolve_key(raw) is None
    assert store.revoke_key(int(record["id"])) is False


def test_unknown_or_malformed_keys_are_rejected(store: TenancyStore):
    assert store.resolve_key("") is None
    assert store.resolve_key("sem-prefixo") is None
    assert store.resolve_key("s10_inexistente") is None


# ------------------------------------------------------------------- gateway


def test_admin_routes_require_the_admin_token(client: TestClient):
    assert client.get("/admin/tenants").status_code == 401
    assert client.get("/admin/tenants", headers={"X-Admin-Token": "errado"}).status_code == 401
    assert client.get("/admin/tenants", headers=ADMIN).status_code == 200


def test_tenant_crud_roundtrip(client: TestClient):
    created = _tenant(client)
    tenant_id = created["id"]

    assert client.get(f"/admin/tenants/{tenant_id}", headers=ADMIN).json()["name"] == "Transportadora Alfa"

    patched = client.patch(
        f"/admin/tenants/{tenant_id}", json={"monthly_quota": 50, "plan": "enterprise"}, headers=ADMIN
    ).json()
    assert patched["monthly_quota"] == 50 and patched["plan"] == "enterprise"

    assert client.delete(f"/admin/tenants/{tenant_id}", headers=ADMIN).json()["active"] is False
    assert client.get("/admin/tenants", headers=ADMIN).json()["count"] == 0
    assert client.get(f"/admin/tenants/{tenant_id}", headers=ADMIN).json()["active"] is False


def test_missing_tenant_returns_404(client: TestClient):
    assert client.get("/admin/tenants/999", headers=ADMIN).status_code == 404
    assert client.patch("/admin/tenants/999", json={"plan": "x"}, headers=ADMIN).status_code == 404
    assert client.delete("/admin/tenants/999", headers=ADMIN).status_code == 404


def test_forecast_requires_a_valid_key(client: TestClient):
    assert client.get("/v1/forecast").status_code == 401
    assert client.get("/v1/forecast", headers={"X-API-Key": "s10_invalida"}).status_code == 401


def test_authenticated_call_is_proxied_and_metered(client: TestClient):
    tenant = _tenant(client)
    key = _key(client, tenant["id"])

    response = client.get("/v1/forecast", headers={"X-API-Key": key})
    assert response.status_code == 200
    assert response.json()["forecast"]["target_date"] == "2026-09-06"

    usage = client.get(f"/admin/tenants/{tenant['id']}/usage", headers=ADMIN).json()
    assert usage["calls"] == 1
    assert usage["by_path"][0]["path"] == "/v1/forecast"
    assert usage["quota_remaining"] == usage["monthly_quota"] - 1


def test_quota_exhaustion_returns_402(client: TestClient):
    tenant = _tenant(client, monthly_quota=2)
    key = _key(client, tenant["id"])
    headers = {"X-API-Key": key}

    assert client.get("/v1/forecast", headers=headers).status_code == 200
    assert client.get("/v1/forecast", headers=headers).status_code == 200
    blocked = client.get("/v1/forecast", headers=headers)
    assert blocked.status_code == 402
    assert "cota mensal" in blocked.json()["detail"]


def test_rate_limit_is_per_tenant_not_per_ip(client: TestClient):
    """Duas transportadoras atras do mesmo IP nao competem pelo mesmo balde."""

    slow = _tenant(client, cnpj="11111111000111", rate_limit=1)
    other = _tenant(client, cnpj="22222222000122", rate_limit=1)
    slow_key, other_key = _key(client, slow["id"]), _key(client, other["id"])

    assert client.get("/v1/forecast", headers={"X-API-Key": slow_key}).status_code == 200
    assert client.get("/v1/forecast", headers={"X-API-Key": slow_key}).status_code == 429
    # O segundo cliente, mesmo IP, segue atendido.
    assert client.get("/v1/forecast", headers={"X-API-Key": other_key}).status_code == 200


def test_revoked_key_loses_access_immediately(client: TestClient):
    tenant = _tenant(client)
    response = client.post(f"/admin/tenants/{tenant['id']}/keys", json={"label": "erp"}, headers=ADMIN)
    key, key_id = response.json()["api_key"], response.json()["id"]

    assert client.get("/v1/forecast", headers={"X-API-Key": key}).status_code == 200
    assert client.delete(f"/admin/keys/{key_id}", headers=ADMIN).status_code == 200
    assert client.get("/v1/forecast", headers={"X-API-Key": key}).status_code == 401


def test_only_declared_paths_are_proxied(client: TestClient):
    tenant = _tenant(client)
    key = _key(client, tenant["id"])
    assert client.get("/v1/health/ready", headers={"X-API-Key": key}).status_code == 404


def test_upstream_failure_is_reported_and_still_metered(store: TenancyStore):
    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream fora do ar")

    settings = GatewaySettings(environment="test", admin_token="token-de-teste", database=":memory:")
    app = create_gateway(settings, store=store, client=_upstream(failing))
    with TestClient(app) as client:
        tenant = _tenant(client)
        key = _key(client, tenant["id"])
        response = client.get("/v1/forecast", headers={"X-API-Key": key})
        assert response.status_code == 503
        assert "indisponivel" in response.json()["detail"]
        # A chamada que falhou tambem conta: a fatura precisa bater com o que o
        # cliente observou.
        assert client.get(f"/admin/tenants/{tenant['id']}/usage", headers=ADMIN).json()["calls"] == 1


def test_whoami_reports_plan_and_quota(client: TestClient):
    tenant = _tenant(client, monthly_quota=100)
    key = _key(client, tenant["id"])
    body = client.get("/v1/me", headers={"X-API-Key": key}).json()
    assert body["tenant"]["name"] == "Transportadora Alfa"
    assert body["key_label"] == "erp"
    assert body["usage"]["monthly_quota"] == 100


def test_production_requires_an_admin_token(monkeypatch):
    monkeypatch.setenv("S10_ENVIRONMENT", "production")
    monkeypatch.delenv("S10_ADMIN_TOKEN", raising=False)
    with pytest.raises(ValueError, match="S10_ADMIN_TOKEN"):
        GatewaySettings.from_environment()
