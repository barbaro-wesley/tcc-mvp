"""Persistencia de clientes, chaves e uso da plataforma S10.

Por que um servico separado
---------------------------
A API de previsao se declara ``read_only`` e carrega uma release imutavel
verificada por SHA-256.  Colocar cadastro, cotas e consumo dentro dela faria o
banco virar dependencia de subida do servico que serve previsao, e misturaria
estado mutavel de clientes com evidencia auditavel.  Este modulo e o estado
mutavel; a previsao continua do outro lado, intacta.

Por que sqlite3 e SQL direto
----------------------------
Sem ORM novo no projeto: ``sqlite3`` e stdlib, o schema cabe em uma tela e o SQL
e ANSI o suficiente para migrar a Postgres trocando a conexao e os tipos.  A
mesma escolha que ``audit.py`` ja faz ao mativer o ledger pequeno e sem
dependencia.

Segredos
--------
A chave de API e mostrada **uma unica vez**, na criacao.  O banco guarda apenas
``sha256(chave)``: um vazamento do arquivo nao entrega as chaves dos clientes.
A comparacao usa ``hmac.compare_digest`` para nao vazar tempo.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
import threading

#: Prefixo legivel para a chave nao ser confundida com outro segredo em um .env.
KEY_PREFIX = "s10_"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    cnpj          TEXT    NOT NULL UNIQUE,
    contact_email TEXT    NOT NULL,
    plan          TEXT    NOT NULL DEFAULT 'piloto',
    monthly_quota INTEGER NOT NULL DEFAULT 10000,
    rate_limit    INTEGER NOT NULL DEFAULT 120,
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id  INTEGER NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    -- Apenas o hash: a chave em claro existe uma vez, na resposta da criacao.
    key_hash   TEXT    NOT NULL UNIQUE,
    key_prefix TEXT    NOT NULL,
    label      TEXT    NOT NULL DEFAULT '',
    active     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT    NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);

CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    api_key_id  INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
    path        TEXT    NOT NULL,
    method      TEXT    NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms  REAL    NOT NULL,
    -- Guardado como YYYY-MM para a cota mensal ser um COUNT indexado, nao um
    -- scan com strftime sobre a tabela inteira.
    period      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_tenant_period ON usage_events(tenant_id, period);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _period(moment: datetime | None = None) -> str:
    return (moment or datetime.now(timezone.utc)).strftime("%Y-%m")


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key() -> str:
    return f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"


@dataclass(frozen=True)
class Tenant:
    id: int
    name: str
    cnpj: str
    contact_email: str
    plan: str
    monthly_quota: int
    rate_limit: int
    active: bool
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedKey:
    """Identidade por tras de uma requisicao autenticada."""

    tenant: Tenant
    api_key_id: int
    key_label: str


class TenancyStore:
    """Acesso ao banco de clientes.  Seguro para uso concorrente do servico."""

    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + lock proprio: uvicorn atende em varias
        # threads, e serializar aqui e mais simples que um pool para a escala
        # de um piloto.
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # ------------------------------------------------------------------ tenants

    def create_tenant(
        self,
        *,
        name: str,
        cnpj: str,
        contact_email: str,
        plan: str = "piloto",
        monthly_quota: int = 10_000,
        rate_limit: int = 120,
    ) -> Tenant:
        if not name.strip() or not cnpj.strip():
            raise ValueError("nome e CNPJ sao obrigatorios")
        if monthly_quota < 0 or rate_limit < 1:
            raise ValueError("cota nao pode ser negativa e rate limit precisa ser positivo")
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "INSERT INTO tenants (name, cnpj, contact_email, plan, monthly_quota,"
                    " rate_limit, active, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (name.strip(), cnpj.strip(), contact_email.strip(), plan,
                     monthly_quota, rate_limit, _now()),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"ja existe cliente com o CNPJ {cnpj}") from exc
            return self._tenant_by_id(int(cursor.lastrowid))

    def list_tenants(self, *, include_inactive: bool = False) -> list[Tenant]:
        query = "SELECT * FROM tenants"
        if not include_inactive:
            query += " WHERE active = 1"
        query += " ORDER BY id"
        with self._lock:
            rows = self._connection.execute(query).fetchall()
        return [self._row_to_tenant(row) for row in rows]

    def get_tenant(self, tenant_id: int) -> Tenant | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone()
        return self._row_to_tenant(row) if row else None

    def update_tenant(self, tenant_id: int, **fields: object) -> Tenant | None:
        allowed = {"name", "contact_email", "plan", "monthly_quota", "rate_limit", "active"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return self.get_tenant(tenant_id)
        if "monthly_quota" in updates and int(updates["monthly_quota"]) < 0:  # type: ignore[arg-type]
            raise ValueError("cota nao pode ser negativa")
        if "rate_limit" in updates and int(updates["rate_limit"]) < 1:  # type: ignore[arg-type]
            raise ValueError("rate limit precisa ser positivo")
        if "active" in updates:
            updates["active"] = int(bool(updates["active"]))
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._lock:
            self._connection.execute(
                f"UPDATE tenants SET {assignments} WHERE id = ?",
                (*updates.values(), tenant_id),
            )
            self._connection.commit()
        return self.get_tenant(tenant_id)

    def deactivate_tenant(self, tenant_id: int) -> bool:
        """Desativa em vez de apagar: o consumo ja registrado precisa sobreviver."""

        with self._lock:
            cursor = self._connection.execute(
                "UPDATE tenants SET active = 0 WHERE id = ?", (tenant_id,)
            )
            self._connection.execute(
                "UPDATE api_keys SET active = 0, revoked_at = ? WHERE tenant_id = ?",
                (_now(), tenant_id),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    # --------------------------------------------------------------------- keys

    def issue_key(self, tenant_id: int, *, label: str = "") -> tuple[str, dict[str, object]]:
        """Emite uma chave nova.  O valor em claro so existe neste retorno."""

        if self.get_tenant(tenant_id) is None:
            raise ValueError(f"cliente {tenant_id} nao existe")
        raw = generate_key()
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO api_keys (tenant_id, key_hash, key_prefix, label, active,"
                " created_at) VALUES (?, ?, ?, ?, 1, ?)",
                (tenant_id, hash_key(raw), raw[:12], label, _now()),
            )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT id, tenant_id, key_prefix, label, active, created_at, revoked_at"
                " FROM api_keys WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        return raw, dict(row)

    def list_keys(self, tenant_id: int) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, tenant_id, key_prefix, label, active, created_at, revoked_at"
                " FROM api_keys WHERE tenant_id = ? ORDER BY id",
                (tenant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def revoke_key(self, key_id: int) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE api_keys SET active = 0, revoked_at = ? WHERE id = ? AND active = 1",
                (_now(), key_id),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def resolve_key(self, raw_key: str) -> ResolvedKey | None:
        """Traduz a chave apresentada em cliente.  Retorna None se invalida."""

        if not raw_key or not raw_key.startswith(KEY_PREFIX):
            return None
        digest = hash_key(raw_key)
        with self._lock:
            row = self._connection.execute(
                "SELECT k.id AS key_id, k.label AS key_label, k.key_hash AS key_hash,"
                " t.* FROM api_keys k JOIN tenants t ON t.id = k.tenant_id"
                " WHERE k.key_hash = ? AND k.active = 1 AND t.active = 1",
                (digest,),
            ).fetchone()
        if row is None:
            return None
        # Redundante apos o WHERE, mas mantem a comparacao final em tempo
        # constante mesmo se a busca virar um scan em outro backend.
        if not hmac.compare_digest(str(row["key_hash"]), digest):
            return None
        return ResolvedKey(
            tenant=self._row_to_tenant(row),
            api_key_id=int(row["key_id"]),
            key_label=str(row["key_label"]),
        )

    # -------------------------------------------------------------------- usage

    def record_usage(
        self,
        *,
        tenant_id: int,
        api_key_id: int | None,
        path: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO usage_events (tenant_id, api_key_id, path, method,"
                " status_code, latency_ms, period, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tenant_id, api_key_id, path, method, status_code,
                 float(latency_ms), _period(), _now()),
            )
            self._connection.commit()

    def usage_this_period(self, tenant_id: int, *, period: str | None = None) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS total FROM usage_events"
                " WHERE tenant_id = ? AND period = ?",
                (tenant_id, period or _period()),
            ).fetchone()
        return int(row["total"])

    def usage_summary(self, tenant_id: int, *, period: str | None = None) -> dict[str, object]:
        window = period or _period()
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            raise ValueError(f"cliente {tenant_id} nao existe")
        with self._lock:
            totals = self._connection.execute(
                "SELECT COUNT(*) AS calls, AVG(latency_ms) AS avg_latency,"
                " SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS errors"
                " FROM usage_events WHERE tenant_id = ? AND period = ?",
                (tenant_id, window),
            ).fetchone()
            by_path = self._connection.execute(
                "SELECT path, COUNT(*) AS calls FROM usage_events"
                " WHERE tenant_id = ? AND period = ? GROUP BY path ORDER BY calls DESC",
                (tenant_id, window),
            ).fetchall()
        calls = int(totals["calls"] or 0)
        quota = int(tenant.monthly_quota)
        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "period": window,
            "calls": calls,
            "errors": int(totals["errors"] or 0),
            "avg_latency_ms": round(float(totals["avg_latency"] or 0.0), 3),
            "monthly_quota": quota,
            "quota_remaining": max(0, quota - calls),
            "quota_used_fraction": round(calls / quota, 4) if quota else None,
            "by_path": [dict(row) for row in by_path],
        }

    # ------------------------------------------------------------------ helpers

    def _tenant_by_id(self, tenant_id: int) -> Tenant:
        row = self._connection.execute(
            "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        return self._row_to_tenant(row)

    @staticmethod
    def _row_to_tenant(row: sqlite3.Row) -> Tenant:
        return Tenant(
            id=int(row["id"]),
            name=str(row["name"]),
            cnpj=str(row["cnpj"]),
            contact_email=str(row["contact_email"]),
            plan=str(row["plan"]),
            monthly_quota=int(row["monthly_quota"]),
            rate_limit=int(row["rate_limit"]),
            active=bool(row["active"]),
            created_at=str(row["created_at"]),
        )
