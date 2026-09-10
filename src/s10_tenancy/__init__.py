"""Camada multi-cliente da plataforma S10: clientes, chaves, cotas e consumo.

Separada da API de previsao de proposito.  ``vs_epl_krls.api`` serve uma release
imutavel e se declara ``read_only``; todo o estado mutavel da plataforma vive
aqui, em outro processo e outro banco.
"""

from .gateway import GatewaySettings, create_gateway
from .store import ResolvedKey, Tenant, TenancyStore, generate_key, hash_key

__all__ = [
    "GatewaySettings",
    "ResolvedKey",
    "Tenant",
    "TenancyStore",
    "create_gateway",
    "generate_key",
    "hash_key",
]
