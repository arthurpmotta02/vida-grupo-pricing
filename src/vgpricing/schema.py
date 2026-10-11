"""Esquema canônico das tabelas de entrada.

Cada tabela tem colunas obrigatórias e opcionais, com o tipo esperado.
O leitor converte os tipos e registra (não descarta silenciosamente) valores
que não puderam ser convertidos.
"""
from __future__ import annotations

ESQUEMA: dict[str, dict[str, dict[str, str]]] = {
    "estipulantes": {
        "obrigatorias": {
            "id_estipulante": "str",
            "cnae_secao": "str",
            "data_inicio_vigencia": "date",
        },
        "opcionais": {
            "cnae_divisao": "str",
            "data_fim_vigencia": "date",
            "tipo_capital": "str",
            "contributario": "bool",
            "taxa_vigente_mensal_mil": "float",
        },
    },
    "vidas": {
        "obrigatorias": {
            "id_estipulante": "str",
            "id_vida": "str",
            "data_nascimento": "date",
            "sexo": "str",
            "data_inicio": "date",
            "capital": "float",
        },
        "opcionais": {
            "data_fim": "date",
            "situacao": "str",
        },
    },
    "sinistros": {
        "obrigatorias": {
            "id_sinistro": "str",
            "id_estipulante": "str",
            "id_vida": "str",
            "cobertura": "str",
            "data_ocorrencia": "date",
            "data_aviso": "date",
        },
        "opcionais": {
            "data_pagamento": "date",
            "valor_indenizado": "float",
            "valor_reserva": "float",
            "status": "str",
        },
    },
    "premios": {
        "obrigatorias": {
            "id_estipulante": "str",
            "inicio_periodo": "date",
            "fim_periodo": "date",
            "premio_emitido": "float",
        },
        "opcionais": {
            "taxa_mensal_mil": "float",
        },
    },
}

VALORES_SEXO = {"M", "F"}
VALORES_SITUACAO = {"ativo", "afastado", "aposentado_invalidez"}
VALORES_STATUS = {"pago", "pendente", "negado"}


def colunas(tabela: str) -> dict[str, str]:
    e = ESQUEMA[tabela]
    return {**e["obrigatorias"], **e["opcionais"]}
