#!/usr/bin/env python3
"""
Gera roteiro de aula usando a API de Inferência do Hugging Face com Qwen.
Não precisa baixar o modelo; requer HF token (em .env ou variável de ambiente).

Uso: python roteiro_aula_qwen_api.py "Tema da aula"
"""

import argparse
import os
import requests
from dotenv import load_dotenv

load_dotenv()


# Modelo na API (pode usar Qwen/Qwen2.5-7B-Instruct ou outro disponível na API)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Nova API (router) usa chat completions
API_URL = "https://router.huggingface.co/v1/chat/completions"


def gerar_roteiro_api(
    tema: str,
    duracao_min: int = 50,
    nivel: str = "ensino médio",
    token: str = None,
) -> str:
    """Gera roteiro de aula via API do Hugging Face."""
    token = (
        token
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not token:
        raise ValueError(
            "Defina HUGGING_FACE_HUB_TOKEN ou HF_TOKEN no arquivo .env ou no ambiente. "
            "Obtenha em: https://huggingface.co/settings/tokens"
        )

    nivel_desc = {
        "fundamental_i": "anos iniciais do ensino fundamental",
        "fundamental_ii": "anos finais do ensino fundamental",
        "ensino_medio": "ensino médio",
        "superior": "ensino superior",
    }.get(nivel.lower().replace(" ", "_"), nivel)

    prompt = f"""Você é um professor experiente. Crie um roteiro de aula completo e objetivo.

Tema da aula: {tema}
Público-alvo: {nivel_desc}
Duração sugerida: {duracao_min} minutos

O roteiro deve conter, de forma clara e numerada:
1. Objetivos de aprendizagem (3 a 5 itens)
2. Materiais necessários
3. Desenvolvimento da aula (etapas com tempo aproximado)
4. Atividades ou dinâmicas
5. Avaliação ou verificação de aprendizagem
6. Sugestões de aprofundamento ou tarefa de casa

Roteiro da aula:
"""

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7,
        "stream": False,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    try:
        out = response.json()
    except Exception:
        out = {}

    if not response.ok:
        err = out.get("error", out.get("message", response.text or f"HTTP {response.status_code}"))
        if isinstance(err, dict):
            msg = err.get("message", str(err))
        else:
            msg = str(err)
        if response.status_code == 401:
            raise RuntimeError(
                "Token inválido ou não autorizado. Verifique o HUGGING_FACE_HUB_TOKEN no .env."
            )
        if response.status_code == 403:
            raise RuntimeError(
                "Acesso negado à API. Confira se o token tem permissão e se o modelo existe."
            )
        if response.status_code == 429:
            raise RuntimeError(
                "Muitas requisições. Aguarde um pouco e tente novamente."
            )
        if response.status_code == 503:
            if "loading" in msg.lower():
                raise RuntimeError(
                    "Modelo ainda está carregando na API. Aguarde ~20s e tente de novo."
                )
            raise RuntimeError(f"API temporariamente indisponível: {msg}")
        raise RuntimeError(f"Erro na API ({response.status_code}): {msg}")

    try:
        return out["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(out["error"])
        return str(out).strip()


def main():
    parser = argparse.ArgumentParser(description="Roteiro de aula via Hugging Face API (Qwen)")
    parser.add_argument("tema", type=str, help="Tema ou título da aula")
    parser.add_argument("--duracao", type=int, default=50, help="Duração em minutos")
    parser.add_argument(
        "--nivel",
        type=str,
        default="ensino_medio",
        choices=["fundamental_i", "fundamental_ii", "ensino_medio", "superior"],
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token do Hugging Face (ou use HUGGING_FACE_HUB_TOKEN / HF_TOKEN)",
    )
    args = parser.parse_args()

    roteiro = gerar_roteiro_api(
        args.tema,
        duracao_min=args.duracao,
        nivel=args.nivel,
        token=args.token,
    )
    print("\n" + "=" * 60 + "\n")
    print(roteiro)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
