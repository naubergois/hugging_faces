#!/usr/bin/env python3
"""
Detecção de objetos via API de Inferência do Hugging Face (router/nova API).
Usa huggingface_hub.InferenceClient, que já aponta para a API correta.

Uso: python detector_objetos_api.py caminho/para/imagem.jpg
"""

import argparse
import base64
import os
from typing import List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Nova API (api-inference descontinuada 410). Router com backend HF.
MODEL_ID = "facebook/detr-resnet-50"
ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"


def _parse_detections(raw: list, threshold: float) -> List[dict]:
    out: List[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        score = item.get("score", 0)
        if score < threshold:
            continue
        label = item.get("label", "?")
        b = item.get("box")
        if isinstance(b, dict):
            box = b
        else:
            box = {}
        out.append({"label": label, "score": float(score), "box": box})
    return out


def detectar_objetos_api(
    image_bytes: bytes,
    threshold: float = 0.8,
    token: Optional[str] = None,
) -> List[dict]:
    """
    Detecta objetos na imagem usando a API do Hugging Face (router).

    Args:
        image_bytes: Conteúdo binário da imagem (JPEG, PNG, etc.).
        threshold: Confiança mínima (0–1) para incluir a detecção. Default 0.8.
        token: Token do Hugging Face (ou use HUGGING_FACE_HUB_TOKEN / HF_TOKEN).

    Returns:
        Lista de dicts com keys: label, score, box (xmin, ymin, xmax, ymax).
    """
    token = (
        token
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not token:
        raise ValueError(
            "Defina HUGGING_FACE_HUB_TOKEN ou HF_TOKEN no .env ou no ambiente. "
            "Obtenha em: https://huggingface.co/settings/tokens"
        )

    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("A imagem está vazia.")

    # Router espera JSON com imagem em base64 (evita erro NoneType no servidor)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {"inputs": image_b64}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-task": "object-detection",
    }
    response = requests.post(
        ROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    try:
        out = response.json()
    except Exception:
        out = {}

    if not response.ok:
        err = out.get("error", out.get("message", response.text or f"HTTP {response.status_code}"))
        if isinstance(err, dict):
            msg = err.get("message", err.get("error", str(err)))
        else:
            msg = str(err)
        if response.status_code == 401:
            raise RuntimeError(
                "Token inválido ou não autorizado. Verifique o HUGGING_FACE_HUB_TOKEN no .env."
            )
        if response.status_code == 403:
            raise RuntimeError(
                "Acesso negado à API. Confira se o token tem permissão 'Make calls to Inference Providers'."
            )
        if response.status_code == 429:
            raise RuntimeError(
                "Muitas requisições. Aguarde um pouco e tente novamente."
            )
        if response.status_code == 503:
            if "loading" in str(msg).lower():
                raise RuntimeError(
                    "Modelo ainda está carregando na API. Aguarde ~20s e tente de novo."
                )
            raise RuntimeError(f"API temporariamente indisponível: {msg}")
        raise RuntimeError(f"Erro na API ({response.status_code}): {msg}")

    if isinstance(out, list):
        return _parse_detections(out, threshold)
    if isinstance(out, dict) and "error" in out:
        raise RuntimeError(out.get("error", str(out)))
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Detecção de objetos via Hugging Face API (DETR)"
    )
    parser.add_argument("imagem", type=str, help="Caminho da imagem (JPEG/PNG)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Confiança mínima (0–1) para exibir detecções (default: 0.8)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token do Hugging Face (ou use HUGGING_FACE_HUB_TOKEN / HF_TOKEN)",
    )
    args = parser.parse_args()

    path = args.imagem
    if not os.path.isfile(path):
        raise SystemExit(f"Arquivo não encontrado: {path}")

    with open(path, "rb") as f:
        image_bytes = f.read()

    deteccoes = detectar_objetos_api(
        image_bytes,
        threshold=args.threshold,
        token=args.token,
    )

    print(f"\n{len(deteccoes)} objeto(s) detectado(s) (threshold={args.threshold}):\n")
    for i, d in enumerate(deteccoes, 1):
        label = d.get("label", "?")
        score = d.get("score", 0)
        box = d.get("box", {})
        print(f"  {i}. {label} — confiança: {score:.2%} — box: {box}")
    print()


if __name__ == "__main__":
    main()
