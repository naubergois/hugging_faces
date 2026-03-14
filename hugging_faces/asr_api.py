#!/usr/bin/env python3
"""
Transcrição de áudio (ASR) via API Hugging Face (router).
Requer áudio em bytes (WAV, FLAC, MP3, etc.).

Uso: python asr_api.py arquivo_audio.wav
"""

import argparse
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Modelo de reconhecimento de fala (multilíngue)
ASR_MODEL_ID = "openai/whisper-large-v3"
ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{ASR_MODEL_ID}"


def _detectar_tipo_audio(data: bytes) -> str:
    """Retorna Content-Type aceito pela API a partir dos primeiros bytes."""
    if len(data) < 12:
        return "audio/wav"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:3] == b"ID3" or (data[:2] == b"\xff\xfb") or (data[:2] == b"\xff\xfa"):
        return "audio/mpeg"
    if data[:4] == b"OggS":
        return "audio/ogg"
    return "audio/wav"


def transcrever_audio_api(
    audio_bytes: bytes,
    token: Optional[str] = None,
) -> str:
    """
    Transcreve áudio para texto usando a API do Hugging Face (Whisper).

    Args:
        audio_bytes: Conteúdo binário do áudio (WAV, FLAC, MP3, etc.).
        token: Token do Hugging Face (ou use HUGGING_FACE_HUB_TOKEN / HF_TOKEN).

    Returns:
        Texto transcrito.
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
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("O áudio está vazio.")

    # Enviar áudio em binário com Content-Type correto (mic/upload costumam ser WAV)
    content_type = _detectar_tipo_audio(audio_bytes)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
        "x-task": "automatic-speech-recognition",
    }
    response = requests.post(
        ROUTER_URL,
        headers=headers,
        data=audio_bytes,
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
            raise RuntimeError("Muitas requisições. Aguarde um pouco e tente novamente.")
        if response.status_code == 503:
            if "loading" in str(msg).lower():
                raise RuntimeError(
                    "Modelo ainda está carregando na API. Aguarde e tente de novo."
                )
            raise RuntimeError(f"API temporariamente indisponível: {msg}")
        raise RuntimeError(f"Erro na API ({response.status_code}): {msg}")

    # Resposta ASR pode ser dict com "text" ou lista de chunks
    if isinstance(out, dict):
        return (out.get("text") or out.get("transcription") or "").strip()
    if isinstance(out, list) and len(out) > 0:
        first = out[0]
        if isinstance(first, dict):
            return (first.get("text") or first.get("transcription") or "").strip()
        return str(first).strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description="Transcrição de áudio via Hugging Face API (Whisper)")
    parser.add_argument("audio", type=str, help="Caminho do arquivo de áudio (WAV, FLAC, MP3)")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    path = args.audio
    if not os.path.isfile(path):
        raise SystemExit(f"Arquivo não encontrado: {path}")

    with open(path, "rb") as f:
        audio_bytes = f.read()

    texto = transcrever_audio_api(audio_bytes, token=args.token)
    print("\nTranscrição:\n", texto, "\n")


if __name__ == "__main__":
    main()
