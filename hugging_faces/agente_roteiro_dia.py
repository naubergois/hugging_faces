#!/usr/bin/env python3
"""
Agente que consulta clima e eventos em Fortaleza e gera um roteiro do dia.
- Ferramentas: clima (wttr.in) e eventos (site Viver Fortal).
- LLM: Hugging Face / Qwen (mesma API do roteiro de aula).
Fluxo tipo agente: executar ferramentas → montar contexto → gerar roteiro com o modelo.

Uso: python agente_roteiro_dia.py
"""

import os
import re
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Router Hugging Face (mesmo do roteiro de aula)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
WTTR_URL = "https://wttr.in/Fortaleza?format=j1"
EVENTOS_URL = "https://www.viverfortal.com.br/"


def buscar_clima_fortaleza() -> str:
    """Acessa wttr.in e retorna o clima atual e previsão do dia em Fortaleza."""
    try:
        r = requests.get(WTTR_URL, headers={"User-Agent": "curl/7.64"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        parts = []

        # Condição atual
        cc = data.get("current_condition", [{}])[0]
        if cc:
            temp = cc.get("temp_C", "?")
            desc = cc.get("weatherDesc", [{}])[0].get("value", "?")
            umid = cc.get("humidity", "?")
            sens = cc.get("FeelsLikeC", "?")
            parts.append(
                f"Condição atual: {desc}, temperatura {temp}°C (sensação {sens}°C), umidade {umid}%."
            )

        # Previsão do dia (primeiro dia em "weather")
        weather = data.get("weather", [])
        if weather:
            dia = weather[0]
            temp_max = dia.get("maxtempC", "?")
            temp_min = dia.get("mintempC", "?")
            parts.append(f"Previsão do dia: mínima {temp_min}°C, máxima {temp_max}°C.")
            astronomy = dia.get("astronomy", [{}])[0]
            if astronomy:
                nascer = astronomy.get("sunrise", "?")
                por = astronomy.get("sunset", "?")
                parts.append(f"Nascer do sol: {nascer}. Pôr do sol: {por}.")

        return " ".join(parts) if parts else "Clima em Fortaleza não disponível no momento."
    except Exception as e:
        return f"Erro ao obter clima: {e}. Sugestão: considere sol e calor típicos de Fortaleza."


def buscar_eventos_fortaleza() -> str:
    """Tenta obter eventos/agenda em Fortaleza a partir de site público."""
    try:
        r = requests.get(
            EVENTOS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"},
            timeout=15,
        )
        r.raise_for_status()
        text = r.text

        # Extrair trechos que parecem eventos (títulos, datas, links)
        # Remover HTML básico para reduzir ruído
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.I)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:4000]  # limite para não estourar contexto

        if len(text) > 200:
            return (
                "Informações extraídas do site Viver Fortal (Fortaleza):\n"
                + text[:2500]
                + "\n\n(Consulte https://www.viverfortal.com.br/ para a agenda completa e atualizada.)"
            )
        return (
            "Agenda do dia: consulte https://www.viverfortal.com.br/ para eventos atuais em Fortaleza. "
            "Sugestões típicas: praias (Iracema, Meireles), Mercado dos Pinhões, Centro Dragão do Mar, feirinhas e restaurantes na orla."
        )
    except Exception as e:
        return (
            f"Não foi possível acessar a agenda online ({e}). "
            "Sugestões gerais: praias, Centro Dragão do Mar, Mercado dos Pinhões, feirinhas e restaurantes na orla."
        )


def gerar_roteiro_com_llm(clima: str, eventos: str, token: Optional[str] = None) -> str:
    """Envia clima e eventos ao LLM e retorna o roteiro do dia."""
    token = (
        token
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not token:
        raise ValueError(
            "Defina HUGGING_FACE_HUB_TOKEN ou HF_TOKEN no .env. "
            "Obtenha em: https://huggingface.co/settings/tokens"
        )

    hoje = datetime.now().strftime("%d/%m/%Y")
    prompt = f"""Você é um guia local de Fortaleza. Com base nas informações abaixo, elabore um **roteiro sugerido para o dia** ({hoje}), em tópicos claros e objetivos.

**Clima em Fortaleza hoje:**
{clima}

**Eventos / agenda / dicas:**
{eventos}

O roteiro deve incluir:
1. Manhã (sugestões de horário e atividade)
2. Tarde
3. Noite
4. Dicas práticas (proteção solar, transporte, etc.) quando fizer sentido pelo clima

Seja conciso e útil. Responda em português."""

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "temperature": 0.6,
        "stream": False,
    }

    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=90)
    try:
        out = response.json()
    except Exception:
        out = {}

    if not response.ok:
        err = out.get("error", out.get("message", response.text or f"HTTP {response.status_code}"))
        msg = err.get("message", err.get("error", str(err))) if isinstance(err, dict) else str(err)
        if response.status_code == 401:
            raise RuntimeError("Token inválido. Verifique HUGGING_FACE_HUB_TOKEN no .env.")
        if response.status_code == 429:
            raise RuntimeError("Muitas requisições. Aguarde e tente novamente.")
        raise RuntimeError(f"Erro na API ({response.status_code}): {msg}")

    try:
        return out["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(out.get("error", str(out)))
        return str(out).strip()


def executar_agente_roteiro_dia(token: Optional[str] = None) -> str:
    """
    Executa o agente: busca clima e eventos em Fortaleza e gera o roteiro do dia.
    Retorna o texto do roteiro.
    """
    clima = buscar_clima_fortaleza()
    eventos = buscar_eventos_fortaleza()
    return gerar_roteiro_com_llm(clima=clima, eventos=eventos, token=token)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Roteiro do dia em Fortaleza (clima + eventos)")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    print("Buscando clima e eventos em Fortaleza…\n")
    try:
        roteiro = executar_agente_roteiro_dia(token=args.token)
        print("=" * 60)
        print("ROTEIRO DO DIA – FORTALEZA")
        print("=" * 60)
        print(roteiro)
        print("=" * 60)
    except (ValueError, RuntimeError) as e:
        print(f"Erro: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
