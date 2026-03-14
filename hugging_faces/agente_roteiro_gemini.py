#!/usr/bin/env python3
"""
Agente com LangChain e Google Gemini usando tools.
O modelo decide quando chamar as ferramentas (clima e eventos) e elabora o roteiro do dia.

Requer: pip install langchain-core langchain-google-genai
Chave no .env: GEMINI_API_KEY ou GOOGLE_API_KEY (obtenha em https://ai.google.dev/)

Uso: python agente_roteiro_gemini.py
"""

import os
import re
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

WTTR_URL = "https://wttr.in/Fortaleza?format=j1"
EVENTOS_URL = "https://www.viverfortal.com.br/"
GEMINI_MODEL = "gemini-2.5-flash"


# --- Ferramentas LangChain (@tool) ---

@tool
def buscar_clima_fortaleza() -> str:
    """Busca o clima atual e a previsão do dia em Fortaleza. Use quando precisar de temperatura, condições do tempo, nascer e pôr do sol para montar o roteiro."""
    try:
        r = requests.get(WTTR_URL, headers={"User-Agent": "curl/7.64"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        parts = []

        cc = data.get("current_condition", [{}])[0]
        if cc:
            temp = cc.get("temp_C", "?")
            desc = cc.get("weatherDesc", [{}])[0].get("value", "?")
            umid = cc.get("humidity", "?")
            sens = cc.get("FeelsLikeC", "?")
            parts.append(
                f"Condição atual: {desc}, temperatura {temp}°C (sensação {sens}°C), umidade {umid}%."
            )

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


@tool
def buscar_eventos_fortaleza() -> str:
    """Busca eventos e agenda do dia em Fortaleza (shows, feiras, atrações). Use para sugerir atividades no roteiro."""
    try:
        r = requests.get(
            EVENTOS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"},
            timeout=15,
        )
        r.raise_for_status()
        text = r.text
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.I)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()[:4000]

        if len(text) > 200:
            return (
                "Informações do site Viver Fortal (Fortaleza):\n"
                + text[:2500]
                + "\n\n(Consulte https://www.viverfortal.com.br/ para a agenda completa.)"
            )
        return (
            "Agenda: consulte https://www.viverfortal.com.br/. "
            "Sugestões: praias (Iracema, Meireles), Mercado dos Pinhões, Dragão do Mar, feirinhas e orla."
        )
    except Exception as e:
        return (
            f"Não foi possível acessar a agenda ({e}). "
            "Sugestões: praias, Dragão do Mar, Mercado dos Pinhões, feirinhas e orla."
        )


def criar_llm_gemini(api_key: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Cria o LLM Gemini via LangChain. Lê a chave do .env (GEMINI_API_KEY ou GOOGLE_API_KEY)."""
    load_dotenv()
    key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise ValueError(
            "Defina GEMINI_API_KEY ou GOOGLE_API_KEY no arquivo .env ou passe api_key. "
            "Obtenha em: https://ai.google.dev/"
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=key,
        temperature=0.6,
    )


def _content_to_str(content) -> str:
    """Converte content do AIMessage (str ou lista de blocos) em string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return " ".join(parts).strip()
    return str(content).strip()


def executar_agente_roteiro_gemini(api_key: Optional[str] = None) -> str:
    """
    Executa o agente com tools: o Gemini decide chamar clima e/ou eventos
    e depois elabora o roteiro do dia em Fortaleza.
    """
    llm = criar_llm_gemini(api_key=api_key)
    tools = [buscar_clima_fortaleza, buscar_eventos_fortaleza]
    llm_com_tools = llm.bind_tools(tools)

    hoje = datetime.now().strftime("%d/%m/%Y")
    system = (
        "Você é um guia local de Fortaleza. Sua tarefa é elaborar um roteiro sugerido para o dia. "
        "Use as ferramentas disponíveis para obter o clima atual e os eventos/agenda em Fortaleza. "
        "Depois, com essas informações, escreva um roteiro em português com: 1) Manhã, 2) Tarde, 3) Noite, 4) Dicas práticas. Seja conciso e útil."
    )
    user_msg = f"Elabore um roteiro para o dia {hoje} em Fortaleza. Use as ferramentas para obter clima e eventos e depois escreva o roteiro."

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ]
    max_rounds = 5

    for _ in range(max_rounds):
        response = llm_com_tools.invoke(messages)

        if not getattr(response, "tool_calls", None):
            return _content_to_str(getattr(response, "content", None) or response)

        messages.append(response)

        for tc in response.tool_calls:
            tool_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
            tool_id = getattr(tc, "id", None) or (tc.get("id", "") if isinstance(tc, dict) else "")
            tool_args = getattr(tc, "args", None) or (tc.get("args", {}) if isinstance(tc, dict) else {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            tool_map = {t.name: t for t in tools}
            if tool_name not in tool_map:
                messages.append(
                    ToolMessage(content=f"Ferramenta '{tool_name}' não encontrada.", tool_call_id=tool_id)
                )
                continue
            try:
                result = tool_map[tool_name].invoke(tool_args)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            except Exception as e:
                messages.append(
                    ToolMessage(content=f"Erro ao executar ferramenta: {e}", tool_call_id=tool_id)
                )

    return _content_to_str(getattr(response, "content", None)) or "Não foi possível concluir o roteiro. Tente novamente."


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Roteiro do dia em Fortaleza (LangChain + Gemini + tools)")
    parser.add_argument("--api-key", type=str, default=None, help="Chave Gemini (ou defina GEMINI_API_KEY no .env)")
    args = parser.parse_args()

    print("Executando agente (Gemini com tools)…\n")
    try:
        roteiro = executar_agente_roteiro_gemini(api_key=args.api_key)
        print("=" * 60)
        print("ROTEIRO DO DIA – FORTALEZA (LangChain + Gemini + tools)")
        print("=" * 60)
        print(roteiro)
        print("=" * 60)
    except (ValueError, RuntimeError) as e:
        print(f"Erro: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
