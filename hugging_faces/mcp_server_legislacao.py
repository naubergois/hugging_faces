#!/usr/bin/env python3
"""
Servidor MCP: consulta à legislação (RAG) + skills extensíveis.
Expõe ferramentas para agentes: consultar_legislacao, listar_skills, registrar_skill, executar_skill.

Requer: pip install mcp
Execute: python mcp_server_legislacao.py
Ou com transporte HTTP: python mcp_server_legislacao.py --http 8000
"""

import json
import os
import sys
from pathlib import Path

# Garantir que o diretório do projeto está no path para importar legislacao_rag
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP(
    "Legislação e Skills",
    json_response=True,
)

# Arquivo onde as skills registradas são salvas
SKILLS_FILE = Path(__file__).resolve().parent / "mcp_skills.json"


def _load_skills() -> dict:
    """Carrega skills do arquivo JSON."""
    if not SKILLS_FILE.exists():
        return {}
    try:
        with open(SKILLS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_skills(skills: dict) -> None:
    """Salva skills no arquivo JSON."""
    with open(SKILLS_FILE, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)


# ---------- Tool: Consultar legislação (RAG) ----------


@mcp.tool()
def consultar_legislacao(pergunta: str) -> str:
    """
    Consulta a base de legislação simulada (5 leis) e retorna uma resposta com base nos trechos relevantes.
    Use para perguntas sobre teletrabalho, férias, proteção de dados, licenças, horas extras.
    """
    try:
        from legislacao_rag import garantir_chroma_carregado, perguntar_legislacao
        garantir_chroma_carregado()
        resultado = perguntar_legislacao(pergunta.strip())
        resposta = resultado.get("resposta", "")
        contexto = resultado.get("contexto", "")
        if contexto:
            return f"{resposta}\n\n[Trechos utilizados]\n{contexto[:1500]}{'...' if len(contexto) > 1500 else ''}"
        return resposta
    except Exception as e:
        return f"Erro ao consultar legislação: {e}"


# ---------- Tools: Skills (registrar e usar novas habilidades) ----------


@mcp.tool()
def listar_skills() -> str:
    """
    Lista todas as skills registradas (habilidades que o agente pode usar).
    Retorna nome, descrição e instruções de cada skill.
    """
    skills = _load_skills()
    if not skills:
        return "Nenhuma skill registrada. Use registrar_skill para adicionar."
    linhas = []
    for nome, dados in skills.items():
        desc = dados.get("descricao", "")
        inst = dados.get("instrucoes", "")[:200]
        linhas.append(f"- **{nome}**: {desc}\n  Instruções: {inst}...")
    return "\n\n".join(linhas)


@mcp.tool()
def registrar_skill(nome: str, descricao: str, instrucoes: str) -> str:
    """
    Registra uma nova skill (habilidade) para o agente.
    nome: identificador único da skill (ex: resumir_artigo).
    descricao: o que a skill faz, em uma frase.
    instrucoes: como executar (ex: 'Resuma o texto em até 3 tópicos objetivos').
    """
    nome = nome.strip().lower().replace(" ", "_")
    if not nome:
        return "Nome da skill não pode ser vazio."
    skills = _load_skills()
    skills[nome] = {
        "descricao": descricao.strip(),
        "instrucoes": instrucoes.strip(),
    }
    _save_skills(skills)
    return f"Skill '{nome}' registrada com sucesso."


@mcp.tool()
def executar_skill(nome: str, entrada: str) -> str:
    """
    Executa uma skill registrada com a entrada fornecida.
    nome: nome da skill (use listar_skills para ver as disponíveis).
    entrada: texto ou dados que a skill deve processar.
    """
    skills = _load_skills()
    if nome not in skills:
        return f"Skill '{nome}' não encontrada. Use listar_skills para ver as disponíveis."
    dados = skills[nome]
    instrucoes = dados.get("instrucoes", "")
    if not instrucoes:
        return "Skill sem instruções definidas."
    # Executar via LLM (Gemini) para flexibilidade
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            return "Configure GEMINI_API_KEY no .env para executar skills."
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=key,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Siga exatamente as instruções abaixo. Responda apenas com o resultado solicitado, em português."),
            ("human", "Instruções: {instrucoes}\n\nEntrada:\n{entrada}\n\nResultado:"),
        ])
        chain = prompt | llm
        resp = chain.invoke({"instrucoes": instrucoes, "entrada": entrada})
        texto = resp.content if hasattr(resp, "content") else str(resp)
        return texto.strip()
    except Exception as e:
        return f"Erro ao executar skill: {e}"


# ---------- Entrypoint ----------


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCP Server: Legislação + Skills")
    parser.add_argument("--http", type=int, default=None, help="Porta HTTP (ex: 8000). Se omitido, usa stdio.")
    args = parser.parse_args()
    if args.http is not None:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.http)
    else:
        mcp.run(transport="stdio")
