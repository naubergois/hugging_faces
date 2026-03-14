#!/usr/bin/env python3
"""
Cliente MCP para o servidor de legislação e skills.
Permite consultar legislação, listar/registrar/executar skills via MCP.

Requer: servidor rodando (python mcp_server_legislacao.py) ou usa subprocess stdio.
"""

import asyncio
import os
import sys
from pathlib import Path

# Diretório do projeto
PROJECT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = PROJECT_DIR / "mcp_server_legislacao.py"


async def _run_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Executa uma tool no servidor MCP via stdio (subprocess)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            content = result.content
            if content and len(content) > 0:
                part = content[0]
                if hasattr(part, "text"):
                    return part.text
                if isinstance(part, dict) and "text" in part:
                    return part["text"]
            return str(result)


def _sync_run(coro):
    """Roda uma coroutine em um novo event loop (para uso síncrono no Streamlit)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def consultar_legislacao_mcp(pergunta: str) -> str:
    """Consulta a legislação via servidor MCP. Retorna a resposta (texto)."""
    return _sync_run(_run_mcp_tool("consultar_legislacao", {"pergunta": pergunta}))


def listar_skills_mcp() -> str:
    """Lista as skills registradas via MCP."""
    return _sync_run(_run_mcp_tool("listar_skills", {}))


def registrar_skill_mcp(nome: str, descricao: str, instrucoes: str) -> str:
    """Registra uma nova skill via MCP."""
    return _sync_run(_run_mcp_tool("registrar_skill", {
        "nome": nome,
        "descricao": descricao,
        "instrucoes": instrucoes,
    }))


def executar_skill_mcp(nome: str, entrada: str) -> str:
    """Executa uma skill registrada via MCP."""
    return _sync_run(_run_mcp_tool("executar_skill", {"nome": nome, "entrada": entrada}))


if __name__ == "__main__":
    # Teste rápido
    print("Listando skills:", listar_skills_mcp()[:200])
    print("Consultando:", consultar_legislacao_mcp("Quantos dias de férias?")[:300])
