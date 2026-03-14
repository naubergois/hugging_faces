#!/usr/bin/env python3
"""
Três agentes de notícias orquestrados com LangGraph:
1. Agente Buscar: busca notícias no site UOL (RSS), retorna dados estruturados (Pydantic/JSON).
2. Agente Classificar: analisa e classifica as notícias por perfil (política, esportes, etc.).
3. Agente Resumir: gera o resumo do dia.

Requer: pip install langchain-core langchain-google-genai langgraph pydantic requests
Chave no .env: GEMINI_API_KEY ou GOOGLE_API_KEY

Uso: python agentes_noticias.py
"""

import json
import os
from typing import Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

UOL_RSS_URL = "https://rss.uol.com.br/feed/noticias.xml"
UOL_RSS_ALT = "https://rss.home.uol.com.br/noticias"
GEMINI_MODEL = "gemini-2.5-flash"


# ---------- Modelos Pydantic (dados estruturados) ----------


class Noticia(BaseModel):
    """Uma notícia em formato estruturado."""

    titulo: str = Field(description="Título da notícia")
    link: str = Field(description="URL da notícia")
    resumo: str = Field(default="", description="Resumo ou descrição curta")
    data_publicacao: str = Field(default="", description="Data/hora de publicação")
    fonte: str = Field(default="UOL", description="Fonte da notícia")


class ListaNoticias(BaseModel):
    """Lista de notícias para saída estruturada do agente 1 (quando refinado por LLM)."""

    noticias: list[Noticia] = Field(description="Lista de notícias")


class NoticiaClassificada(BaseModel):
    """Notícia com perfil/categoria atribuído."""

    titulo: str = Field(description="Título da notícia")
    link: str = Field(description="URL")
    resumo: str = Field(default="", description="Resumo")
    perfil: str = Field(
        description="Perfil/categoria: um de politica, economia, esportes, entretenimento, tecnologia, cotidiano, internacional, outros"
    )


class ListaNoticiasClassificadas(BaseModel):
    """Lista de notícias classificadas por perfil."""

    noticias: list[NoticiaClassificada] = Field(description="Notícias com perfil atribuído")


class ResumoDoDia(BaseModel):
    """Resumo do dia gerado pelo agente 3."""

    resumo_geral: str = Field(description="Resumo em um ou dois parágrafos das principais notícias do dia")
    destaques: list[str] = Field(description="Até 5 destaques em tópicos")
    perfis_abordados: list[str] = Field(default_factory=list, description="Perfis que aparecem no resumo")


# ---------- Agente 1: Buscar notícias no UOL (LLM faz o parse para Pydantic) ----------


def _fetch_rss_uol(url: str = UOL_RSS_URL) -> str:
    """Baixa o feed RSS do UOL e retorna o XML como string."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}
    for u in (url, UOL_RSS_ALT):
        try:
            r = requests.get(u, headers=headers, timeout=15)
            r.raise_for_status()
            return r.text
        except Exception:
            continue
    raise RuntimeError("Não foi possível acessar o RSS do UOL.")


def agente_buscar_noticias() -> list[Noticia]:
    """
    Agente 1: busca notícias no site UOL (RSS bruto) e usa o LLM para extrair
    e estruturar cada notícia em formato Pydantic/JSON (sem parser manual).
    """
    raw_rss = _fetch_rss_uol()
    # Limitar tamanho para caber no contexto do modelo (~30k chars ≈ primeiros itens)
    conteudo = raw_rss[:35_000] if len(raw_rss) > 35_000 else raw_rss

    llm = _criar_llm()
    parser = PydanticOutputParser(pydantic_object=ListaNoticias)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um extrator de notícias. Recebe um feed RSS (XML) do site UOL e deve extrair todas as notícias no formato estruturado solicitado. "
            "Para cada <item> extraia: titulo, link, resumo (description sem HTML), data_publicacao (pubDate ou similar), fonte sempre 'UOL'. "
            "Resposta apenas no formato JSON esperado, sem texto adicional.",
        ),
        (
            "human",
            "Extraia todas as notícias do feed RSS abaixo e retorne na estrutura solicitada.\n\n"
            "Conteúdo do feed:\n{conteudo}\n\n{format_instructions}",
        ),
    ])
    chain = prompt | llm | parser
    result: ListaNoticias = chain.invoke({
        "conteudo": conteudo,
        "format_instructions": parser.get_format_instructions(),
    })
    return result.noticias[:20]  # limite 20 notícias


# ---------- Agente 2: Classificar por perfil (LangChain + Pydantic) ----------


def _criar_llm():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no .env")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=key,
        temperature=0.3,
    )


def agente_classificar_noticias(noticias: list[Noticia]) -> list[NoticiaClassificada]:
    """
    Agente 2: analisa as notícias e classifica cada uma por perfil (política, esportes, etc.).
    """
    if not noticias:
        return []
    llm = _criar_llm()
    parser = PydanticOutputParser(pydantic_object=ListaNoticiasClassificadas)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você classifica notícias em um único perfil por notícia. Perfis válidos: politica, economia, esportes, entretenimento, tecnologia, cotidiano, internacional, outros. Responda apenas com o JSON esperado."),
        ("human", "Classifique cada notícia abaixo atribuindo exatamente um perfil.\n\n{texto}\n\n{format_instructions}"),
    ])
    texto = "\n".join(
        f"- {n.titulo} | {n.resumo[:200] if n.resumo else 'Sem resumo'}"
        for n in noticias
    )
    chain = prompt | llm | parser
    result: ListaNoticiasClassificadas = chain.invoke({
        "texto": texto,
        "format_instructions": parser.get_format_instructions(),
    })
    # Garantir que links e resumos originais sejam preservados
    by_titulo = {n.titulo: n for n in noticias}
    out: list[NoticiaClassificada] = []
    for nc in result.noticias:
        orig = by_titulo.get(nc.titulo)
        if orig:
            out.append(NoticiaClassificada(titulo=nc.titulo, link=orig.link, resumo=orig.resumo or nc.resumo, perfil=nc.perfil))
        else:
            out.append(nc)
    return out


# ---------- Agente 3: Resumir do dia (LangChain + Pydantic) ----------


def agente_resumir_do_dia(noticias_classificadas: list[NoticiaClassificada]) -> ResumoDoDia:
    """
    Agente 3: gera o resumo do dia a partir das notícias classificadas.
    """
    if not noticias_classificadas:
        return ResumoDoDia(
            resumo_geral="Nenhuma notícia disponível para resumir.",
            destaques=[],
            perfis_abordados=[],
        )
    llm = _criar_llm()
    parser = PydanticOutputParser(pydantic_object=ResumoDoDia)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um editor que produz o resumo do dia a partir das notícias. Seja objetivo e claro. Até 5 destaques."),
        ("human", "Com base nas notícias abaixo, elabore o resumo do dia em português.\n\n{texto}\n\n{format_instructions}"),
    ])
    texto = "\n".join(
        f"[{n.perfil}] {n.titulo}: {n.resumo[:150] if n.resumo else ''}"
        for n in noticias_classificadas
    )
    chain = prompt | llm | parser
    return chain.invoke({
        "texto": texto,
        "format_instructions": parser.get_format_instructions(),
    })


# ---------- Estado do LangGraph ----------


class EstadoGrafo(TypedDict, total=False):
    """Estado do grafo LangGraph (mutável)."""
    noticias: list[Noticia]
    noticias_classificadas: list[NoticiaClassificada]
    resumo_do_dia: Optional[ResumoDoDia]
    erro: Optional[str]


# ---------- Nós do grafo ----------


def no_buscar(state: EstadoGrafo) -> EstadoGrafo:
    try:
        noticias = agente_buscar_noticias()
        return {**state, "noticias": noticias, "erro": None}
    except Exception as e:
        return {**state, "noticias": [], "erro": str(e)}


def no_classificar(state: EstadoGrafo) -> EstadoGrafo:
    if state.get("erro") or not state.get("noticias"):
        return state
    try:
        classificadas = agente_classificar_noticias(state["noticias"])
        return {**state, "noticias_classificadas": classificadas, "erro": None}
    except Exception as e:
        return {**state, "erro": str(e)}


def no_resumir(state: EstadoGrafo) -> EstadoGrafo:
    if state.get("erro"):
        return state
    noticias_cl = state.get("noticias_classificadas") or []
    try:
        resumo = agente_resumir_do_dia(noticias_cl)
        return {**state, "resumo_do_dia": resumo, "erro": None}
    except Exception as e:
        return {**state, "erro": str(e)}


# ---------- Montagem do LangGraph ----------


def criar_grafo_noticias() -> StateGraph:
    grafo = StateGraph(EstadoGrafo)
    grafo.add_node("buscar", no_buscar)
    grafo.add_node("classificar", no_classificar)
    grafo.add_node("resumir", no_resumir)
    grafo.set_entry_point("buscar")
    grafo.add_edge("buscar", "classificar")
    grafo.add_edge("classificar", "resumir")
    grafo.add_edge("resumir", END)
    return grafo


def executar_pipeline_noticias() -> dict:
    """
    Orquestra os 3 agentes com LangGraph e retorna o estado final
    (notícias, notícias classificadas, resumo do dia).
    """
    comp = criar_grafo_noticias().compile()
    estado_inicial: EstadoGrafo = {
        "noticias": [],
        "noticias_classificadas": [],
        "resumo_do_dia": None,
        "erro": None,
    }
    final = comp.invoke(estado_inicial)
    return final


# ---------- CLI ----------


def main():
    print("Orquestrando agentes de notícias (LangGraph): Buscar → Classificar → Resumir\n")
    try:
        resultado = executar_pipeline_noticias()
    except ValueError as e:
        print(f"Configuração: {e}")
        return
    if resultado.get("erro"):
        print("Erro:", resultado["erro"])
        return
    noticias = resultado.get("noticias") or []
    classificadas = resultado.get("noticias_classificadas") or []
    resumo = resultado.get("resumo_do_dia")
    print("=" * 60)
    print("NOTÍCIAS BUSCADAS (UOL) – formato estruturado (JSON)")
    print("=" * 60)
    print(json.dumps([n.model_dump() for n in noticias], ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    print("NOTÍCIAS CLASSIFICADAS POR PERFIL")
    print("=" * 60)
    for n in classificadas:
        print(f"  [{n.perfil}] {n.titulo}")
    print("\n" + "=" * 60)
    print("RESUMO DO DIA")
    print("=" * 60)
    if resumo:
        print(resumo.resumo_geral)
        print("\nDestaques:")
        for d in resumo.destaques:
            print("  -", d)
    else:
        print("(Nenhum resumo gerado.)")


if __name__ == "__main__":
    main()
