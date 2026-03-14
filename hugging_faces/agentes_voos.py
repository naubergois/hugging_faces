#!/usr/bin/env python3
"""
Quatro agentes de voos orquestrados com LangGraph:
1. Agente Voos: obtém voos que saem de Fortaleza com preço (Pydantic).
2. Agente Custo-Benefício: analisa custo-benefício por perfil da população.
3. Agente Marketing: cria campanha para os voos mais promissores.
4. Agente Distribuição: envia a campanha aos e-mails cadastrados.

Requer: pip install langchain-core langchain-google-genai langgraph pydantic
Chave no .env: GEMINI_API_KEY ou GOOGLE_API_KEY

Uso: python agentes_voos.py
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
ORIGEM_PADRAO = "FOR"  # Fortaleza


# ---------- Modelos Pydantic ----------


class Voo(BaseModel):
    """Um voo com preço e dados essenciais."""

    origem: str = Field(description="Código IATA da origem (ex: FOR)")
    destino: str = Field(description="Código IATA do destino (ex: GRU)")
    destino_nome: str = Field(default="", description="Nome da cidade de destino")
    data_saida: str = Field(description="Data/hora de saída")
    preco_reais: float = Field(description="Preço em reais")
    companhia: str = Field(default="", description="Nome da companhia aérea")
    duracao_minutos: int = Field(default=0, description="Duração do voo em minutos")
    escalas: int = Field(default=0, description="Número de escalas")


class ListaVoos(BaseModel):
    """Lista de voos para saída estruturada."""

    voos: list[Voo] = Field(description="Lista de voos")


class AnalisePerfil(BaseModel):
    """Análise de custo-benefício para um perfil."""

    perfil: str = Field(description="Nome do perfil: familia, executivo, mochileiro, idoso, estudante")
    voo_recomendado_destino: str = Field(description="Destino do voo mais recomendado para este perfil")
    justificativa: str = Field(description="Breve justificativa do custo-benefício")
    nota_custo_beneficio: int = Field(ge=1, le=10, description="Nota de 1 a 10")


class AnaliseCustoBeneficio(BaseModel):
    """Análise de custo-benefício por perfil da população."""

    analises: list[AnalisePerfil] = Field(description="Análise para cada perfil considerado")
    voos_mais_promissores: list[str] = Field(description="Lista de destinos dos voos mais promissores no geral")


class CampanhaMarketing(BaseModel):
    """Campanha de marketing dos voos mais promissores."""

    titulo: str = Field(description="Título da campanha")
    texto_principal: str = Field(description="Texto da campanha (e-mail ou anúncio)")
    destinos_destacados: list[str] = Field(description="Destinos em destaque na campanha")
    chamada_acao: str = Field(description="Chamada para ação (ex: Reserve já)")


class EmailEnviado(BaseModel):
    """Registro de e-mail para qual a campanha foi enviada."""

    email: str = Field(description="Endereço de e-mail")
    status: str = Field(description="enviado ou falha")
    mensagem: str = Field(default="", description="Mensagem de retorno ou erro")


class ResultadoDistribuicao(BaseModel):
    """Resultado da distribuição da campanha por e-mail."""

    total_enviados: int = Field(description="Quantidade de e-mails enviados com sucesso")
    total_falhas: int = Field(description="Quantidade de falhas")
    detalhes: list[EmailEnviado] = Field(description="Detalhe por e-mail")


# ---------- Fonte de dados: voos saindo de Fortaleza (mock) ----------


def _gerar_voos_fortaleza() -> list[Voo]:
    """
    Retorna lista de voos saindo de Fortaleza com preços.
    Em produção, substituir por API real (Amadeus, etc.) ou scraping.
    """
    base = datetime.now() + timedelta(days=7)
    return [
        Voo(
            origem="FOR",
            destino="GRU",
            destino_nome="São Paulo (Guarulhos)",
            data_saida=(base + timedelta(days=1)).strftime("%d/%m/%Y 06:00"),
            preco_reais=450.00,
            companhia="Latam",
            duracao_minutos=195,
            escalas=0,
        ),
        Voo(
            origem="FOR",
            destino="GRU",
            destino_nome="São Paulo (Guarulhos)",
            data_saida=(base + timedelta(days=2)).strftime("%d/%m/%Y 14:30"),
            preco_reais=520.00,
            companhia="Gol",
            duracao_minutos=200,
            escalas=0,
        ),
        Voo(
            origem="FOR",
            destino="CNF",
            destino_nome="Belo Horizonte (Confins)",
            data_saida=(base + timedelta(days=1)).strftime("%d/%m/%Y 10:15"),
            preco_reais=380.00,
            companhia="Azul",
            duracao_minutos=175,
            escalas=0,
        ),
        Voo(
            origem="FOR",
            destino="SSA",
            destino_nome="Salvador",
            data_saida=(base + timedelta(days=3)).strftime("%d/%m/%Y 08:00"),
            preco_reais=290.00,
            companhia="Gol",
            duracao_minutos=95,
            escalas=0,
        ),
        Voo(
            origem="FOR",
            destino="REC",
            destino_nome="Recife",
            data_saida=(base + timedelta(days=1)).strftime("%d/%m/%Y 07:45"),
            preco_reais=220.00,
            companhia="Azul",
            duracao_minutos=75,
            escalas=0,
        ),
        Voo(
            origem="FOR",
            destino="BSB",
            destino_nome="Brasília",
            data_saida=(base + timedelta(days=2)).strftime("%d/%m/%Y 11:00"),
            preco_reais=550.00,
            companhia="Latam",
            duracao_minutos=195,
            escalas=0,
        ),
        Voo(
            origem="FOR",
            destino="GRU",
            destino_nome="São Paulo (Guarulhos)",
            data_saida=(base + timedelta(days=5)).strftime("%d/%m/%Y 06:00"),
            preco_reais=399.00,
            companhia="Azul",
            duracao_minutos=205,
            escalas=1,
        ),
    ]


# ---------- Agente 1: Buscar voos Fortaleza (Pydantic) ----------


def agente_buscar_voos(origem: str = ORIGEM_PADRAO) -> list[Voo]:
    """
    Agente 1: obtém os voos que saem de Fortaleza (ou origem informada) com preço.
    Retorna dados estruturados em Pydantic.
    """
    if origem.upper() != "FOR":
        # Mock só tem FOR; em produção aqui entraria a API real
        return []
    return _gerar_voos_fortaleza()


# ---------- LLM compartilhado ----------


def _criar_llm():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no .env")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=key,
        temperature=0.4,
    )


# ---------- Agente 2: Análise custo-benefício por perfil ----------


def agente_analisar_custo_beneficio(voos: list[Voo]) -> AnaliseCustoBeneficio:
    """
    Agente 2: analisa o custo-benefício dos voos por perfil da população
    (família, executivo, mochileiro, idoso, estudante).
    """
    if not voos:
        return AnaliseCustoBeneficio(analises=[], voos_mais_promissores=[])
    llm = _criar_llm()
    parser = PydanticOutputParser(pydantic_object=AnaliseCustoBeneficio)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um analista de viagens. Analise os voos listados e avalie o custo-benefício para cada perfil: "
            "familia (prioriza preço e horário), executivo (prioriza direto e horário), mochileiro (prioriza preço baixo), "
            "idoso (prioriza conforto e direto), estudante (prioriza preço). Indique o voo mais recomendado por perfil e "
            "liste os destinos dos voos mais promissores no geral. Responda apenas no formato JSON solicitado.",
        ),
        (
            "human",
            "Voos disponíveis (origem Fortaleza - FOR):\n{texto}\n\n{format_instructions}",
        ),
    ])
    texto = "\n".join(
        f"- {v.destino} ({v.destino_nome}): R$ {v.preco_reais:.2f} | {v.companhia} | {v.duracao_minutos} min | {v.escalas} escala(s) | {v.data_saida}"
        for v in voos
    )
    chain = prompt | llm | parser
    return chain.invoke({
        "texto": texto,
        "format_instructions": parser.get_format_instructions(),
    })


# ---------- Agente 3: Campanha de marketing ----------


def agente_campanha_marketing(
    voos: list[Voo],
    analise: AnaliseCustoBeneficio,
) -> CampanhaMarketing:
    """
    Agente 3: cria a campanha de marketing para os voos mais promissores.
    """
    if not voos or not analise.voos_mais_promissores:
        return CampanhaMarketing(
            titulo="Ofertas de voos desde Fortaleza",
            texto_principal="Confira as melhores ofertas de voos saindo de Fortaleza.",
            destinos_destacados=[v.destino_nome or v.destino for v in voos[:3]],
            chamada_acao="Reserve já no nosso site.",
        )
    llm = _criar_llm()
    parser = PydanticOutputParser(pydantic_object=CampanhaMarketing)
    destinos_promissores = ", ".join(analise.voos_mais_promissores)
    texto_voos = "\n".join(
        f"- {v.destino_nome or v.destino}: R$ {v.preco_reais:.2f} ({v.companhia})"
        for v in voos
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você cria campanhas de marketing para viagens. Gere uma campanha curta e atrativa para e-mail, "
            "destacando os voos mais promissores. Inclua título, texto principal, destinos em destaque e uma chamada para ação. "
            "Responda apenas no formato JSON solicitado.",
        ),
        (
            "human",
            "Voos mais promissores (destinos): {destinos}\n\nDados dos voos:\n{voos}\n\n{format_instructions}",
        ),
    ])
    chain = prompt | llm | parser
    return chain.invoke({
        "destinos": destinos_promissores,
        "voos": texto_voos,
        "format_instructions": parser.get_format_instructions(),
    })


# ---------- Agente 4: Distribuir campanha por e-mail ----------


def _obter_emails_cadastrados() -> list[str]:
    """
    Retorna lista de e-mails cadastrados.
    Em produção: ler de .env, banco ou API (ex: EMAILS_CAMPANHA=a@x.com,b@y.com).
    """
    env_emails = os.environ.get("EMAILS_CAMPANHA_VOOS", "").strip()
    if env_emails:
        return [e.strip() for e in env_emails.split(",") if e.strip()]
    # Mock: lista padrão para demonstração
    return [
        "cliente1@exemplo.com",
        "cliente2@exemplo.com",
        "viajante@exemplo.com",
    ]


def agente_distribuir_campanha(
    campanha: CampanhaMarketing,
    emails: Optional[list[str]] = None,
) -> ResultadoDistribuicao:
    """
    Agente 4: distribui a campanha de marketing aos e-mails cadastrados.
    Simula envio; em produção integrar com SendGrid, AWS SES, etc.
    """
    lista = emails or _obter_emails_cadastrados()
    detalhes: list[EmailEnviado] = []
    enviados = 0
    falhas = 0
    for email in lista:
        # Simulação: todos "enviados"; em produção chamar API de e-mail
        try:
            # Aqui seria: send_email(to=email, subject=campanha.titulo, body=campanha.texto_principal)
            detalhes.append(EmailEnviado(email=email, status="enviado", mensagem="OK"))
            enviados += 1
        except Exception as e:
            detalhes.append(EmailEnviado(email=email, status="falha", mensagem=str(e)))
            falhas += 1
    return ResultadoDistribuicao(
        total_enviados=enviados,
        total_falhas=falhas,
        detalhes=detalhes,
    )


# ---------- Estado LangGraph ----------


class EstadoVoos(TypedDict, total=False):
    voos: list[Voo]
    analise: Optional[AnaliseCustoBeneficio]
    campanha: Optional[CampanhaMarketing]
    distribuicao: Optional[ResultadoDistribuicao]
    erro: Optional[str]


# ---------- Nós do grafo ----------


def no_buscar_voos(state: EstadoVoos) -> EstadoVoos:
    try:
        voos = agente_buscar_voos()
        return {**state, "voos": voos, "erro": None}
    except Exception as e:
        return {**state, "voos": [], "erro": str(e)}


def no_analisar(state: EstadoVoos) -> EstadoVoos:
    if state.get("erro") or not state.get("voos"):
        return state
    try:
        analise = agente_analisar_custo_beneficio(state["voos"])
        return {**state, "analise": analise, "erro": None}
    except Exception as e:
        return {**state, "erro": str(e)}


def no_campanha(state: EstadoVoos) -> EstadoVoos:
    if state.get("erro"):
        return state
    voos = state.get("voos") or []
    analise = state.get("analise")
    if not analise:
        return {**state, "erro": "Análise não disponível para criar campanha."}
    try:
        campanha = agente_campanha_marketing(voos, analise)
        return {**state, "campanha": campanha, "erro": None}
    except Exception as e:
        return {**state, "erro": str(e)}


def no_distribuir(state: EstadoVoos) -> EstadoVoos:
    if state.get("erro"):
        return state
    campanha = state.get("campanha")
    if not campanha:
        return {**state, "erro": "Campanha não disponível para distribuição."}
    try:
        resultado = agente_distribuir_campanha(campanha)
        return {**state, "distribuicao": resultado, "erro": None}
    except Exception as e:
        return {**state, "erro": str(e)}


# ---------- Grafo ----------


def criar_grafo_voos() -> StateGraph:
    grafo = StateGraph(EstadoVoos)
    grafo.add_node("buscar_voos", no_buscar_voos)
    grafo.add_node("analisar_custo_beneficio", no_analisar)
    grafo.add_node("campanha_marketing", no_campanha)
    grafo.add_node("distribuir_emails", no_distribuir)
    grafo.set_entry_point("buscar_voos")
    grafo.add_edge("buscar_voos", "analisar_custo_beneficio")
    grafo.add_edge("analisar_custo_beneficio", "campanha_marketing")
    grafo.add_edge("campanha_marketing", "distribuir_emails")
    grafo.add_edge("distribuir_emails", END)
    return grafo


def executar_pipeline_voos() -> dict:
    """Orquestra os 4 agentes e retorna o estado final."""
    comp = criar_grafo_voos().compile()
    estado_inicial: EstadoVoos = {
        "voos": [],
        "analise": None,
        "campanha": None,
        "distribuicao": None,
        "erro": None,
    }
    return comp.invoke(estado_inicial)


# ---------- CLI ----------


def main():
    print("Orquestrando 4 agentes de voos (LangGraph): Buscar → Analisar → Campanha → Distribuir\n")
    try:
        resultado = executar_pipeline_voos()
    except ValueError as e:
        print(f"Configuração: {e}")
        return
    if resultado.get("erro"):
        print("Erro:", resultado["erro"])
        return
    voos = resultado.get("voos") or []
    analise = resultado.get("analise")
    campanha = resultado.get("campanha")
    dist = resultado.get("distribuicao")
    print("=" * 60)
    print("VOOS SAINDO DE FORTALEZA (Pydantic)")
    print("=" * 60)
    print(json.dumps([v.model_dump() for v in voos], ensure_ascii=False, indent=2))
    if analise:
        print("\n" + "=" * 60)
        print("ANÁLISE CUSTO-BENEFÍCIO POR PERFIL")
        print("=" * 60)
        for a in analise.analises:
            print(f"  [{a.perfil}] {a.voo_recomendado_destino} – nota {a.nota_custo_beneficio}: {a.justificativa[:80]}...")
        print("  Voos mais promissores:", analise.voos_mais_promissores)
    if campanha:
        print("\n" + "=" * 60)
        print("CAMPANHA DE MARKETING")
        print("=" * 60)
        print("Título:", campanha.titulo)
        print("Texto:", campanha.texto_principal[:200] + "...")
        print("Chamada:", campanha.chamada_acao)
    if dist:
        print("\n" + "=" * 60)
        print("DISTRIBUIÇÃO POR E-MAIL")
        print("=" * 60)
        print(f"  Enviados: {dist.total_enviados} | Falhas: {dist.total_falhas}")
        for d in dist.detalhes:
            print(f"  - {d.email}: {d.status}")


if __name__ == "__main__":
    main()
