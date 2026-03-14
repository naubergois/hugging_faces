#!/usr/bin/env python3
"""
RAG sobre legislação simulada: 5 leis, ChromaDB e agente LangChain + LangGraph.
Perguntas são respondidas com base nos trechos recuperados do banco vetorial.

Requer: pip install chromadb langchain-chroma langchain-google-genai langgraph
Chave no .env: GEMINI_API_KEY ou GOOGLE_API_KEY

Uso: python legislacao_rag.py "Qual o prazo para férias?"
"""

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_legislacao"
COLLECTION_NAME = "legislacao"


# ---------- 5 Legislações simuladas ----------


LEGISLACOES = [
    Document(
        page_content="""Lei Simulada nº 1/2024 - Dispõe sobre o Teletrabalho.
Art. 1º O teletrabalho é a prestação de serviços preponderantemente fora das dependências do empregador, com a utilização de tecnologias de informação e comunicação.
Art. 2º A jornada de trabalho do teletrabalhador deve ser pactuada por acordo individual ou coletivo, com registro em folha de ponto eletrônico ou equivalente.
Art. 3º O empregador deve arcar com as despesas necessárias à realização do teletrabalho, inclusive equipamentos e conexão à internet, salvo acordo em contrário.
Art. 4º O teletrabalhador tem os mesmos direitos dos demais trabalhadores, incluindo férias, 13º salário e FGTS.""",
        metadata={"lei": "Lei 1/2024", "titulo": "Teletrabalho"},
    ),
    Document(
        page_content="""Lei Simulada nº 2/2024 - Férias anuais remuneradas.
Art. 1º Todo empregado tem direito a férias anuais de 30 dias corridos, após cada período de 12 meses de vigência do contrato.
Art. 2º As férias podem ser divididas em até três períodos, sendo um deles de no mínimo 14 dias corridos.
Art. 3º O empregador deve notificar o empregado sobre a época das férias com antecedência mínima de 30 dias.
Art. 4º É vedado o início das férias em período que implique perda de dias de descanso em dias de repouso semanal. O abono de férias corresponde a um terço do salário.""",
        metadata={"lei": "Lei 2/2024", "titulo": "Férias"},
    ),
    Document(
        page_content="""Lei Simulada nº 3/2024 - Proteção de dados pessoais.
Art. 1º O tratamento de dados pessoais deve observar a boa-fé, finalidade, necessidade e transparência.
Art. 2º O consentimento do titular é uma das bases legais para o tratamento. Dados sensíveis exigem consentimento específico e destacado.
Art. 3º O controlador deve manter registro das operações de tratamento. A Autoridade Nacional pode solicitar tais registros.
Art. 4º O titular tem direito a confirmação da existência de tratamento, acesso aos dados, correção, anonimização, portabilidade e eliminação. O direito à eliminação não se aplica quando a conservação for necessária ao cumprimento de obrigação legal.""",
        metadata={"lei": "Lei 3/2024", "titulo": "Proteção de Dados"},
    ),
    Document(
        page_content="""Lei Simulada nº 4/2024 - Licenças e afastamentos.
Art. 1º A licença-maternidade é de 120 dias, sem prejuízo do emprego e do salário. A licença-paternidade é de 20 dias corridos.
Art. 2º Em caso de doença devidamente atestada, o empregado pode se afastar mantendo estabilidade conforme previsto em lei. O auxílio-doença é pago pela Previdência após o 16º dia de afastamento.
Art. 3º A licença para tratamento de saúde de familiar pode ser concedida por até 60 dias, consecutivos ou não, por ano, mediante atestado médico.
Art. 4º O empregado que doar sangue tem direito a um dia de folga no dia da doação.""",
        metadata={"lei": "Lei 4/2024", "titulo": "Licenças"},
    ),
    Document(
        page_content="""Lei Simulada nº 5/2024 - Horas extras e banco de horas.
Art. 1º A jornada de trabalho não pode exceder 8 horas diárias e 44 horas semanais. O trabalho em horário superior deve ser remunerado com acréscimo de no mínimo 50% sobre a hora normal.
Art. 2º O banco de horas pode ser instituído por acordo individual escrito ou por convenção/acordo coletivo, para compensação em até um ano.
Art. 3º Em regime de compensação, a jornada máxima em um dia pode ser de até 10 horas, desde que não ultrapasse 44 horas na semana.
Art. 4º O empregado em regime de tempo parcial não pode fazer horas extras, salvo acordo de prorrogação com pagamento da hora extra.""",
        metadata={"lei": "Lei 5/2024", "titulo": "Horas extras"},
    ),
]


# ---------- ChromaDB + embeddings ----------


def _get_embeddings():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no .env")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=key,
    )


def _get_llm():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no .env")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=key,
        temperature=0.2,
    )


def criar_vectorstore(force_recreate: bool = False) -> Chroma:
    """Cria ou carrega o ChromaDB com as 5 legislações (embeddings via Gemini)."""
    embeddings = _get_embeddings()
    persist_dir = str(CHROMA_DIR)
    if force_recreate and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    if not CHROMA_DIR.exists() or force_recreate:
        # Quebrar textos em chunks menores para melhor retrieval
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", "Art.", " ", ""],
        )
        chunks = splitter.split_documents(LEGISLACOES)
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
        )
        return vectorstore
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


# ---------- RAG: retriever + prompt ----------


def get_retriever(top_k: int = 4):
    """Retriever do Chroma (top_k trechos mais relevantes)."""
    vs = criar_vectorstore()
    return vs.as_retriever(search_kwargs={"k": top_k})


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ---------- Estado e nós LangGraph ----------


class EstadoRAG(TypedDict):
    pergunta: str
    contexto: str
    resposta: str


def node_retrieve(state: EstadoRAG) -> EstadoRAG:
    """Nó 1: recupera trechos relevantes no ChromaDB."""
    retriever = get_retriever()
    docs = retriever.invoke(state["pergunta"])
    contexto = format_docs(docs)
    return {**state, "contexto": contexto}


def node_responder(state: EstadoRAG) -> EstadoRAG:
    """Nó 2: gera resposta com base no contexto (RAG)."""
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um assistente jurídico que responde com base APENAS nas leis fornecidas no contexto. "
            "Se a resposta não estiver no contexto, diga que não encontrou na legislação disponível. "
            "Responda em português, de forma clara e objetiva. Cite o número da lei ou do artigo quando possível.",
        ),
        ("human", "Contexto (trechos das leis):\n\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:"),
    ])
    chain = prompt | llm
    resposta = chain.invoke({"contexto": state["contexto"], "pergunta": state["pergunta"]})
    texto = resposta.content if hasattr(resposta, "content") else str(resposta)
    return {**state, "resposta": texto}


# ---------- Grafo LangGraph ----------


def criar_grafo_rag() -> StateGraph:
    grafo = StateGraph(EstadoRAG)
    grafo.add_node("retrieve", node_retrieve)
    grafo.add_node("responder", node_responder)
    grafo.set_entry_point("retrieve")
    grafo.add_edge("retrieve", "responder")
    grafo.add_edge("responder", END)
    return grafo


def perguntar_legislacao(pergunta: str, top_k: int = 4) -> dict:
    """
    Executa o agente RAG: recupera trechos no ChromaDB e gera resposta com o LLM.
    Retorna dict com pergunta, contexto (trechos) e resposta.
    """
    grafo = criar_grafo_rag().compile()
    estado_inicial: EstadoRAG = {
        "pergunta": pergunta,
        "contexto": "",
        "resposta": "",
    }
    final = grafo.invoke(estado_inicial)
    return final


# ---------- Inicialização do Chroma (chamada na primeira pergunta) ----------


def garantir_chroma_carregado():
    """Garante que o ChromaDB existe e está populado (chame ao iniciar a app)."""
    criar_vectorstore()


# ---------- CLI ----------


def main():
    import sys
    garantir_chroma_carregado()
    if len(sys.argv) < 2:
        print("Uso: python legislacao_rag.py \"Sua pergunta sobre a legislação\"")
        return
    pergunta = " ".join(sys.argv[1:])
    print("Pergunta:", pergunta)
    print()
    resultado = perguntar_legislacao(pergunta)
    print("Contexto (trechos utilizados):")
    print("-" * 60)
    print(resultado["contexto"][:1500] + "..." if len(resultado["contexto"]) > 1500 else resultado["contexto"])
    print("-" * 60)
    print("Resposta:")
    print(resultado["resposta"])


if __name__ == "__main__":
    main()
