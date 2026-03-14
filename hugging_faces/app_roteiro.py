"""
Tela Streamlit: roteiros de aula, detecção de objetos e pergunta por voz (Hugging Face API).
Execute: streamlit run app_roteiro.py
"""

import io
from datetime import datetime
from typing import List

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from agente_roteiro_dia import (
    buscar_clima_fortaleza,
    buscar_eventos_fortaleza,
    gerar_roteiro_com_llm,
)
from agente_roteiro_gemini import executar_agente_roteiro_gemini
from agentes_noticias import criar_grafo_noticias, executar_pipeline_noticias
from agentes_voos import criar_grafo_voos, executar_pipeline_voos
from legislacao_rag import garantir_chroma_carregado, perguntar_legislacao, LEGISLACOES
from mcp_cliente_legislacao import (
    consultar_legislacao_mcp,
    executar_skill_mcp,
    listar_skills_mcp,
    registrar_skill_mcp,
)
from asr_api import transcrever_audio_api
from detector_objetos_api import detectar_objetos_api
from roteiro_aula_qwen_api import gerar_roteiro_api, responder_pergunta_api

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

load_dotenv()


def desenhar_caixas(imagem: Image.Image, deteccoes: List[dict]) -> Image.Image:
    """Desenha bounding boxes e labels na imagem."""
    img = imagem.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    cores = ["red", "lime", "blue", "yellow", "magenta", "cyan", "orange"]
    for i, d in enumerate(deteccoes):
        box = d.get("box") or {}
        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)
        label = d.get("label", "?")
        score = d.get("score", 0)
        cor = cores[i % len(cores)]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=cor, width=3)
        texto = f"{label} {score:.0%}"
        draw.text((xmin, max(0, ymin - 18)), texto, fill=cor, font=font)
    return img


st.set_page_config(
    page_title="Hugging Face – Roteiro, Detecção e Voz",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Menu lateral
st.sidebar.title("📋 Menu")
st.sidebar.caption("Escolha a funcionalidade")
PAGINAS = [
    "📋 Roteiro de Aula",
    "🔍 Detecção de Objetos",
    "🎤 Pergunta por voz",
    "🌤️ Roteiro do dia (Fortaleza)",
    "📰 Agentes de Notícias",
    "✈️ Agentes de Voos",
    "📜 Perguntas sobre Legislação",
    "⚖️ Pesquisa Jurídica",
    "🔌 Testar MCP (Cliente e Servidor)",
]
pagina = st.sidebar.radio(
    "Navegação",
    PAGINAS,
    label_visibility="collapsed",
)

# Conteúdo conforme a página selecionada
if pagina == "📋 Roteiro de Aula":
    st.title("📋 Criar Roteiro de Aula")
    st.caption("Geração de roteiros com Qwen via Hugging Face API")

    with st.form("form_roteiro", clear_on_submit=False):
        tema = st.text_input(
            "Tema ou título da aula",
            placeholder="Ex: Fotossíntese, Equações do 2º grau, Revolução Francesa",
            help="Descreva o assunto da aula.",
        )
        col1, col2 = st.columns(2)
        with col1:
            duracao = st.number_input(
                "Duração (minutos)",
                min_value=25,
                max_value=120,
                value=50,
                step=5,
            )
        with col2:
            nivel = st.selectbox(
                "Nível de ensino",
                options=[
                    "fundamental_i",
                    "fundamental_ii",
                    "ensino_medio",
                    "superior",
                ],
                format_func=lambda x: {
                    "fundamental_i": "Fundamental I (anos iniciais)",
                    "fundamental_ii": "Fundamental II (anos finais)",
                    "ensino_medio": "Ensino médio",
                    "superior": "Ensino superior",
                }[x],
            )
        submitted = st.form_submit_button("Gerar roteiro")

    if submitted:
        if not tema or not tema.strip():
            st.error("Informe o tema ou título da aula.")
        else:
            with st.spinner("Gerando roteiro com Qwen…"):
                try:
                    roteiro = gerar_roteiro_api(
                        tema=tema.strip(),
                        duracao_min=int(duracao),
                        nivel=nivel,
                    )
                    st.success("Roteiro gerado.")
                    st.divider()
                    st.markdown(roteiro)
                    st.download_button(
                        label="Baixar roteiro (txt)",
                        data=roteiro,
                        file_name=f"roteiro_{tema.strip()[:30].replace(' ', '_')}.txt",
                        mime="text/plain",
                    )
                except ValueError as e:
                    st.error(str(e))
                    st.info("Crie um arquivo `.env` com `HUGGING_FACE_HUB_TOKEN=seu_token`. Veja `.env.example`.")
                except RuntimeError as e:
                    st.error(str(e))

elif pagina == "🔍 Detecção de Objetos":
    st.title("🔍 Detecção de Objetos")
    st.caption("Envie uma imagem e veja os objetos detectados pela API Hugging Face (DETR)")

    st.subheader("📤 Upload da imagem")
    arquivo = st.file_uploader(
        "Arraste um arquivo ou clique para selecionar (JPEG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="A imagem será enviada para a API de inferência do Hugging Face.",
        label_visibility="collapsed",
    )

    if arquivo is not None:
        st.image(arquivo, caption="Sua imagem", use_column_width=True)

        st.subheader("⚙️ Detecção")
        col_conf, col_btn = st.columns([2, 1])
        with col_conf:
            threshold = st.slider(
                "Confiança mínima (ex.: 0,80 = 80%)",
                min_value=0.1,
                max_value=0.99,
                value=0.8,
                step=0.05,
                format="%.2f",
                help="Detecções abaixo desse valor são ocultadas.",
            )
        with col_btn:
            st.write("")  # alinhar botão
            detectar = st.button("Detectar objetos", type="primary", use_container_width=True)

        if detectar:
            bytes_imagem = arquivo.getvalue()
            with st.spinner("Detectando objetos…"):
                try:
                    deteccoes = detectar_objetos_api(bytes_imagem, threshold=threshold)
                    arquivo.seek(0)
                    img = Image.open(arquivo).convert("RGB")

                    if not deteccoes:
                        st.info("Nenhum objeto detectado acima do limite de confiança.")
                        st.image(img, caption="Sua imagem (sem detecções)", use_column_width=True)
                    else:
                        st.success(f"{len(deteccoes)} objeto(s) detectado(s).")
                        img_com_caixas = desenhar_caixas(img, deteccoes)

                        st.subheader("🖼️ Imagem com objetos detectados")
                        st.image(img_com_caixas, caption="Caixas e rótulos dos objetos", use_column_width=True)

                        with st.expander("Ver lista de objetos"):
                            tabela = [
                                {"Objeto": d.get("label", "?"), "Confiança": f"{d.get('score', 0):.1%}"}
                                for d in deteccoes
                            ]
                            st.dataframe(tabela, use_container_width=True, hide_index=True)
                except ValueError as e:
                    st.error(str(e))
                    st.info("Configure `HUGGING_FACE_HUB_TOKEN` no arquivo `.env`.")
                except RuntimeError as e:
                    st.error(str(e))


elif pagina == "🎤 Pergunta por voz":
    st.title("🎤 Pergunta por voz")
    st.caption("Use o microfone ou envie um áudio; transcrevemos e respondemos com um modelo Hugging Face (Qwen)")

    audio_bytes = None

    st.subheader("🎙️ Usar microfone")
    if mic_recorder is not None:
        audio_dict = mic_recorder(
            start_prompt="🎤 Clique para gravar",
            stop_prompt="⏹️ Clique para parar",
            just_once=False,
            use_container_width=False,
            key="pergunta_voz_mic",
        )
        if audio_dict and audio_dict.get("bytes"):
            audio_bytes = audio_dict["bytes"]
    else:
        if hasattr(st, "audio_input"):
            audio_in = st.audio_input("Gravar com o microfone", type="wav", label_visibility="collapsed")
            if audio_in is not None:
                audio_bytes = audio_in.read()
        if audio_bytes is None:
            st.caption("_Instale `streamlit-mic-recorder` para o botão de microfone: pip install streamlit-mic-recorder_")

    if audio_bytes is None:
        st.subheader("📁 Ou envie um arquivo de áudio")
        arquivo_audio = st.file_uploader(
            "WAV, MP3, FLAC, OGG, WebM",
            type=["wav", "mp3", "flac", "ogg", "webm"],
            label_visibility="collapsed",
        )
        if arquivo_audio is not None:
            audio_bytes = arquivo_audio.read()

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Transcrever e responder", type="primary"):
            with st.spinner("Transcrevendo e gerando resposta…"):
                try:
                    transcricao = transcrever_audio_api(audio_bytes)
                    if not transcricao or not transcricao.strip():
                        st.warning("Não foi possível transcrever o áudio. Tente falar mais claro ou enviar outro áudio.")
                    else:
                        st.subheader("📝 O que você disse")
                        st.write(transcricao)
                        resposta = responder_pergunta_api(transcricao)
                        st.subheader("💬 Resposta")
                        st.markdown(resposta)
                except ValueError as e:
                    st.error(str(e))
                    st.info("Configure `HUGGING_FACE_HUB_TOKEN` no arquivo `.env`.")
                except RuntimeError as e:
                    st.error(str(e))


elif pagina == "🌤️ Roteiro do dia (Fortaleza)":
    st.title("🌤️ Roteiro do dia – Fortaleza")
    st.caption("O agente consulta o clima e eventos em Fortaleza e monta um roteiro sugerido para o dia.")

    modelo_roteiro = st.radio(
        "Modelo para gerar o roteiro",
        ["Hugging Face (Qwen)", "Gemini (Google)"],
        horizontal=True,
        help="Hugging Face usa Qwen via API; Gemini usa LangChain + Google (requer GEMINI_API_KEY no .env).",
    )

    if st.button("Gerar roteiro do dia", type="primary"):
        with st.spinner("Consultando clima e eventos…"):
            try:
                clima = buscar_clima_fortaleza()
                eventos = buscar_eventos_fortaleza()
            except Exception as e:
                st.error(f"Erro ao buscar dados: {e}")
                clima = ""
                eventos = ""

        if clima or eventos:
            st.subheader("🌡️ Dados climáticos – Fortaleza")
            if clima:
                st.info(clima)
            else:
                st.caption("_Clima não disponível._")

            with st.expander("📅 Eventos / agenda (resumo)"):
                st.text(eventos if eventos else "Nenhum dado de eventos.")

        with st.spinner(f"Gerando roteiro com {modelo_roteiro}…"):
            try:
                if modelo_roteiro == "Hugging Face (Qwen)":
                    roteiro = gerar_roteiro_com_llm(
                        clima=clima or "Clima não obtido.",
                        eventos=eventos or "Eventos não obtidos.",
                    )
                else:
                    roteiro = executar_agente_roteiro_gemini()
                st.success("Roteiro gerado.")
                st.divider()
                st.subheader("📋 Roteiro do dia")
                st.markdown(roteiro)
                st.download_button(
                    label="Baixar roteiro (txt)",
                    data=roteiro,
                    file_name=f"roteiro_fortaleza_{datetime.now().strftime('%Y-%m-%d')}.txt",
                    mime="text/plain",
                )
            except ValueError as e:
                st.error(str(e))
                if modelo_roteiro == "Hugging Face (Qwen)":
                    st.info("Configure `HUGGING_FACE_HUB_TOKEN` no arquivo `.env`.")
                else:
                    st.info("Configure `GEMINI_API_KEY` ou `GOOGLE_API_KEY` no arquivo `.env`.")
            except RuntimeError as e:
                st.error(str(e))


elif pagina == "📰 Agentes de Notícias":
    st.title("📰 Agentes de Notícias (UOL)")
    st.caption("Pipeline com 3 agentes orquestrados por LangGraph: Buscar → Classificar → Resumir.")

    # Grafo: sempre exibir (não depende da execução)
    st.subheader("🔄 Grafo do pipeline (LangGraph)")
    lg = None
    try:
        comp = criar_grafo_noticias().compile()
        lg = comp.get_graph()
        png_bytes = lg.draw_mermaid_png()
        st.image(Image.open(io.BytesIO(png_bytes)), use_container_width=True)
    except Exception as e:
        st.warning(f"Visualização PNG do grafo indisponível ({e}). Exibindo diagrama em texto.")
        try:
            if lg is not None:
                mermaid = lg.draw_mermaid()
                st.code(mermaid, language="text")
            else:
                comp = criar_grafo_noticias().compile()
                mermaid = comp.get_graph().draw_mermaid()
                st.code(mermaid, language="text")
        except Exception:
            st.code("buscar → classificar → resumir → END", language="text")

    st.divider()
    st.subheader("▶️ Executar pipeline")

    if st.button("Executar 3 agentes (Buscar → Classificar → Resumir)", type="primary", key="run_noticias"):
        progress = st.progress(0, text="Iniciando…")
        try:
            progress.progress(10, text="Buscando notícias no UOL (agente 1)…")
            resultado = executar_pipeline_noticias()
            progress.progress(100, text="Concluído.")
        except ValueError as e:
            st.error(str(e))
            st.info("Configure `GEMINI_API_KEY` ou `GOOGLE_API_KEY` no arquivo `.env`.")
            resultado = None
        except Exception as e:
            st.error(str(e))
            resultado = None

        if resultado is not None:
            if resultado.get("erro"):
                st.error("Erro no pipeline: " + resultado["erro"])
            else:
                noticias = resultado.get("noticias") or []
                classificadas = resultado.get("noticias_classificadas") or []
                resumo = resultado.get("resumo_do_dia")

                st.success("Pipeline executado com sucesso.")

                with st.expander("📄 Notícias buscadas (JSON estruturado)", expanded=True):
                    if noticias:
                        st.json([n.model_dump() if hasattr(n, "model_dump") else n for n in noticias])
                    else:
                        st.caption("Nenhuma notícia retornada.")

                with st.expander("🏷️ Notícias classificadas por perfil", expanded=True):
                    if classificadas:
                        for n in classificadas:
                            st.markdown(f"- **[{n.perfil}]** {n.titulo}")
                            if n.resumo:
                                st.caption(n.resumo[:200] + ("…" if len(n.resumo) > 200 else ""))
                    else:
                        st.caption("Nenhuma classificação.")

                st.subheader("📋 Resumo do dia")
                if resumo:
                    st.markdown(resumo.resumo_geral)
                    st.markdown("**Destaques:**")
                    for d in resumo.destaques:
                        st.markdown(f"- {d}")
                    if resumo.perfis_abordados:
                        st.caption("Perfis: " + ", ".join(resumo.perfis_abordados))
                else:
                    st.caption("Nenhum resumo gerado.")

elif pagina == "✈️ Agentes de Voos":
    NODOS_VOOS = [
        ("buscar_voos", "Buscar voos (Fortaleza)", "🛫"),
        ("analisar_custo_beneficio", "Analisar custo-benefício por perfil", "📊"),
        ("campanha_marketing", "Campanha de marketing", "📣"),
        ("distribuir_emails", "Distribuir por e-mail", "📧"),
    ]
    st.title("✈️ Agentes de Voos")
    st.caption("Pipeline: Buscar voos → Analisar custo-benefício → Campanha → Distribuir e-mails.")

    # Grafo de execução
    st.subheader("🔄 Grafo de execução (LangGraph)")
    lg_voos = None
    try:
        comp_voos = criar_grafo_voos().compile()
        lg_voos = comp_voos.get_graph()
        png_voos = lg_voos.draw_mermaid_png()
        st.image(Image.open(io.BytesIO(png_voos)), use_container_width=True)
    except Exception as e:
        st.warning(f"Visualização do grafo indisponível ({e}).")
        try:
            comp_voos = criar_grafo_voos().compile()
            mermaid_voos = comp_voos.get_graph().draw_mermaid()
            st.code(mermaid_voos, language="text")
        except Exception:
            st.code("buscar_voos → analisar_custo_beneficio → campanha_marketing → distribuir_emails → END", language="text")

    st.divider()
    st.subheader("▶️ Executar pipeline")

    if st.button("Executar 4 agentes de voos", type="primary", key="run_voos"):
        estado_inicial = {
            "voos": [],
            "analise": None,
            "campanha": None,
            "distribuicao": None,
            "erro": None,
        }
        status_containers = [st.empty() for _ in NODOS_VOOS]
        try:
            comp_voos = criar_grafo_voos().compile()
            resultado_final = None
            # Mostrar "Executando…" em todos; ao receber cada chunk, destacar o nó concluído
            for i, (_, label, emoji) in enumerate(NODOS_VOOS):
                status_containers[i].info(f"**{emoji} {label}** — Executando…")
            # stream_mode="values" → cada chunk é estado completo; último = resultado final
            concluidos = set()
            for state_chunk in comp_voos.stream(estado_inicial, stream_mode="values"):
                resultado_final = state_chunk
                # Destacar nó pelo conteúdo do estado (evita depender da ordem dos chunks)
                if state_chunk.get("voos") and "buscar_voos" not in concluidos:
                    concluidos.add("buscar_voos")
                    i, label, emoji = 0, NODOS_VOOS[0][1], NODOS_VOOS[0][2]
                    status_containers[i].success(f"**{emoji} {label}** ✓ → {len(state_chunk['voos'])} voos")
                if state_chunk.get("analise") and "analisar_custo_beneficio" not in concluidos:
                    concluidos.add("analisar_custo_beneficio")
                    i, label, emoji = 1, NODOS_VOOS[1][1], NODOS_VOOS[1][2]
                    n = len(getattr(state_chunk["analise"], "analises", []))
                    status_containers[i].success(f"**{emoji} {label}** ✓ → {n} perfis")
                if state_chunk.get("campanha") and "campanha_marketing" not in concluidos:
                    concluidos.add("campanha_marketing")
                    i, label, emoji = 2, NODOS_VOOS[2][1], NODOS_VOOS[2][2]
                    tit = (getattr(state_chunk["campanha"], "titulo", "") or "")[:45]
                    if len(getattr(state_chunk["campanha"], "titulo", "") or "") > 45:
                        tit += "…"
                    status_containers[i].success(f"**{emoji} {label}** ✓ → {tit}")
                if state_chunk.get("distribuicao") and "distribuir_emails" not in concluidos:
                    concluidos.add("distribuir_emails")
                    i, label, emoji = 3, NODOS_VOOS[3][1], NODOS_VOOS[3][2]
                    n = getattr(state_chunk["distribuicao"], "total_enviados", 0)
                    status_containers[i].success(f"**{emoji} {label}** ✓ → {n} enviados")
        except ValueError as e:
            st.error(str(e))
            st.info("Configure `GEMINI_API_KEY` ou `GOOGLE_API_KEY` no `.env`.")
            resultado_final = None
        except Exception as e:
            st.error(str(e))
            resultado_final = None
        for i in range(len(NODOS_VOOS)):
            status_containers[i].empty()

        if resultado_final is not None:
            if resultado_final.get("erro"):
                st.error("Erro no pipeline: " + resultado_final["erro"])
            else:
                st.success("Pipeline concluído.")
                voos = resultado_final.get("voos") or []
                analise = resultado_final.get("analise")
                campanha = resultado_final.get("campanha")
                dist = resultado_final.get("distribuicao")

                # Saída de cada agente
                st.subheader("📋 Saída de cada agente")

                with st.expander("🛫 Agente 1 — Buscar voos (Fortaleza): saída", expanded=True):
                    if voos:
                        for v in voos:
                            preco = getattr(v, "preco_reais", 0)
                            dest = getattr(v, "destino_nome", None) or getattr(v, "destino", "")
                            comp = getattr(v, "companhia", "")
                            st.markdown(f"- **{dest}** — R$ {preco:.2f} ({comp})")
                    else:
                        st.caption("Nenhum voo.")

                with st.expander("📊 Agente 2 — Análise custo-benefício por perfil: saída", expanded=True):
                    if analise:
                        analises_list = getattr(analise, "analises", [])
                        for a in analises_list:
                            perfil = getattr(a, "perfil", "")
                            dest = getattr(a, "voo_recomendado_destino", "")
                            nota = getattr(a, "nota_custo_beneficio", 0)
                            just = (getattr(a, "justificativa", "") or "")[:120]
                            st.markdown(f"- **[{perfil}]** {dest} (nota {nota}): {just}…")
                        prom = getattr(analise, "voos_mais_promissores", [])
                        st.caption("Mais promissores: " + ", ".join(prom))
                    else:
                        st.caption("Nenhuma análise.")

                with st.expander("📣 Agente 3 — Campanha de marketing: saída", expanded=True):
                    if campanha:
                        titulo_c = getattr(campanha, "titulo", "") or ""
                        texto_c = getattr(campanha, "texto_principal", "") or ""
                        cta_c = getattr(campanha, "chamada_acao", "") or ""
                        st.markdown(f"**{titulo_c}**")
                        st.markdown(texto_c)
                        st.markdown(f"*{cta_c}*")
                    else:
                        st.caption("Nenhuma campanha.")

                with st.expander("📧 Agente 4 — Distribuição por e-mail: saída", expanded=True):
                    if dist:
                        st.metric("Enviados", getattr(dist, "total_enviados", 0))
                        st.metric("Falhas", getattr(dist, "total_falhas", 0))
                        detalhes_list = getattr(dist, "detalhes", [])
                        for d in detalhes_list:
                            email = getattr(d, "email", "")
                            status = getattr(d, "status", "")
                            st.markdown(f"- {email}: **{status}**")
                    else:
                        st.caption("Nenhuma distribuição.")

elif pagina == "📜 Perguntas sobre Legislação":
    st.title("📜 RAG sobre Legislação")
    st.caption("5 leis simuladas no ChromaDB. Consulte direto (RAG) ou via MCP, com skills extensíveis.")
    with st.expander("📄 Legislações disponíveis (simuladas)", expanded=False):
        for i, doc in enumerate(LEGISLACOES, 1):
            meta = getattr(doc, "metadata", {}) or {}
            titulo = meta.get("titulo", meta.get("lei", f"Lei {i}"))
            st.markdown(f"**{i}. {titulo}**")
            st.caption(doc.page_content[:200] + "…" if len(doc.page_content) > 200 else doc.page_content)

    modo_leg = st.radio(
        "Modo de consulta",
        ["RAG direto (LangChain + LangGraph)", "Via MCP (servidor + skills)"],
        horizontal=True,
        key="modo_legislacao",
    )

    pergunta_leg = st.text_input(
        "Pergunte sobre a legislação",
        placeholder="Ex: Qual o prazo para férias? Quais direitos do teletrabalhador?",
        key="pergunta_legislacao",
    )

    if modo_leg == "RAG direto (LangChain + LangGraph)":
        if st.button("Consultar (RAG)", type="primary", key="btn_rag_leg"):
            if not pergunta_leg or not pergunta_leg.strip():
                st.warning("Digite uma pergunta.")
            else:
                try:
                    garantir_chroma_carregado()
                    with st.spinner("Buscando nos trechos e gerando resposta…"):
                        resultado = perguntar_legislacao(pergunta_leg.strip())
                    with st.expander("📎 Trechos utilizados (contexto)", expanded=False):
                        st.text(resultado.get("contexto", ""))
                    st.subheader("Resposta")
                    st.markdown(resultado.get("resposta", ""))
                except ValueError as e:
                    st.error(str(e))
                    st.info("Configure GEMINI_API_KEY ou GOOGLE_API_KEY no .env")
                except Exception as e:
                    st.error(str(e))

    else:
        st.caption("O servidor MCP é iniciado em subprocess ao clicar. Skills permitem adicionar novas habilidades ao agente.")
        if st.button("Consultar via MCP", type="primary", key="btn_mcp_leg"):
            if not pergunta_leg or not pergunta_leg.strip():
                st.warning("Digite uma pergunta.")
            else:
                try:
                    with st.spinner("Chamando servidor MCP…"):
                        resposta_mcp = consultar_legislacao_mcp(pergunta_leg.strip())
                    st.subheader("Resposta (via MCP)")
                    st.markdown(resposta_mcp)
                except Exception as e:
                    st.error(str(e))

        st.subheader("🛠️ Skills (MCP)")
        st.caption("Registre novas skills para o agente ou execute as existentes.")
        with st.expander("Listar skills registradas", expanded=True):
            if st.button("Atualizar lista", key="btn_list_skills"):
                try:
                    with st.spinner("Chamando MCP…"):
                        st.session_state["leg_lista_skills"] = listar_skills_mcp()
                except Exception as e:
                    st.session_state["leg_lista_skills"] = f"Erro: {e}"
            if "leg_lista_skills" in st.session_state:
                st.markdown(st.session_state["leg_lista_skills"])
            else:
                st.caption("Clique em **Atualizar lista** para ver as skills registradas.")

        with st.expander("Registrar nova skill", expanded=False):
            nome_skill = st.text_input("Nome da skill (ex: resumir_artigo)", key="nome_skill")
            desc_skill = st.text_input("Descrição (o que a skill faz)", key="desc_skill")
            inst_skill = st.text_area("Instruções (como executar)", placeholder="Ex: Resuma o texto em até 3 tópicos.", key="inst_skill")
            if st.button("Registrar skill", key="btn_reg_skill"):
                if nome_skill and desc_skill and inst_skill:
                    try:
                        msg = registrar_skill_mcp(nome_skill, desc_skill, inst_skill)
                        st.success(msg)
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Preencha nome, descrição e instruções.")

        with st.expander("Executar uma skill", expanded=False):
            nome_exec = st.text_input("Nome da skill", key="nome_exec_skill")
            entrada_exec = st.text_area("Entrada (texto para a skill processar)", key="entrada_exec_skill")
            if st.button("Executar", key="btn_exec_skill"):
                if nome_exec:
                    try:
                        with st.spinner("Executando…"):
                            out = executar_skill_mcp(nome_exec, entrada_exec or "")
                        st.markdown(out)
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Informe o nome da skill.")

elif pagina == "⚖️ Pesquisa Jurídica":
    st.title("⚖️ Pesquisa Jurídica")
    st.caption("Consulte a base de legislação simulada (5 leis). Digite sua pergunta e veja a resposta com base nos trechos encontrados.")
    pergunta_jur = st.text_area(
        "Sua pergunta",
        placeholder="Ex: Qual o prazo para férias? O empregador deve pagar internet no teletrabalho? Quantos dias de licença-maternidade?",
        height=100,
        key="pergunta_pesquisa_juridica",
    )
    if st.button("Pesquisar", type="primary", key="btn_pesquisa_juridica"):
        if not pergunta_jur or not pergunta_jur.strip():
            st.warning("Digite uma pergunta.")
        else:
            try:
                garantir_chroma_carregado()
                with st.spinner("Consultando a base e gerando resposta…"):
                    resultado = perguntar_legislacao(pergunta_jur.strip())
                st.subheader("Resposta")
                st.markdown(resultado.get("resposta", ""))
                with st.expander("📎 Trechos da legislação utilizados", expanded=False):
                    st.text(resultado.get("contexto", ""))
            except ValueError as e:
                st.error(str(e))
                st.info("Configure GEMINI_API_KEY ou GOOGLE_API_KEY no .env")
            except Exception as e:
                st.error(str(e))

elif pagina == "🔌 Testar MCP (Cliente e Servidor)":
    st.title("🔌 Testar MCP — Cliente e Servidor")
    st.caption("O cliente inicia o servidor em subprocess (stdio) ao chamar cada tool. Use os blocos abaixo para testar cada ferramenta.")

    st.subheader("📡 Servidor")
    with st.expander("Como rodar o servidor standalone", expanded=False):
        st.code("python mcp_server_legislacao.py          # stdio (padrão)\npython mcp_server_legislacao.py --http 8000  # HTTP em http://127.0.0.1:8000/mcp", language="text")
        st.caption("Nesta tela o cliente dispara o servidor automaticamente via stdio ao clicar nos botões.")

    st.divider()
    st.subheader("🧪 Testar tools do cliente")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1. consultar_legislacao**")
        pergunta_mcp = st.text_input("Pergunta", placeholder="Ex: Quantos dias de férias?", key="mcp_test_pergunta")
        if st.button("Chamar consultar_legislacao", key="mcp_btn_consultar"):
            if pergunta_mcp and pergunta_mcp.strip():
                try:
                    with st.spinner("Servidor em subprocess…"):
                        out = consultar_legislacao_mcp(pergunta_mcp.strip())
                    st.session_state["mcp_result_consultar"] = out
                except Exception as e:
                    st.session_state["mcp_result_consultar"] = f"Erro: {e}"
            else:
                st.warning("Digite uma pergunta.")
        if "mcp_result_consultar" in st.session_state:
            st.text_area("Resposta", value=st.session_state["mcp_result_consultar"], height=180, key="mcp_out_consultar", disabled=True)

        st.markdown("**2. listar_skills**")
        if st.button("Chamar listar_skills", key="mcp_btn_listar"):
            try:
                with st.spinner("Servidor em subprocess…"):
                    out = listar_skills_mcp()
                st.session_state["mcp_result_listar"] = out
            except Exception as e:
                st.session_state["mcp_result_listar"] = f"Erro: {e}"
        if "mcp_result_listar" in st.session_state:
            st.text_area("Resposta", value=st.session_state["mcp_result_listar"], height=120, key="mcp_out_listar", disabled=True)

    with col2:
        st.markdown("**3. registrar_skill**")
        r_nome = st.text_input("Nome", placeholder="ex: resumir_artigo", key="mcp_reg_nome")
        r_desc = st.text_input("Descrição", placeholder="Resumir texto em tópicos", key="mcp_reg_desc")
        r_inst = st.text_area("Instruções", placeholder="Resuma em até 3 tópicos.", height=60, key="mcp_reg_inst")
        if st.button("Chamar registrar_skill", key="mcp_btn_reg"):
            if r_nome and r_desc and r_inst:
                try:
                    with st.spinner("Servidor em subprocess…"):
                        out = registrar_skill_mcp(r_nome, r_desc, r_inst)
                    st.session_state["mcp_result_reg"] = out
                except Exception as e:
                    st.session_state["mcp_result_reg"] = f"Erro: {e}"
            else:
                st.warning("Preencha nome, descrição e instruções.")
        if "mcp_result_reg" in st.session_state:
            st.success(st.session_state["mcp_result_reg"])

        st.markdown("**4. executar_skill**")
        e_nome = st.text_input("Nome da skill", key="mcp_exec_nome")
        e_entrada = st.text_area("Entrada", placeholder="Texto para a skill processar", height=80, key="mcp_exec_entrada")
        if st.button("Chamar executar_skill", key="mcp_btn_exec"):
            if e_nome:
                try:
                    with st.spinner("Servidor em subprocess…"):
                        out = executar_skill_mcp(e_nome, e_entrada or "")
                    st.session_state["mcp_result_exec"] = out
                except Exception as e:
                    st.session_state["mcp_result_exec"] = f"Erro: {e}"
            else:
                st.warning("Informe o nome da skill.")
        if "mcp_result_exec" in st.session_state:
            st.text_area("Resposta", value=st.session_state["mcp_result_exec"], height=120, key="mcp_out_exec", disabled=True)
