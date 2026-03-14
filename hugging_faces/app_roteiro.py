"""
Tela Streamlit: roteiros de aula, detecção de objetos e pergunta por voz (Hugging Face API).
Execute: streamlit run app_roteiro.py
"""

import io
from typing import List

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from asr_api import transcrever_audio_api
from detector_objetos_api import detectar_objetos_api
from roteiro_aula_qwen_api import gerar_roteiro_api, responder_pergunta_api

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

load_dotenv()

st.set_page_config(
    page_title="Hugging Face – Roteiro, Detecção e Voz",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

tab_roteiro, tab_deteccao, tab_voz = st.tabs([
    "📋 Roteiro de Aula",
    "🔍 Detecção de Objetos",
    "🎤 Pergunta por voz",
])

with tab_roteiro:
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


with tab_deteccao:
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


with tab_voz:
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
