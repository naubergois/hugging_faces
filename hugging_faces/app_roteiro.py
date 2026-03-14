"""
Tela Streamlit para criação de roteiros de aula com Qwen (API Hugging Face).
Execute: streamlit run app_roteiro.py
"""

import streamlit as st
from dotenv import load_dotenv

from roteiro_aula_qwen_api import gerar_roteiro_api

load_dotenv()

st.set_page_config(
    page_title="Roteiro de Aula",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

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
