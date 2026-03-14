# Roteiro de Aula com Qwen (Hugging Face)

Geração de roteiros de aula usando o modelo **Qwen** no Hugging Face.

## Opção 1: Modelo local (transformers)

Requer instalação do modelo no seu ambiente (funciona offline após o download).

### Instalação

```bash
pip install -r requirements.txt
```

### Uso

```bash
# Tema simples
python roteiro_aula_qwen.py "Fotossíntese"

# Com duração e nível
python roteiro_aula_qwen.py "Equações do 2º grau" --duracao 50 --nivel ensino_medio

# Usar modelo maior (melhor qualidade, mais RAM/VRAM)
python roteiro_aula_qwen.py "Mitose e meiose" --modelo Qwen/Qwen2.5-7B-Instruct
```

**Níveis:** `fundamental_i`, `fundamental_ii`, `ensino_medio`, `superior`

O script usa por padrão `Qwen2.5-0.5B-Instruct` (leve). Para melhor texto, use `Qwen/Qwen2.5-7B-Instruct` ou `Qwen/Qwen2.5-14B-Instruct` se tiver GPU com bastante memória.

---

## Opção 2: API do Hugging Face

Não baixa o modelo; usa a API (requer conta e token).

### Token

1. Crie uma conta em [huggingface.co](https://huggingface.co).
2. Gere um token em [Settings → Access Tokens](https://huggingface.co/settings/tokens).
3. Coloque o token no arquivo `.env` (copie de `.env.example` e preencha):

```bash
HUGGING_FACE_HUB_TOKEN=seu_token_aqui
```

### Uso (linha de comando)

```bash
pip install -r requirements.txt
python roteiro_aula_qwen_api.py "Fotossíntese" --duracao 50 --nivel ensino_medio
```

### Uso (interface Streamlit)

```bash
pip install -r requirements.txt
streamlit run app_roteiro.py
```

Abra o navegador na URL indicada (geralmente http://localhost:8501). Preencha tema, duração e nível e clique em **Gerar roteiro**. O token é lido do `.env`.

---

## Estrutura do roteiro gerado

O prompt pede ao Qwen um roteiro com:

1. Objetivos de aprendizagem  
2. Materiais necessários  
3. Desenvolvimento (etapas com tempo)  
4. Atividades ou dinâmicas  
5. Avaliação / verificação  
6. Sugestões de aprofundamento ou tarefa de casa  

Ajuste o texto do prompt em `roteiro_aula_qwen.py` ou `roteiro_aula_qwen_api.py` se quiser outro formato.
