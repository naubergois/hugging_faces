#!/usr/bin/env python3
"""
Gera roteiro de aula usando o modelo Qwen via Hugging Face.
Uso: python roteiro_aula_qwen.py "Tema da aula" [--duracao 50] [--nivel ensino_medio]
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Modelo Qwen recomendado (pode trocar por Qwen2.5-7B-Instruct ou outro)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # leve para CPU; use 7B/72B para melhor qualidade


def carregar_modelo(model_id: str = MODEL_ID, device: str = None):
    """Carrega tokenizer e modelo Qwen."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Carregando {model_id} em {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    return tokenizer, model, device


def gerar_roteiro(
    tema: str,
    tokenizer,
    model,
    device: str,
    duracao_min: int = 50,
    nivel: str = "ensino médio",
    max_new_tokens: int = 1500,
) -> str:
    """Gera um roteiro de aula em texto a partir do tema."""
    nivel_desc = {
        "fundamental_i": "anos iniciais do ensino fundamental",
        "fundamental_ii": "anos finais do ensino fundamental",
        "ensino_medio": "ensino médio",
        "superior": "ensino superior",
    }.get(nivel.lower().replace(" ", "_"), nivel)

    prompt = f"""Você é um professor experiente. Crie um roteiro de aula completo e objetivo.

Tema da aula: {tema}
Público-alvo: {nivel_desc}
Duração sugerida: {duracao_min} minutos

O roteiro deve conter, de forma clara e numerada:
1. Objetivos de aprendizagem (3 a 5 itens)
2. Materiais necessários
3. Desenvolvimento da aula (etapas com tempo aproximado)
4. Atividades ou dinâmicas
5. Avaliação ou verificação de aprendizagem
6. Sugestões de aprofundamento ou tarefa de casa

Roteiro da aula:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove o prompt da resposta, deixando só o roteiro
    if "Roteiro da aula:" in resposta:
        resposta = resposta.split("Roteiro da aula:")[-1].strip()
    return resposta


def main():
    parser = argparse.ArgumentParser(description="Gera roteiro de aula com Qwen (Hugging Face)")
    parser.add_argument("tema", type=str, help="Tema ou título da aula")
    parser.add_argument("--duracao", type=int, default=50, help="Duração em minutos (padrão: 50)")
    parser.add_argument(
        "--nivel",
        type=str,
        default="ensino médio",
        choices=["fundamental_i", "fundamental_ii", "ensino_medio", "superior"],
        help="Nível de ensino",
    )
    parser.add_argument("--modelo", type=str, default=MODEL_ID, help="ID do modelo no Hugging Face")
    parser.add_argument("--max-tokens", type=int, default=1500, help="Máximo de tokens na resposta")
    args = parser.parse_args()

    tokenizer, model, device = carregar_modelo(args.modelo)
    roteiro = gerar_roteiro(
        args.tema,
        tokenizer,
        model,
        device,
        duracao_min=args.duracao,
        nivel=args.nivel,
        max_new_tokens=args.max_tokens,
    )
    print("\n" + "=" * 60 + "\n")
    print(roteiro)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
