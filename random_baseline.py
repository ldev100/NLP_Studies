import json
import random
import re
from pathlib import Path

INPUT_FILE = "clean_synthetic_anamneses_gold_standart.json"
OUTPUT_DIR = Path("datasets_experimento_ruido")

TAXAS = {
    "rate10": 0.10,
    "rate25": 0.25,
    "rate50": 0.50,
}

MIN_TOKEN_LEN = 3
SEED = 42

def perturbar_caractere(palavra):
    """
    Aplica UMA perturbação aleatória a nível de caractere.
    Operações: substituir, deletar, inserir ou trocar letras adjacentes.
    """
    if len(palavra) <= 2:
        return palavra

    chars = list(palavra)
    operacao = random.choice(["substituir", "deletar", "inserir", "trocar"])

    if operacao == "substituir":
        pos = random.randint(0, len(chars) - 1)
        novas = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != chars[pos].lower()]
        chars[pos] = random.choice(novas)

    elif operacao == "deletar" and len(chars) > 3:
        pos = random.randint(1, len(chars) - 2)
        chars.pop(pos)

    elif operacao == "inserir":
        pos = random.randint(1, len(chars) - 1)
        chars.insert(pos, random.choice("abcdefghijklmnopqrstuvwxyz"))

    elif operacao == "trocar" and len(chars) > 2:
        pos = random.randint(0, len(chars) - 2)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    return "".join(chars)


def processar_anamnese(entry, taxa):
    """Aplica ruído aleatório em uma anamnese limpa."""
    texto = entry["texto"]

    tokens_limpos = re.findall(r'\b[\w\-]+\b|[^\w\s]', texto)

    tokens_resultado = []
    labels = []

    for token in tokens_limpos:
        if (token.isalpha() and
            len(token) >= MIN_TOKEN_LEN and
            random.random() < taxa):

            token_perturbado = perturbar_caractere(token)
            tokens_resultado.append(token_perturbado)
            labels.append("TYPO")
        else:
            tokens_resultado.append(token)
            labels.append("CLEAN")

    texto_processado = " ".join(tokens_resultado)

    return {
        "id": entry.get("id", 0),
        "texto_limpo": texto,
        "texto_processado": texto_processado,
        "tokens": tokens_resultado,
        "labels": labels,
    }


def gerar_dataset_random(anamneses, taxa, config_name):
    """Gera um dataset completo com uma taxa de ruído específica."""
    dataset = []
    total_tokens = 0
    total_perturbados = 0

    for entry in anamneses:
        resultado = processar_anamnese(entry, taxa)
        resultado["config"] = config_name
        dataset.append(resultado)

        total_tokens += len(resultado["labels"])
        total_perturbados += resultado["labels"].count("TYPO")

    taxa_real = total_perturbados / max(total_tokens, 1)
    return dataset, total_tokens, total_perturbados, taxa_real


def main():
    random.seed(SEED)

    print("[1/2] Carregando anamneses limpas...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        anamneses = json.load(f)
    print(f"       {len(anamneses)} anamneses carregadas")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("[2/2] Gerando datasets RANDOM...")
    print()

    for nome_taxa, valor_taxa in TAXAS.items():
        config_name = f"RANDOM_{nome_taxa.upper()}"
        print(f"  Gerando {config_name} (taxa alvo: {valor_taxa*100:.0f}%)...")

        dataset, total_tok, total_pert, taxa_real = gerar_dataset_random(
            anamneses, valor_taxa, config_name
        )

        filename = f"dataset_random_{nome_taxa}.json"
        filepath = OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"    Tokens totais: {total_tok}")
        print(f"    Tokens perturbados: {total_pert}")
        print(f"    Taxa real: {taxa_real*100:.2f}%")
        print(f"    Arquivo: {filepath}")

        exemplo = dataset[0]
        ruidos = [(t, l) for t, l in zip(exemplo["tokens"], exemplo["labels"]) if l != "CLEAN"]
        print(f"    Exemplo (ID {exemplo['id']}): {len(ruidos)} tokens perturbados")
        for t, l in ruidos[:3]:
            print(f"      '{t}' → {l}")
        print()

    print("=" * 60)
    print("[✓] 3 datasets RANDOM gerados!")
    print(f"    dataset_random_taxa10.json (10%)")
    print(f"    dataset_random_taxa25.json (25%)")
    print(f"    dataset_random_taxa50.json (50%)")
    print("=" * 60)


if __name__ == "__main__":
    main()