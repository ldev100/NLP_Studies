import json
from collections import Counter

TEST_FILE = "test_set_real.json"
MAPPING_FILE = "template_mapeamento_top20.json"

def main():
    print("[1/3] Carregando test set real...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"       {len(test_data)} anamneses")

    print("[2/3] Carregando vocabulário top-20...")
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    vocab_abbrev = set()
    for token in mapping.get("abbreviations", {}):
        vocab_abbrev.add(token.lower())

    vocab_typo = set()
    for token in mapping.get("typos", {}):
        vocab_typo.add(token.lower())

    vocab_total = vocab_abbrev | vocab_typo

    print(f"       {len(vocab_abbrev)} abreviações no top-20")
    print(f"       {len(vocab_typo)} typos no top-20")
    print(f"       {len(vocab_total)} tokens únicos total")

    print("[3/3] Calculando cobertura...")
    print()

    total_clean = 0
    total_abbrev = 0
    total_typo = 0

    abbrev_in_vocab = 0
    abbrev_not_in_vocab = 0
    typo_in_vocab = 0
    typo_not_in_vocab = 0

    abbrev_missed = Counter()
    typo_missed = Counter()
    abbrev_covered = Counter()
    typo_covered = Counter()

    for entry in test_data:
        for token, label in zip(entry["tokens"], entry["labels"]):
            token_lower = token.lower()

            if label == "CLEAN":
                total_clean += 1

            elif label == "ABBREV":
                total_abbrev += 1
                if token_lower in vocab_abbrev:
                    abbrev_in_vocab += 1
                    abbrev_covered[token_lower] += 1
                else:
                    abbrev_not_in_vocab += 1
                    abbrev_missed[token_lower] += 1

            elif label == "TYPO":
                total_typo += 1
                if token_lower in vocab_typo:
                    typo_in_vocab += 1
                    typo_covered[token_lower] += 1
                else:
                    typo_not_in_vocab += 1
                    typo_missed[token_lower] += 1

    total_noise = total_abbrev + total_typo
    noise_in_vocab = abbrev_in_vocab + typo_in_vocab
    noise_not_in_vocab = abbrev_not_in_vocab + typo_not_in_vocab

    print("=" * 70)
    print("COBERTURA DO VOCABULÁRIO TOP-20 NO TEST SET REAL")
    print("=" * 70)
    print()

    print(f"  Total de tokens no test set: {total_clean + total_noise}")
    print(f"    CLEAN:  {total_clean} ({100*total_clean/(total_clean+total_noise):.1f}%)")
    print(f"    ABBREV: {total_abbrev} ({100*total_abbrev/(total_clean+total_noise):.1f}%)")
    print(f"    TYPO:   {total_typo} ({100*total_typo/(total_clean+total_noise):.1f}%)")
    print()

    print("-" * 70)
    print("COBERTURA POR CATEGORIA")
    print("-" * 70)
    print()

    if total_abbrev > 0:
        pct_abbrev = 100 * abbrev_in_vocab / total_abbrev
        print(f"  ABBREV: {abbrev_in_vocab}/{total_abbrev} cobertos ({pct_abbrev:.1f}%)")
        print(f"    Cobertos:     {abbrev_in_vocab}")
        print(f"    Não cobertos: {abbrev_not_in_vocab}")
    else:
        pct_abbrev = 0
        print(f"  ABBREV: nenhum token ABBREV no test set")

    print()

    if total_typo > 0:
        pct_typo = 100 * typo_in_vocab / total_typo
        print(f"  TYPO: {typo_in_vocab}/{total_typo} cobertos ({pct_typo:.1f}%)")
        print(f"    Cobertos:     {typo_in_vocab}")
        print(f"    Não cobertos: {typo_not_in_vocab}")
    else:
        pct_typo = 0
        print(f"  TYPO: nenhum token TYPO no test set")

    print()

    if total_noise > 0:
        pct_total = 100 * noise_in_vocab / total_noise
        print(f"  TOTAL NOISE: {noise_in_vocab}/{total_noise} cobertos ({pct_total:.1f}%)")
        print(f"    => Teto teórico de recall: {pct_total:.1f}%")
    print()

    print("-" * 70)
    print("TOP 20 ABREVIAÇÕES NÃO COBERTAS (fora do top-20 de treino)")
    print("-" * 70)
    for token, count in abbrev_missed.most_common(20):
        print(f"    {token:20s} : {count} ocorrências")

    print()
    print("-" * 70)
    print("TOP 20 TYPOS NÃO COBERTOS (fora do top-20 de treino)")
    print("-" * 70)
    for token, count in typo_missed.most_common(20):
        print(f"    {token:20s} : {count} ocorrências")

    print()
    print("-" * 70)
    print("ABREVIAÇÕES COBERTAS (presentes no top-20)")
    print("-" * 70)
    for token, count in abbrev_covered.most_common():
        print(f"    {token:20s} : {count} ocorrências")

    print()
    print("-" * 70)
    print("TYPOS COBERTOS (presentes no top-20)")
    print("-" * 70)
    for token, count in typo_covered.most_common():
        print(f"    {token:20s} : {count} ocorrências")

    print()
    print("=" * 70)
    print("TRECHO PARA O PAPER (Discussion)")
    print("=" * 70)
    print()
    print(f"An analysis of vocabulary coverage reveals that")
    print(f"{noise_in_vocab} of {total_noise} noise tokens in the")
    print(f"test set ({pct_total:.1f}%) correspond to patterns present")
    print(f"in the top-20 training vocabulary. For abbreviations,")
    print(f"coverage is {pct_abbrev:.1f}% ({abbrev_in_vocab}/{total_abbrev}),")
    print(f"while for typos it is {pct_typo:.1f}%")
    print(f"({typo_in_vocab}/{total_typo}). This establishes a")
    print(f"theoretical recall ceiling of {pct_total:.1f}% for the")
    print(f"EPNI framework with the current vocabulary size,")
    print(f"explaining the gap between precision (above 0.92)")
    print(f"and recall observed in the results.")


if __name__ == "__main__":
    main()