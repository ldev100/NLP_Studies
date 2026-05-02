import json
import random
import re
from pathlib import Path
from tqdm import tqdm

INPUT_ANAMNESES = "clean_synthetic_anamneses_gold_standart.json"
ARQUIVOS_MAPEAMENTO = {
    "TOP10": "template_mapeamento_top10.json",
    "TOP15": "template_mapeamento_top15.json",
    "TOP20": "template_mapeamento_top20.json"
}
OUTPUT_DIR = "datasets_experimento_ruido"

TAXAS_PROGRESSIVAS = [0.10, 0.25, 0.50] 
RANDOM_SEED = 42

def preparar_mapa_reverso(caminho_json):
    """
    Transforma o JSON (erro -> significado) em (correto -> erro).
    """
    try:
        with open(caminho_json, "r", encoding="utf-8") as f:
            dados = json.load(f)
        
        mapa_reverso = {}
        
        for categoria in ["typos", "abbreviations"]:
            for erro, info in dados.get(categoria, {}).items():
                correto = info["significado"].lower()
                
                if correto not in mapa_reverso or info["count"] > mapa_reverso[correto]["count"]:
                    mapa_reverso[correto] = {
                        "substituta": erro,
                        "tipo": "ABBREV" if categoria == "abbreviations" else "TYPO",
                        "count": info["count"]
                    }
        return mapa_reverso
    except FileNotFoundError:
        print(f"[!] Erro: Arquivo {caminho_json} não encontrado.")
        return None

def injetar_ruido(texto, mapa_reverso, taxa):
    """Percorre o texto e decide, via probabilidade, se injeta o ruído."""
    palavras = re.findall(r'\b\w+\b|\S', texto)
    tokens_resultado = []
    labels = []
    
    for palavra in palavras:
        palavra_lower = palavra.lower()
        
        if palavra_lower in mapa_reverso and random.random() < taxa:
            info = mapa_reverso[palavra_lower]
            subst = info["substituta"]
            
            final = subst.capitalize() if palavra[0].isupper() else subst
            
            tokens_resultado.append(final)
            labels.append(info["tipo"])
        else:
            tokens_resultado.append(palavra)
            labels.append("CLEAN")
            
    return " ".join(tokens_resultado), tokens_resultado, labels

def main():
    random.seed(RANDOM_SEED)
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(exist_ok=True)
    
    print("[1/3] Carregando anamneses limpas...")
    with open(INPUT_ANAMNESES, "r", encoding="utf-8") as f:
        anamneses = json.load(f)

    print("[2/3] Iniciando processamento dos 9 datasets...")

    for label_top, arquivo_json in ARQUIVOS_MAPEAMENTO.items():
        mapa_reverso = preparar_mapa_reverso(arquivo_json)
        if not mapa_reverso: continue
        
        for taxa in TAXAS_PROGRESSIVAS:
            tag_config = f"{label_top}_TAXA{int(taxa*100)}"
            print(f"\n>>> Gerando Dataset: {tag_config}")
            
            dataset_final = []
            for entry in tqdm(anamneses, desc=f"Injetando {tag_config}"):
                txt_p, tokens, labels = injetar_ruido(entry["texto"], mapa_reverso, taxa)
                
                dataset_final.append({
                    "id": entry["id"],
                    "texto_limpo": entry["texto"],
                    "texto_processado": txt_p,
                    "tokens": tokens,
                    "labels": labels,
                    "config": tag_config
                })
            
            nome_arquivo = f"dataset_{tag_config.lower()}.json"
            with open(out_path / nome_arquivo, "w", encoding="utf-8") as f:
                json.dump(dataset_final, f, ensure_ascii=False, indent=2)

    print(f"\n[3/3] Sucesso! Verifique a pasta '{OUTPUT_DIR}' para os resultados.")

if __name__ == "__main__":
    main()