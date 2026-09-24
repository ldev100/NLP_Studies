"""
medir_faltantes.py  (v2)

Subcomandos:

  densidade          densidade efetiva de ruido por split de treino
  lowercase          ablacao de caixa alta (rode com o checkpoint EPNI e com o REAL)
  particoes          recall do modelo por particao de conhecimento do vocabulario
                     (substitui o antigo "fora-do-top20")
  formas-exclusivas  monta a lista de formas que so existem na taxonomia por causa
                     dos 500 registros do teste, para o dicionario_baseline.py
  split-dev          escolhe a configuracao numa particao de desenvolvimento

    python medir_faltantes.py densidade --datasets datasets_experimento_ruido
    python medir_faltantes.py lowercase --ckpt modelos_finais/BERTimbau_best
    python medir_faltantes.py particoes --preds "resultados_experimento/predicoes/preds_*top20_rate50*.json"
    python medir_faltantes.py formas-exclusivas --corpus corpus_fonte.json
    python medir_faltantes.py split-dev --preds-dir resultados_experimento/predicoes
"""

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LABELS = ["CLEAN", "TYPO", "ABBREV"]
L2I = {l: i for i, l in enumerate(LABELS)}
CLEAN, TYPO, ABBREV = L2I["CLEAN"], L2I["TYPO"], L2I["ABBREV"]

TOKEN_RE = re.compile(r"\b[\w\-]+\b|[^\w\s]", re.UNICODE)  # mesmo de injetar_ruido_ngram.py


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _exigir(filepath, dica=""):
    if not Path(filepath).exists():
        raise SystemExit(f"[!] Arquivo nao encontrado: {filepath}\n    {dica}".rstrip())
    return filepath


def carregar_formas(filepath):
    """Conjunto de formas minusculas de um arquivo de taxonomia (varios formatos)."""
    _exigir(filepath, "Aponte para o arquivo da taxonomia da Fase 1 "
                      "(--taxonomia-abbrev / --taxonomia-typos).")
    data = json.load(open(filepath, encoding="utf-8"))
    formas = set()
    if isinstance(data, dict) and "tokens" in data:
        formas.update(str(t).lower() for t in data["tokens"])
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                t = e.get("token", e.get("word", e.get("forma", "")))
                if t:
                    formas.add(str(t).lower())
            else:
                formas.add(str(e).lower())
    elif isinstance(data, dict):
        formas.update(str(k).lower() for k in data if k not in ("total", "tokens", "config"))
    return formas


def carregar_top20(filepath):
    _exigir(filepath, "Aponte para o template_mapeamento_top20.json (--top20).")
    m = json.load(open(filepath, encoding="utf-8"))
    return ({k.lower() for k in m.get("abbreviations", {})},
            {k.lower() for k in m.get("typos", {})})


# ------------------------------------------------------------------ 1. densidade

def cmd_densidade(args):
    arquivos = sorted(glob.glob(str(Path(args.datasets) / "*.json")))
    if args.clean and Path(args.clean).exists():
        arquivos = [args.clean] + arquivos
    print(f"{'split':26s} {'docs':>6s} {'tokens':>9s} {'ABBREV':>8s} {'TYPO':>7s} {'ruido %':>8s}")
    linhas = []
    for f in arquivos:
        data = json.load(open(f, encoding="utf-8"))
        c = Counter(l for e in data for l in e["labels"])
        tot = sum(c.values())
        pct = 100 * (c["ABBREV"] + c["TYPO"]) / tot
        nome = Path(f).stem.replace("dataset_", "")
        print(f"{nome:26s} {len(data):6d} {tot:9d} {c['ABBREV']:8d} {c['TYPO']:7d} {pct:8.2f}")
        linhas.append({"split": nome, "tokens": tot, "abbrev": c["ABBREV"],
                       "typo": c["TYPO"], "densidade_pct": round(pct, 2)})
    json.dump(linhas, open("densidade_por_split.json", "w"), indent=2)
    print("\nCompare a coluna 'ruido %' com os 13,1% do teste antes de reescrever a 4.3.2 e a 5.2.")


# ----------------------------------------------------------------- 2. lowercase

HUB_FALLBACK = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "biobertpt": "pucpr/biobertpt-all",
}


def _resolver_ckpt(ckpt):
    """Aceita tanto a pasta do Trainer quanto o checkpoint-N de dentro dela."""
    p = Path(ckpt)
    if (p / "config.json").exists():
        return str(p)
    subs = [d for d in p.glob("checkpoint-*") if (d / "config.json").exists()]
    if subs:
        escolhido = max(subs, key=lambda d: int(d.name.rsplit("-", 1)[-1]))
        print(f"    checkpoint resolvido para {escolhido}")
        return str(escolhido)
    raise SystemExit(
        f"[!] Nao achei config.json em {ckpt} nem em subpastas checkpoint-*.\n"
        f"    Confira o caminho com: ls {ckpt}"
    )


def _carregar_tokenizer(ckpt, override=None, vocab_size=None, nome_original=None):
    """O script de treino antigo nao salvava o tokenizer junto do checkpoint.

    Tenta o proprio checkpoint, depois o override, depois o modelo do Hub
    correspondente ao nome da pasta. O tokenizador nao e ajustado no fine-tuning,
    entao o do Hub e identico ao usado no treino.
    """
    from transformers import AutoTokenizer
    tentativas = []
    if override:
        tentativas.append(override)
    tentativas.append(str(ckpt))
    nome = str(nome_original or ckpt).lower()
    for chave, hub in HUB_FALLBACK.items():
        if chave in nome and hub not in tentativas:
            tentativas.append(hub)

    for alvo in tentativas:
        try:
            tok = AutoTokenizer.from_pretrained(alvo)
        except Exception as e:
            print(f"    [!] tokenizer de '{alvo}': {type(e).__name__}")
            continue
        if vocab_size and len(tok) != vocab_size:
            print(f"    [!] '{alvo}' tem {len(tok)} tokens, o modelo espera "
                  f"{vocab_size}. Descartado.")
            continue
        if alvo != str(ckpt):
            print(f"    tokenizer carregado de '{alvo}' (o checkpoint nao tem)")
        return tok

    raise SystemExit(
        "[!] Nenhum tokenizer compativel. Passe --tokenizer com o id do Hub "
        "usado no treino, por exemplo:\n"
        "    --tokenizer neuralmind/bert-base-portuguese-cased"
    )


def _inferir_docs(ckpt, test_data, lower=False, tokenizer_id=None, max_len=512):
    """Roda o checkpoint no teste e devolve os documentos no MESMO formato que o
    fine_tuning_v2.py grava, para que particoes, split-dev e reavaliar_word_level
    leiam tudo igual."""
    import torch
    from transformers import AutoModelForTokenClassification
    ckpt_original, ckpt = ckpt, _resolver_ckpt(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(ckpt)
    model.eval()
    tok = _carregar_tokenizer(ckpt, tokenizer_id, model.config.vocab_size,
                              nome_original=ckpt_original)
    docs, truncados = [], 0
    with torch.no_grad():
        for e in test_data:
            toks = [t.lower() for t in e["tokens"]] if lower else e["tokens"]
            enc = tok(toks, is_split_into_words=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            wids = enc.word_ids(batch_index=0)
            out = model(**enc).logits[0].argmax(-1).tolist()
            sub_pred, sub_gold, por_palavra = [], [], defaultdict(list)
            for pos, w in enumerate(wids):
                if w is None or w >= len(e["labels"]):
                    continue
                sub_pred.append(int(out[pos]))
                sub_gold.append(L2I[e["labels"][w]])
                por_palavra[w].append(int(out[pos]))
            if len(por_palavra) < len(e["tokens"]):
                truncados += len(e["tokens"]) - len(por_palavra)
            word_pred, word_gold, word_form = [], [], []
            for w in range(len(e["tokens"])):
                if w not in por_palavra:
                    continue
                ps = por_palavra[w]
                c = Counter(ps); topo = max(c.values())
                emp = [k for k, v in c.items() if v == topo]
                word_pred.append(emp[0] if len(emp) == 1 else ps[0])
                word_gold.append(L2I[e["labels"][w]])
                word_form.append(e["tokens"][w].lower())
            docs.append({"sub_pred": sub_pred, "sub_gold": sub_gold,
                         "word_pred": word_pred, "word_gold": word_gold,
                         "word_form": word_form})
    if truncados:
        print(f"    [!] {truncados} palavras fora por truncagem em {max_len} sub-tokens")
    else:
        print(f"    nenhuma palavra perdida por truncagem (max_len={max_len})")
    return docs


def _achatar(docs):
    return ([g for d in docs for g in d["word_gold"]],
            [p for d in docs for p in d["word_pred"]])


def cmd_gerar_preds(args):
    """Gera o preds_*.json a partir de um checkpoint que ja existe.

    Serve para rodar particoes e split-dev antes do grid novo, usando o que
    sobrou em modelos_finais/.
    """
    test_data = json.load(open(args.test, encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = _inferir_docs(args.ckpt, test_data, tokenizer_id=args.tokenizer,
                         max_len=args.max_len)
    destino = out_dir / f"preds_{args.nome}.json"
    json.dump(docs, open(destino, "w"))
    gold, pred = _achatar(docs)
    g, p = np.array(gold), np.array(pred)
    P, R, F = prf(int(((g == ABBREV) & (p == ABBREV)).sum()),
                  int(((g != ABBREV) & (p == ABBREV)).sum()),
                  int(((g == ABBREV) & (p != ABBREV)).sum()))
    print(f"    {len(docs)} documentos -> {destino}")
    print(f"    ABBREV por palavra: P={P:.3f} R={R:.3f} F1={F:.3f}")


def cmd_lowercase(args):
    test_data = json.load(open(args.test, encoding="utf-8"))
    for lower in [False, True]:
        gold, pred = _achatar(
            _inferir_docs(args.ckpt, test_data, lower, tokenizer_id=args.tokenizer,
                          max_len=args.max_len))
        g, p = np.array(gold), np.array(pred)
        tp = int(((g == ABBREV) & (p == ABBREV)).sum())
        fp = int(((g != ABBREV) & (p == ABBREV)).sum())
        fn = int(((g == ABBREV) & (p != ABBREV)).sum())
        P, R, F = prf(tp, fp, fn)
        print(f"  {'minusculas' if lower else 'original  '}  "
              f"ABBREV  P={P:.3f}  R={R:.3f}  F1={F:.3f}")
    print("\nA queda de recall entre as duas linhas e o numero que falta na 5.4.")


# ------------------------------------------------- 3. particoes de conhecimento

def cmd_particoes(args):
    """Quebra os tokens ABBREV do gold em tres particoes e mede o recall em cada uma.

      A  no Top-20                     visto no treino; os dois dicionarios acertam
      B  na taxonomia, fora do Top-20  nao visto no treino; so o DICT-Full acerta
      C  fora da taxonomia             desconhecido dos dois, recall 0 por construcao

    A coluna C e a que sustenta o artigo depois que o DICT-Full passou o modelo:
    sao os tokens que nenhum casamento exato alcanca, qualquer que seja o tamanho
    do dicionario.
    """
    top20_abbrev, _ = carregar_top20(args.top20)
    taxonomia = carregar_formas(args.taxonomia_abbrev)
    print(f"Top-20: {len(top20_abbrev)} formas | taxonomia completa: {len(taxonomia)} formas\n")

    def particao(forma):
        if forma in top20_abbrev:
            return "A_top20"
        if forma in taxonomia:
            return "B_taxonomia"
        return "C_fora"

    arquivos = sorted(glob.glob(args.preds))
    if not arquivos:
        print(f"[!] Nenhum arquivo em {args.preds}")
        pasta = Path(args.preds).parent
        existentes = sorted(glob.glob(str(pasta / "preds_*.json"))) if pasta.exists() else []
        if existentes:
            print("    Disponiveis nessa pasta:")
            for f in existentes:
                print(f"      {Path(f).name}")
            print("    Ajuste o --preds.")
        else:
            print("    Nenhum preds_*.json existe ainda. Gere um a partir de um "
                  "checkpoint que voce ja tem:")
            print("      python medir_faltantes.py gerar-preds \\")
            print("          --ckpt modelos_finais/BERTimbau_best \\")
            print("          --nome BERTimbau_top20_rate50_ckptantigo")
        return

    por_particao = defaultdict(list)
    prec_fora_top20, prec_fora_tax = [], []
    n_tokens = {}

    for arq in arquivos:
        docs = json.load(open(arq))
        acertos, totais = Counter(), Counter()
        tp_ft = fp_ft = tp_fx = fp_fx = 0
        for d in docs:
            for forma, g, p in zip(d["word_form"], d["word_gold"], d["word_pred"]):
                if g == ABBREV:
                    part = particao(forma)
                    totais[part] += 1
                    acertos[part] += int(p == ABBREV)
                if p == ABBREV and forma not in top20_abbrev:
                    tp_ft += int(g == ABBREV); fp_ft += int(g != ABBREV)
                if p == ABBREV and forma not in taxonomia:
                    tp_fx += int(g == ABBREV); fp_fx += int(g != ABBREV)
        for part in ["A_top20", "B_taxonomia", "C_fora"]:
            if totais[part]:
                por_particao[part].append(acertos[part] / totais[part])
                n_tokens[part] = totais[part]
        prec_fora_top20.append(tp_ft / (tp_ft + fp_ft) if tp_ft + fp_ft else 0.0)
        prec_fora_tax.append(tp_fx / (tp_fx + fp_fx) if tp_fx + fp_fx else 0.0)
        print(f"  lido: {Path(arq).stem}")

    total = sum(n_tokens.values())
    dicts = {"A_top20": (1.0, 1.0), "B_taxonomia": (0.0, 1.0), "C_fora": (0.0, 0.0)}
    rotulo = {"A_top20": "A no Top-20", "B_taxonomia": "B na taxonomia",
              "C_fora": "C fora de tudo"}
    print(f"\n{'particao':16s} {'# tokens':>9s} {'% total':>8s} "
          f"{'recall modelo':>17s} {'DICT-Top20':>11s} {'DICT-Full':>10s}")
    linhas = {}
    for part in ["A_top20", "B_taxonomia", "C_fora"]:
        if part not in n_tokens:
            continue
        v = por_particao[part]
        media = float(np.mean(v))
        desvio = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        d20, dfull = dicts[part]
        print(f"{rotulo[part]:16s} {n_tokens[part]:9d} {100*n_tokens[part]/total:7.1f}% "
              f"{media:11.3f} ±{desvio:.3f} {d20:11.3f} {dfull:10.3f}")
        linhas[part] = {"n": n_tokens[part], "recall_modelo": round(media, 4),
                        "std": round(desvio, 4), "dict_top20": d20, "dict_full": dfull}

    print(f"\nPrecisao do modelo restrita as formas fora do Top-20:    "
          f"{np.mean(prec_fora_top20):.3f}")
    print(f"Precisao do modelo restrita as formas fora da taxonomia: "
          f"{np.mean(prec_fora_tax):.3f}")
    print("Compare com os 0.42 do regex de caixa alta, nao com a precisao geral de 0.897.")
    json.dump({"particoes": linhas,
               "precisao_fora_top20": round(float(np.mean(prec_fora_top20)), 4),
               "precisao_fora_taxonomia": round(float(np.mean(prec_fora_tax)), 4)},
              open("particoes_vocabulario.json", "w"), indent=2, ensure_ascii=False)
    print("\nSalvo em particoes_vocabulario.json")


# --------------------------------------------- 4. formas exclusivas do teste

def _texto_dos_registros(path, campo=None):
    """Extrai uma lista de textos de um arquivo de corpus, em varios formatos."""
    if not Path(path).exists():
        raise SystemExit(
            f"[!] Arquivo nao encontrado: {path}\n"
            f"    --corpus precisa apontar para o arquivo real das 8.411 anamneses\n"
            f"    (o que alimentou a Fase 1). Aceita .json, .csv ou .txt.\n"
            f"    Ex.: --corpus ../fase1/anamneses.csv --campo-texto texto"
        )
    if path.endswith(".csv") or path.endswith(".tsv"):
        import csv as _csv
        sep = "\t" if path.endswith(".tsv") else ","
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            linhas = list(_csv.DictReader(fh, delimiter=sep))
        if not linhas:
            return []
        colunas = list(linhas[0].keys())
        if campo and campo not in colunas:
            raise SystemExit(f"[!] Coluna '{campo}' nao existe. Colunas: {colunas}")
        if not campo:
            campo = max(colunas, key=lambda c: np.mean(
                [len(str(l.get(c) or "")) for l in linhas[:200]]))
            print(f"    coluna de texto inferida: '{campo}'")
        return [str(l.get(campo) or "") for l in linhas]
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        textos = []
        for e in data:
            if isinstance(e, str):
                textos.append(e)
            elif isinstance(e, dict):
                if campo and campo in e:
                    textos.append(str(e[campo]))
                elif isinstance(e.get("tokens"), list):
                    textos.append(" ".join(map(str, e["tokens"])))
                else:
                    cand = [v for v in e.values() if isinstance(v, str) and len(v) > 40]
                    textos.append(max(cand, key=len) if cand else "")
        return textos
    return list(open(path, encoding="utf-8", errors="replace"))


def cmd_formas_exclusivas(args):
    """Lista as formas da taxonomia que so aparecem nos 500 registros do teste.

    Sem contagem por forma nos arquivos de taxonomia, esta e a via para o
    dicionario_baseline.py --excluir-formas.
    """
    taxonomia = carregar_formas(args.taxonomia_abbrev) | carregar_formas(args.taxonomia_typos)
    print(f"Taxonomia: {len(taxonomia)} formas")

    test_data = json.load(open(args.test, encoding="utf-8"))
    no_teste = Counter(t.lower() for e in test_data for t in e["tokens"])

    textos = _texto_dos_registros(args.corpus, args.campo_texto)
    print(f"Corpus-fonte: {len(textos)} registros")
    no_corpus = Counter()
    for texto in textos:
        no_corpus.update(t.lower() for t in TOKEN_RE.findall(texto))

    exclusivas, sem_ocorrencia = [], 0
    for forma in sorted(taxonomia):
        c_corpus = no_corpus.get(forma, 0)
        c_teste = no_teste.get(forma, 0)
        fora_do_teste = c_corpus if args.corpus_sem_teste else c_corpus - c_teste
        if c_teste == 0:
            continue  # nao afeta o dicionario neste teste
        if fora_do_teste <= 0:
            exclusivas.append(forma)
        if c_corpus == 0:
            sem_ocorrencia += 1

    json.dump(exclusivas, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n{len(exclusivas)} formas da taxonomia so ocorrem nos registros do teste.")
    if exclusivas:
        print("Exemplos:", ", ".join(exclusivas[:15]))
    if sem_ocorrencia:
        print(f"[!] {sem_ocorrencia} formas nao foram encontradas no corpus com esta "
              f"tokenizacao. Confira --campo-texto antes de confiar no resultado.")
    print(f"\nSalvo em {args.out}. Agora rode:")
    print(f"  python dicionario_baseline.py --modo ruidosa --excluir-formas {args.out}")
    if not args.corpus_sem_teste:
        print("\n(assumindo que o corpus-fonte inclui os 500 registros do teste; "
              "se nao incluir, repita com --corpus-sem-teste)")


# ---------------------------------------------------------------- 5. split dev

def cmd_split_dev(args):
    arquivos = sorted(glob.glob(str(Path(args.preds_dir) / "preds_*.json")))
    if not arquivos:
        print("Nenhum preds_*.json encontrado. Rode o grid com o fine_tuning_v2.py.")
        return

    n_docs = len(json.load(open(arquivos[0])))
    rng = np.random.default_rng(args.seed)
    dev_idx = set(rng.choice(n_docs, size=int(n_docs * args.frac), replace=False).tolist())
    print(f"dev: {len(dev_idx)} anamneses | teste: {n_docs - len(dev_idx)} | seed {args.seed}")

    def f1_abbrev(gold, pred):
        g, p = np.array(gold), np.array(pred)
        return prf(int(((g == ABBREV) & (p == ABBREV)).sum()),
                   int(((g != ABBREV) & (p == ABBREV)).sum()),
                   int(((g == ABBREV) & (p != ABBREV)).sum()))[2]

    por_config = defaultdict(list)
    for arq in arquivos:
        nome = Path(arq).stem.replace("preds_", "")
        partes = nome.split("_")
        modelo, config = partes[0], "_".join(partes[1:-1])
        docs = json.load(open(arq))
        res = {}
        for parte, escolha in [("dev", True), ("test", False)]:
            g, p = [], []
            for i, d in enumerate(docs):
                if (i in dev_idx) is not escolha:
                    continue
                g += d["word_gold"]; p += d["word_pred"]
            res[parte] = f1_abbrev(g, p)
        por_config[(modelo, config)].append(res)

    print(f"\n{'modelo':12s} {'config':16s} {'F1 dev':>8s} {'F1 teste':>10s}")
    medias = {}
    for (modelo, config), runs in sorted(por_config.items()):
        dev = float(np.mean([r["dev"] for r in runs]))
        test = float(np.mean([r["test"] for r in runs]))
        medias[(modelo, config)] = (dev, test)
        print(f"{modelo:12s} {config:16s} {dev:8.4f} {test:10.4f}")

    print()
    for modelo in sorted({m for m, _ in medias}):
        cand = {c: v for (m, c), v in medias.items() if m == modelo}
        escolhida = max(cand, key=lambda c: cand[c][0])
        melhor = max(cand, key=lambda c: cand[c][1])
        print(f"{modelo}: escolhida no dev -> {escolhida} "
              f"(dev {cand[escolhida][0]:.4f}, teste {cand[escolhida][1]:.4f})")
        print(f"{modelo}: maxima no teste  -> {melhor} ({cand[melhor][1]:.4f})")
    json.dump({f"{m}|{c}": v for (m, c), v in medias.items()},
              open("selecao_dev_teste.json", "w"), indent=2)
    print("\nReporte a linha 'escolhida no dev'.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("densidade")
    a.add_argument("--datasets", default="datasets_experimento_ruido")
    a.add_argument("--clean", default="clean_synthetic_anamneses_gold_standart_tokens.json")
    a.set_defaults(func=cmd_densidade)

    b = sub.add_parser("lowercase")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--test", default="test_set_real.json")
    b.add_argument("--tokenizer", default=None,
                   help="id do Hub, se o checkpoint nao tiver os arquivos do tokenizer")
    b.add_argument("--max-len", type=int, default=512,
                   help="512 avalia o documento inteiro; 300 reproduz o corte antigo")
    b.set_defaults(func=cmd_lowercase)

    c = sub.add_parser("particoes", aliases=["fora-do-top20"])
    c.add_argument("--preds", default="resultados_experimento/predicoes/preds_*top20_rate50*.json")
    c.add_argument("--top20", default="template_mapeamento_top20.json")
    c.add_argument("--taxonomia-abbrev", default="potential_abbreviations.json")
    c.set_defaults(func=cmd_particoes)

    d = sub.add_parser("formas-exclusivas")
    d.add_argument("--corpus", required=True, help="arquivo do corpus-fonte (8.411 anamneses)")
    d.add_argument("--campo-texto", default=None, help="nome do campo de texto, se for JSON")
    d.add_argument("--test", default="test_set_real.json")
    d.add_argument("--taxonomia-abbrev", default="potential_abbreviations.json")
    d.add_argument("--taxonomia-typos", default="potential_typos.json")
    d.add_argument("--corpus-sem-teste", action="store_true",
                   help="use se o corpus-fonte NAO contiver os 500 registros do teste")
    d.add_argument("--out", default="formas_exclusivas_do_teste.json")
    d.set_defaults(func=cmd_formas_exclusivas)

    g = sub.add_parser("gerar-preds")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--nome", required=True,
                   help="ex.: BERTimbau_top20_rate50_seed42 (vira preds_<nome>.json)")
    g.add_argument("--test", default="test_set_real.json")
    g.add_argument("--tokenizer", default=None,
                   help="id do Hub, se o checkpoint nao tiver os arquivos do tokenizer")
    g.add_argument("--max-len", type=int, default=512,
                   help="512 avalia o documento inteiro; 300 reproduz o corte antigo")
    g.add_argument("--out-dir", default="resultados_experimento/predicoes")
    g.set_defaults(func=cmd_gerar_preds)

    e = sub.add_parser("split-dev")
    e.add_argument("--preds-dir", default="resultados_experimento/predicoes")
    e.add_argument("--frac", type=float, default=0.3)
    e.add_argument("--seed", type=int, default=20260922)
    e.set_defaults(func=cmd_split_dev)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()