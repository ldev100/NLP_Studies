import re
import json
import glob
import random
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
import ollama

TEST_REAL_FILE  = Path("test_set_real.json")
DATASET_DIR     = Path("datasets_experimento_ruido_1196")
FEWSHOT_GLOB    = str(DATASET_DIR / "dataset_top20_rate10.json")   # fonte dos shots (densidade ~ real)
ABBREV_TAXONOMY = Path("potential_abbreviations.json")
TYPO_TAXONOMY   = Path("potential_typos.json")

MODELS  = ["qwen2.5:7b", "qwen2.5:14b"]   
SEEDS   = [42, 123, 456]                  
K_SHOT  = 4                              
TOP_K   = 20                              
NUM_CTX = 8192                            
MAX_EXAMPLE_TOKENS = 120                  
KEEP_ALIVE = "15m"                        

OUT_JSON  = Path("llm_baseline_results.json")
OUT_LATEX = Path("llm_baseline_latex.txt")

LABEL_LIST = ["CLEAN", "TYPO", "ABBREV"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
NOISE_LABELS = {"ABBREV", "TYPO"}

SCHEMA = {
    "type": "object",
    "properties": {
        "noise": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "label": {"type": "string", "enum": ["ABBREV", "TYPO"]},
                },
                "required": ["i", "label"],
            },
        }
    },
    "required": ["noise"],
}

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rep = classification_report(
        y_true, y_pred, labels=[0, 1, 2],
        target_names=LABEL_LIST, output_dict=True, zero_division=0,
    )
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2], average="macro", zero_division=0,
    )
    bt = np.where(y_true == 0, 0, 1)
    bp = np.where(y_pred == 0, 0, 1)
    brep = classification_report(
        bt, bp, labels=[0, 1],
        target_names=["CLEAN", "NOISE"], output_dict=True, zero_division=0,
    )
    return {
        "abbrev_precision": rep["ABBREV"]["precision"],
        "abbrev_recall":    rep["ABBREV"]["recall"],
        "abbrev_f1":        rep["ABBREV"]["f1-score"],
        "typo_precision":   rep["TYPO"]["precision"],
        "typo_recall":      rep["TYPO"]["recall"],
        "typo_f1":          rep["TYPO"]["f1-score"],
        "macro_precision_noise": float(mp),
        "macro_recall_noise":    float(mr),
        "macro_f1_noise":        float(mf1),
        "binary_noise_precision": brep["NOISE"]["precision"],
        "binary_noise_recall":    brep["NOISE"]["recall"],
        "binary_noise_f1":        brep["NOISE"]["f1-score"],
        "overall_accuracy": rep["accuracy"],
    }


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def topk_surfaces(path, k):
    if not Path(path).exists():
        return []
    data = load_json(path)
    items = sorted(data["tokens"].items(),
                   key=lambda kv: kv[1]["count"], reverse=True)
    return [s for s, _ in items[:k]]


def load_top20_vocab():
    return {"abbrev": topk_surfaces(ABBREV_TAXONOMY, TOP_K),
            "typo":   topk_surfaces(TYPO_TAXONOMY, TOP_K)}


def load_fewshot_pool():
    files = sorted(glob.glob(FEWSHOT_GLOB))
    if not files:
        files = sorted(glob.glob(str(DATASET_DIR / "dataset_top*.json")))
    if not files:
        raise SystemExit(f"ERRO: nenhuma fonte de few-shot em {DATASET_DIR}/ "
                         f"(esperado {FEWSHOT_GLOB}).")
    return load_json(files[0])


def pick_fewshot(pool, k, seed):
    """k exemplos do sintetico rate10 (densidade ~ real). Garante que ABBREV e
    TYPO aparecam pelo menos uma vez (cobertura de classe), e preenche o resto
    aleatoriamente -> demonstracao representativa, sem viesar a densidade."""
    rng = random.Random(seed)
    cand = [ex for ex in pool if len(ex["tokens"]) <= MAX_EXAMPLE_TOKENS] or list(pool)
    rng.shuffle(cand)
    chosen, have_ab, have_ty = [], False, False
    for ex in cand:                       # 1a passada: cobre cada classe
        if len(chosen) >= k:
            break
        a, t = "ABBREV" in ex["labels"], "TYPO" in ex["labels"]
        if (a and not have_ab) or (t and not have_ty):
            chosen.append(ex)
            have_ab |= a
            have_ty |= t
    for ex in cand:                       # 2a passada: preenche aleatorio
        if len(chosen) >= k:
            break
        if ex not in chosen:
            chosen.append(ex)
    return chosen[:k]


def system_prompt(vocab=None):
    s = (
        "Voce e um anotador de texto clinico em portugues do Brasil. Recebe uma "
        "lista NUMERADA de tokens, um por linha no formato 'indice<TAB>token'. "
        "A MAIORIA dos tokens e CLEAN (corretos, escritos por extenso). "
        "Identifique APENAS os tokens que sao ruido:\n"
        "- ABBREV: abreviacao ou sigla medica (ex.: BEG, PA, MMII, AP, AC).\n"
        "- TYPO: erro de digitacao ou ortografia (ex.: 'hipogastro' por 'hipogastrio').\n"
        "Responda APENAS com um objeto JSON {\"noise\": [{\"i\": <indice>, "
        "\"label\": \"ABBREV\"|\"TYPO\"}, ...]} contendo somente os tokens com ruido. "
        "Todo indice que voce NAO citar e considerado CLEAN. Se nao houver ruido, "
        "responda {\"noise\": []}. Copie o indice exatamente como aparece ao lado do token."
    )
    if vocab and (vocab.get("abbrev") or vocab.get("typo")):
        s += (f"\n\nListas de referencia (podem aparecer no texto):"
              f"\n- Abreviacoes conhecidas: {', '.join(vocab['abbrev'])}."
              f"\n- Erros conhecidos: {', '.join(vocab['typo'])}.")
    return s


def numbered(tokens):
    return "\n".join(f"{i}\t{t}" for i, t in enumerate(tokens))


def noise_payload(tokens, labels):
    noise = [{"i": i, "label": lab}
             for i, (t, lab) in enumerate(zip(tokens, labels))
             if lab in NOISE_LABELS]
    return json.dumps({"noise": noise}, ensure_ascii=False)


def shot_to_messages(ex):
    return [
        {"role": "user", "content": numbered(ex["tokens"])},
        {"role": "assistant", "content": noise_payload(ex["tokens"], ex["labels"])},
    ]


def build_messages(test_tokens, shots, vocab):
    msgs = [{"role": "system", "content": system_prompt(vocab)}]
    for ex in shots:
        msgs += shot_to_messages(ex)
    msgs.append({"role": "user", "content": numbered(test_tokens)})
    return msgs


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_ITEM_RE_A = re.compile(r'\{\s*"i"\s*:\s*(\d+)\s*,\s*"label"\s*:\s*"(ABBREV|TYPO)"\s*\}')
_ITEM_RE_B = re.compile(r'\{\s*"label"\s*:\s*"(ABBREV|TYPO)"\s*,\s*"i"\s*:\s*(\d+)\s*\}')


def _salvage_items(s):
    out = []
    for m in _ITEM_RE_A.finditer(s):
        out.append({"i": int(m.group(1)), "label": m.group(2)})
    for m in _ITEM_RE_B.finditer(s):
        out.append({"i": int(m.group(2)), "label": m.group(1)})
    return out


def _extract_content(resp):
    try:
        return resp.message.content             
    except AttributeError:
        return resp["message"]["content"]        


def llm_tag_sequence(tokens, shots, vocab, model, seed):
    """Retorna (labels, info). labels tem len == len(tokens); CLEAN por default."""
    n = len(tokens)
    msgs = build_messages(tokens, shots, vocab)
    opts = {"temperature": 0.0, "seed": seed, "top_p": 1.0,
            "num_ctx": NUM_CTX, "num_predict": 16 * n + 128}
    try:
        resp = ollama.chat(model=model, messages=msgs, format=SCHEMA,
                           options=opts, keep_alive=KEEP_ALIVE)
    except TypeError:
        resp = ollama.chat(model=model, messages=msgs, format="json",
                           options=opts, keep_alive=KEEP_ALIVE)

    content = _FENCE.sub("", _extract_content(resp).strip())
    labels = ["CLEAN"] * n

    strict_ok = True
    try:
        items = json.loads(content).get("noise", [])
    except Exception:
        items, strict_ok = [], False
    use = items if strict_ok else _salvage_items(content)

    n_pred, oob = 0, 0
    for it in use:
        n_pred += 1
        i, lab = it.get("i"), it.get("label")
        if isinstance(i, bool) or not isinstance(i, int):
            oob += 1
            continue
        if 0 <= i < n and lab in NOISE_LABELS:
            labels[i] = lab
        else:
            oob += 1

    info = {"parse_ok": strict_ok, "n_pred": n_pred, "oob": oob,
            "raw": None if strict_ok else content[:300]}
    return labels, info


def evaluate_run(test_data, shots, vocab, model, seed, verbose=True):
    y_true, y_pred = [], []
    parse_fail, total_oob, total_pred = 0, 0, 0
    raw_failures = []
    for j, ex in enumerate(test_data):
        pred, info = llm_tag_sequence(ex["tokens"], shots, vocab, model, seed)
        if not info["parse_ok"]:
            parse_fail += 1
            if len(raw_failures) < 3:
                raw_failures.append(info["raw"])
        total_oob += info["oob"]
        total_pred += info["n_pred"]
        for g, p in zip(ex["labels"], pred):
            y_true.append(LABEL2ID[g])
            y_pred.append(LABEL2ID[p])
        if verbose and (j + 1) % 50 == 0:
            print(f"      ... {j + 1}/{len(test_data)} anamneses")
    m = compute_metrics(y_true, y_pred)
    nrec = max(1, len(test_data))
    m["parse_fail_rate"] = parse_fail / nrec          
    m["oob_per_record"] = total_oob / nrec           
    m["pred_noise_per_record"] = total_pred / nrec    
    m["_raw_failures"] = raw_failures                 
    return m


def aggregate(per_seed):
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed], dtype=float)
        agg[k + "_mean"] = float(vals.mean())
        agg[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return agg

def latex_row_tabdict(name, a):
    return (f"{name} & "
            f"{a['abbrev_precision_mean']:.3f} & {a['abbrev_recall_mean']:.3f} & "
            f"{a['abbrev_f1_mean']:.3f} & "
            f"{a['typo_precision_mean']:.3f} & {a['typo_recall_mean']:.3f} & "
            f"{a['typo_f1_mean']:.3f} & {a['binary_noise_f1_mean']:.3f} \\\\")


def pretty_name(model, cond):
    base = model.replace("qwen2.5:", "Qwen2.5-").replace("qwen3:", "Qwen3-")
    base = base[:-1] + "B" if base.endswith("b") else base
    tag = "cold" if cond == "cold" else f"+Top{TOP_K}"
    return f"LLM {base} ({tag})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--k", type=int, default=K_SHOT)
    ap.add_argument("--conditions", nargs="+", default=["cold", "top20"],
                    choices=["cold", "top20"])
    ap.add_argument("--limit", type=int, default=None,
                    help="usa apenas as N primeiras anamneses (teste rapido)")
    args = ap.parse_args()

    if not TEST_REAL_FILE.exists():
        raise SystemExit(f"ERRO: {TEST_REAL_FILE} nao encontrado.")
    test_data = load_json(TEST_REAL_FILE)
    if args.limit:
        test_data = test_data[:args.limit]
    print(f"Test set: {len(test_data)} anamneses")
    print(f"Modelos: {args.models} | seeds: {args.seeds} | k={args.k} | "
          f"condicoes: {args.conditions}")

    pool = load_fewshot_pool()
    top20 = load_top20_vocab()
    print(f"Few-shot pool: {len(pool)} exemplos sinteticos | "
          f"Top{TOP_K}: {len(top20['abbrev'])} ABBREV / {len(top20['typo'])} TYPO")

    results = {}
    if OUT_JSON.exists():
        try:
            results = load_json(OUT_JSON)
            ja = len([k for k in results if not k.startswith("__")])
            print(f"Retomando: {ja} bloco(s) ja completo(s) em {OUT_JSON}")
        except Exception:
            results = {}

    total = len(args.models) * len(args.conditions)
    done = 0
    for model in args.models:
        for cond in args.conditions:
            done += 1
            key = f"{model}__{cond}"
            if key in results:
                print(f"\n[{done}/{total}] [PULANDO] {key} (ja completo)")
                continue

            vocab = top20 if cond == "top20" else None
            per_seed = []
            for seed in args.seeds:
                shots = pick_fewshot(pool, args.k, seed)
                print(f"\n[{done}/{total}] {model} | {cond} | seed {seed} "
                      f"| {len(shots)} shots")
                m = evaluate_run(test_data, shots, vocab, model, seed)
                print(f"      ABBREV F1={m['abbrev_f1']:.3f}  "
                      f"TYPO F1={m['typo_f1']:.3f}  Binary F1={m['binary_noise_f1']:.3f}  "
                      f"(parse_fail {m['parse_fail_rate']*100:.1f}%, "
                      f"oob/rec {m['oob_per_record']:.2f})")
                per_seed.append(m)

            agg = aggregate(per_seed)
            results[key] = {"model": model, "condition": cond, "k": args.k,
                            "per_seed": per_seed, "aggregate": agg}
            name = pretty_name(model, cond)
            print(f"  => {name}: ABBREV F1 {agg['abbrev_f1_mean']:.3f}"
                  f"+/-{agg['abbrev_f1_std']:.3f} | Binary F1 "
                  f"{agg['binary_noise_f1_mean']:.3f}+/-{agg['binary_noise_f1_std']:.3f}")

            # checkpoint imediato: o run e retomavel a partir daqui
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    latex_lines = []
    for key in sorted(k for k in results if not k.startswith("__")):
        v = results[key]
        latex_lines.append(latex_row_tabdict(pretty_name(v["model"], v["condition"]),
                                             v["aggregate"]))
    with open(OUT_LATEX, "w", encoding="utf-8") as f:
        f.write("% Tabela 3 (tab:dict) -- colunas: "
                "Method & ABBREV P & R & F1 & TYPO P & R & F1 & Binary F1\n")
        f.write("% medias sobre os seeds; bolde manualmente o vencedor por coluna.\n")
        f.write("\n".join(latex_lines) + "\n")

    print(f"\n[OK] metricas  -> {OUT_JSON}")
    print(f"[OK] LaTeX      -> {OUT_LATEX}")
    print("\nLinhas LaTeX (tab:dict):")
    print("\n".join(latex_lines))


if __name__ == "__main__":
    main()