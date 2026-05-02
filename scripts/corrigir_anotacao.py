import json
import shutil
from collections import Counter

TEST_FILE = "test_set_real.json"
BACKUP_FILE = "test_set_real_backup_pre_correcao.json"

CLEAN_PARA_ABBREV = {
    "beg",
    "bnf",
    "ndn",
    "nbz",
    "fant",
    "ivas",
    "im",
    "vo",
    "geca",
    "amox",
    "ef",
    "oto",
    "ml",
    "rash",
    "clav",
    "med",
    "hda",
    "pifr",
    "pa",
    "has",
    "seg",
    "cond",
    "mt",
    "sf",
    "itu",
    "les",
    "2seg",
    "3s",
    "24h",
    "2s",
    "id",
    "48h",
    "4a",
    "6h",
    "3x",
    "8h",
    "1x",
    "pre",
    "of",
    "38c",
    "8m",
    "3a",
    "pes",
    "am",
    "eup",
    "tec",
    "2x",
    "6a",
    "fr",
    "lab",
    "100mg",
    "alt",
    "39c",
    "min",
    "10h",
    "sn",
    "so",
    "dm",
    "2m",
    "hs",
    "ac",
    "rcr",
    "1a",
    "11m",
    "5x",
    "bcnf",
    "19h",
    "6m",
    "cef",
    "30h",
    "rha",
    "400mg",
    "16h",
    "qp",
    "7a",
    "72h",
    "12h",
    "fis",
    "3ep",
    "39oc",
    "bpnm",
    "an",
    "tax",
    "3cp",
    "db",
    "11kg",
    "ul",
    "10m",
    "200mg",
    "acv",
    "cn",
    "hlf",
    "sp",
    "sd",
    "su",
    "eos",
    "20mg",
    "ra",
    "rx",
    "1mg",
    "2ml",
    "1h",
    "30min",
    "mv",
    "24hs",
    "3ml",
    "4ml",
    "5ml",
    "gts",
    "hma",
    "4h",
    "23h",
    "nasosc",
    "pred",
    "fm",
    "adm",
    "11h",
    "10d",
    "7d",
    "3d",
    "13a",
    "7ml",
    "4x",
    "aaa",
    "2t",
    "ss",
    "ap",
    "5a",
    "9m",
    "2a",
    "diag",
    "12kg",
    "amig",
    "ok",
    "linf",
    "mod",
    "estr",
    "sat",
    "post",
    "5h",
    "40kg",
    "hv",
    "hj",
    "40min",
    "9h",
    "po",
    "abd",
    "6x",
    "aht",
    "uti",
    "neo",
    "ecg",
    "23kg",
}

CLEAN_PARA_TYPO = {
    "refre",
    "menigea",
    "versicular",
    "vomtios",
    "supercificial",
    "lonfonodomegalias",
    "giordanno",
    "giornado",
    "milagia",
    "perveo",
    "creptantes",
    "aoutras",
    "alimetares",
    "dispenia",
    "rinossoro",
    "cesareana",
    "prostada",
    "antipirexia",
    "predsin",
    "oncilon",
    "hidrat",
    "sulfa",
    "genito",
    "petequeias",
    "conica",
    "tondilas",
    "absominal",
    "nitofurantoina",
    "inciou",
    "malestar",
    "azopte",
    "defices",
    "condutoauditivo",
    "mucosanguiolenta",
    "rerefe",
    "holocrania",
    "acniterica",
    "aferidada",
    "nferiroes",
    "stres",
    "epsodio",
    "dispineias",
    "cetropofeno",
    "avmys",
    "frixotide",
    "epneico",
    "sinotclav",
    "taquineia",
    "rovisig",
    "murph",
    "ulmoes",
    "enterogermin",
    "orofaringia",
    "flagass",
    "euppneico",
    "defict",
    "inicada",
    "palpaccao",
    "eupneicoa",
    "orient",
    "biotindeido",
    "presenets",
    "mebros",
    "supercifial",
    "paretesia",
    "papulo",
    "mobilildade",
    "idnolor",
    "sgripais",
    "fascies",
    "verlamox",
    "cheixa",
    "edemasiada",
    "acompanahda",
    "mergalias",
    "pupulas",
    "corisa",
    "apresetou",
    "prensenca",
    "emgas",
    "dipiron",
    "ertitematosa",
    "nebules",
    "bactrin",
    "leuco",
    "uratos",
    "solivito",
    "sintomaicos",
    "hipermeiado",
    "prsente",
    "multigripe",
    "comrobidades",
    "hipremia",
    "acetil",
    "hipetrofiadas",
    "ansal",
    "hipertimpanismo",
    "precoupado",
    "acomapnhado",
    "inicado",
    "ambundante",
    "liquidopastosa",
    "garnganta",
    "aopos",
    "febricola",
    "musculatua",
    "hipermia",
    "superfcial",
    "cefaliea",
    "mesntrual",
    "taruma",
    "bracoes",
    "ictus",
    "itensificando",
    "alergiaca",
    "iniiciadas",
    "perioitaria",
    "berote",
    "incio",
    "retroorbitaria",
    "apresnetando",
    "spoiling",
    "lagrimejamento",
    "primiero",
    "primiera",
    "raiox",
    "putaceas",
    "prevo",
    "ceretide",
    "claritomincina",
    "epsodios",
    "fisioloigica",
    "amoxiclina",
    "clavulatano",
    "genelar",
    "2550gramos",
    "bronquilites",
    "sitnomas",
    "cincinatti",
    "hioras",
    "corisando",
    "abdomem",
    "jiujitsu",
    "forneco",
    "dfdores",
    "enxaquecoide",
    "exantemq",
}

def main():
    print("[1/4] Carregando test set...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"       {len(test_data)} anamneses")

    print("[2/4] Criando backup...")
    shutil.copy2(TEST_FILE, BACKUP_FILE)
    print(f"       Salvo em {BACKUP_FILE}")

    print("[3/4] Aplicando correções...")

    correcoes_abbrev = Counter()
    correcoes_typo = Counter()

    for entry in test_data:
        for i, (token, label) in enumerate(zip(entry["tokens"], entry["labels"])):
            token_lower = token.lower()

            if label == "CLEAN" and token_lower in CLEAN_PARA_ABBREV:
                entry["labels"][i] = "ABBREV"
                correcoes_abbrev[token_lower] += 1

            elif label == "CLEAN" and token_lower in CLEAN_PARA_TYPO:
                entry["labels"][i] = "TYPO"
                correcoes_typo[token_lower] += 1

    print("[4/4] Salvando...")
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    total_correcoes = sum(correcoes_abbrev.values()) + sum(correcoes_typo.values())

    print()
    print("=" * 60)
    print("RELATÓRIO DE CORREÇÃO")
    print("=" * 60)
    print(f"  Total de correções: {total_correcoes}")
    print()

    print("  CLEAN → ABBREV:")
    for token, count in correcoes_abbrev.most_common():
        print(f"    {token:25s} : {count}")
    print(f"  Subtotal: {sum(correcoes_abbrev.values())}")

    print()
    print("  CLEAN → TYPO:")
    for token, count in correcoes_typo.most_common():
        print(f"    {token:25s} : {count}")
    print(f"  Subtotal: {sum(correcoes_typo.values())}")

    total_clean = 0
    total_abbrev = 0
    total_typo = 0
    for entry in test_data:
        for label in entry["labels"]:
            if label == "CLEAN":
                total_clean += 1
            elif label == "ABBREV":
                total_abbrev += 1
            elif label == "TYPO":
                total_typo += 1

    total = total_clean + total_abbrev + total_typo
    print()
    print("  Nova distribuição:")
    print(f"    CLEAN:  {total_clean} ({100*total_clean/total:.1f}%)")
    print(f"    ABBREV: {total_abbrev} ({100*total_abbrev/total:.1f}%)")
    print(f"    TYPO:   {total_typo} ({100*total_typo/total:.1f}%)")

    print()
    print("  TOKENS AMBÍGUOS (verificar manualmente):")
    print("    'es' (299)     — palavra ou abreviação?")
    print("    'les' (19)     — lesões truncado?")
    print("    'id' (7)       — idade?")
    print("    'seg' (4)      — segmento ou seguimento?")
    print("    'alterac' (207)— truncação por encoding?")
    print("    'medicac' (50) — truncação por encoding?")
    print("    'congest' (21) — truncação?")
    print("    'evacuac' (15) — truncação?")
    print()
    print(f"  Arquivo salvo: {TEST_FILE}")
    print(f"  Backup em: {BACKUP_FILE}")
    print()
    print("  PRÓXIMO PASSO: verifique os tokens ambíguos")
    print("  listados acima e, se necessário, adicione-os")
    print("  aos sets de correção e rode novamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
