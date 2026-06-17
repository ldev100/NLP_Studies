# EPNI: Empirical Probabilistic Noise Injection

Código do experimento de injeção de ruído empírico em texto clínico sintético.
A ideia é gerar anamneses limpas com um LLM, inserir abreviações e erros de
digitação tirados de uma taxonomia real, e medir em anamneses reais se esses
dados treinam um detector de ruído melhor do que dado limpo, perturbação
aleatória, casamento por dicionário ou um LLM via prompt.

## Pipeline

```
[1] gerar anamneses limpas (Gemini 2.0 Flash)
[2] injetar ruído -> CLEAN + 3 RANDOM + 9 EPNI (13 configs)
[3] montar o test set real (500 anamneses + taxonomia)
[4] treinar e avaliar (BERTimbau + BioBERTpt, 3 seeds)  -> Tabela 2
[5] baseline de dicionário (DICT-Top20, DICT-Full)      -> Tabela 3
[6] baseline LLM few-shot via Ollama                     -> Tabela 3
[7] juntar as tabelas
```

## Antes de começar

Arquivos que vêm do estudo da taxonomia e da base clínica:

- `potential_abbreviations.json`: abreviações (superfície e contagem)
- `potential_typos.json`: erros de digitação
- `base_real.json`: anamneses reais

Instale o que os scripts usam:

```bash
pip install -r requirements.txt
```

São `transformers`, `torch`, `datasets`, `evaluate`, `scikit-learn`, `numpy`,
`pandas` e `matplotlib`. O passo 6 ainda precisa do cliente `ollama` e do Ollama
rodando local (detalhes no próprio passo).

Chave do gerador:

```bash
export GEMINI_API_KEY="sua-chave"
```

---

## 1. Gerar anamneses limpas

```bash
python 01_gerar_anamneses.py
```

Gera 1.196 anamneses de pronto-socorro em português, em prosa, com tudo escrito
por extenso e sem abreviação. A diversidade vem de variar queixa, perfil do
paciente, comorbidades, tamanho e nível de detalhe entre as gerações.

Saída: `anamneses_sinteticas_limpas.json`.

Usei o Gemini 2.0 Flash pela qualidade em português e pelo custo baixo, mas
qualquer LLM que escreva prosa clínica fluente na língua serve no lugar. O
script grava por lote, então dá para parar e retomar.

---

## 2. Injetar ruído (~5 min)

```bash
python 02_injetar_ruido.py
```

Pega as anamneses limpas e cria as 13 versões de treino. O grid cruza vocabulário
com taxa de injeção:

- vocabulário: Top-10, Top-15, Top-20 (rank de frequência na taxonomia)
- taxa: 10%, 25%, 50%

Isso dá 9 combinações EPNI. Somando o CLEAN (sem ruído) e os 3 RANDOM
(perturbação de caractere nas mesmas taxas, sem usar a taxonomia), fecham 13.
Cada forma limpa vira sua versão ruidosa pela tabela de mapeamento, por exemplo
"bom estado geral" para "BEG".

Precisa de `anamneses_sinteticas_limpas.json` e das duas taxonomias.

Sai tudo em `datasets_experimento_ruido_1196/`:

- `clean_synthetic_anamneses_gold_standart_tokens.json`
- `dataset_random_rate10.json`, `_rate25`, `_rate50`
- `dataset_top10_rate10.json` até `dataset_top20_rate50.json` (9 arquivos)

Cada um traz `tokens` e `labels` (CLEAN/ABBREV/TYPO) por token.

---

## 3. Montar o test set real (~1 min)

```bash
python 04_preparar_test_real.py
```

Seleciona 500 anamneses reais anotadas token a token por um pesquisador com
conhecimento clínico. Como conferência, cada token é checado contra a taxonomia
(comparando a superfície depois de normalizar a caixa). A concordância fica em
96,0% sobre os 37.394 tokens e em 83,5% se olhar só os 4.899 tokens de ruído
(84,3% nas abreviações, 75,7% nos typos). A distribuição final é 86,9% CLEAN,
11,8% ABBREV e 1,3% TYPO.

Entra `base_real.json` mais as taxonomias, sai `test_set_real.json` com `tokens`
e `labels`.

A `carregar_base_real()` tenta adivinhar o formato da base, mas é o ponto mais
provável de precisar de ajuste conforme como os dados reais estão guardados.

---

## 4. Treinar e avaliar (Tabela 2)

```bash
python 03_treinar_avaliar.py
```

Faz o fine-tune do BERTimbau (modelo geral) e do BioBERTpt (adaptado à clínica)
em cada uma das 13 configurações, com 3 seeds, e avalia no test set real.

Os hiperparâmetros: learning rate 2e-5, batch 4, comprimento máximo 300, até 10
épocas com early stopping de paciência 3, e entropia cruzada ponderada com pesos
0,2 para CLEAN, 5,0 para TYPO e 15,0 para ABBREV. Tudo que é escolha de modelo
sai de um split do próprio sintético; as 500 reais só aparecem na avaliação
final. Seeds 42, 123 e 456.

São 13 × 2 × 3 = 78 runs, algo entre 10 e 15 minutos cada no M4 com MPS, então
conte com algo na faixa de 13 a 20 horas.

Resultados em `resultados_experimento_1196/`: `metricas_completas.json`, a tabela
resumo em `tabela_principal.csv` e `tabela_latex.txt`, mais alguns gráficos
auxiliares (as figuras de resultado estão comentadas no `.tex`). Essa é a
Tabela 2 do paper, em nível de sub-token. A melhor configuração chega a 0,819 de
F1 em abreviação (BERTimbau, TOP20-R50).

---

## 5. Baseline de dicionário (Tabela 3, ~1 min)

```bash
python dict_baseline.py
```

Dois taggers de casamento exato rodando no mesmo test set, com a mesma métrica em
nível de palavra dos modelos treinados. O DICT-Top20 usa só o vocabulário Top-20,
que é o mesmo que os modelos TOP20 viram, e serve de comparação direta com o
melhor EPNI. O DICT-Full usa a taxonomia inteira, funcionando como um teto quase
oráculo, já que carrega o mesmo prior que validou as anotações.

Entra o test set, os arquivos `dataset_top20_rate*.json` (de onde sai o Top-20) e
as taxonomias. Sai `dict_baseline_results.json` e as linhas em
`dict_baseline_latex.txt`. No paper o DICT-Top20 dá 0,678 de F1 em abreviação e o
DICT-Full 0,698, com o DICT-Full liderando em typo (0,623). A avaliação é em
nível de palavra por padrão (`DO_SUBWORD = False`).

---

## 6. Baseline LLM few-shot via Ollama (Tabela 3)

Compara o detector treinado com um LLM de instrução genérico chamado por prompt,
para ver se compensa treinar um classificador dedicado. Roda local para não
mandar anamnese real para nenhum serviço de fora.

Preparar o Ollama (uma vez):

```bash
ollama serve                 # ou abra o app; o servidor sobe na porta 11434

ollama pull qwen2.5:7b       # ~4.7 GB
ollama pull qwen2.5:14b      # ~9 GB

ollama --version             # precisa de >= 0.5 por causa do JSON estruturado; usei a 0.30.8
ollama list                  # anote os IDs: 7b 845dbda0ea48, 14b 7cdf5a0187d5

pip install ollama
```

Rodar:

```bash
# checagem rápida antes do sweep
python llm_baseline.py --limit 20 --models qwen2.5:7b --seeds 42

# sweep completo; o caffeinate impede o Mac de dormir no meio
caffeinate -i python llm_baseline.py
```

Cada modelo rotula os tokens de cada anamnese com 4 exemplos few-shot tirados do
split rate-10 (densidade parecida com a real), em duas condições: uma fria, sem
vocabulário no prompt, e outra com o Top-20 listado, espelhando o que o DICT-Full
tem de prior. A decodificação é gulosa (temperatura zero), então a variação entre
os 3 seeds vem só de quais exemplos entram no prompt. A saída é presa a um schema
JSON que lista apenas os tokens ruidosos por índice, com um resgate por regex caso
o JSON venha cortado. O run grava a cada bloco de modelo e condição, dá para
interromper e continuar.

Entra o test set, o `dataset_top20_rate10.json` (fonte dos exemplos) e as
taxonomias. Sai `llm_baseline_results.json`, com as métricas por modelo, condição
e seed e alguns diagnósticos (`parse_fail_rate`, `oob_per_record`,
`pred_noise_per_record`), e as 4 linhas em `llm_baseline_latex.txt`.

São 6000 chamadas no total (2 modelos × 2 condições × 3 seeds × 500). O 14B é o
gargalo, então reserve algumas horas e feche os outros apps, porque os 9 GB dele
apertam os 16 GB. No paper deu 0,220 e 0,213 de F1 em abreviação para o 7B (frio
e com Top-20) e 0,352 e 0,343 para o 14B, todos bem abaixo do dicionário e do
EPNI.

---

## 7. Juntar as tabelas

A Tabela 2 sai de `resultados_experimento_1196/tabela_latex.txt`. A Tabela 3 é a
junção de `dict_baseline_latex.txt` e `llm_baseline_latex.txt` com a linha do
melhor EPNI, tudo em nível de palavra. A Tabela 4 (visto contra não visto do
TOP20-R50) vem de um re-treino dedicado no passo 4.

Para conferir os números na hora de montar: o melhor F1 de abreviação é 0,819 em
sub-token e 0,763 em palavra, o teto do casamento exato é 0,678, o DICT-Full dá
0,698 e o melhor LLM fica em 0,352. A detecção de typo segue baixa em todos os
modelos treinados (F1 até 0,178), enquanto o DICT-Full chega a 0,623.

---

## Diretórios

```
experiment/
├── requirements.txt
├── 01_gerar_anamneses.py
├── 02_injetar_ruido.py
├── 03_treinar_avaliar.py
├── 04_preparar_test_real.py
├── dict_baseline.py
├── llm_baseline.py
├── potential_abbreviations.json
├── potential_typos.json
├── base_real.json
├── anamneses_sinteticas_limpas.json
├── test_set_real.json
├── datasets_experimento_ruido_1196/
│   ├── clean_synthetic_anamneses_gold_standart_tokens.json
│   ├── dataset_random_rate10.json | rate25 | rate50
│   └── dataset_top10_rate10.json ... dataset_top20_rate50.json
├── resultados_experimento_1196/
│   ├── metricas_completas.json
│   ├── tabela_principal.csv
│   └── tabela_latex.txt
├── dict_baseline_results.json
├── dict_baseline_latex.txt
├── llm_baseline_results.json
└── llm_baseline_latex.txt
```

---

## Problemas comuns

**MPS não disponível.** Precisa de PyTorch 2.0 ou mais novo. No M4,
`pip install torch torchvision torchaudio`.

**Estouro de memória no treino.** Baixe o `BATCH_SIZE` de 4 para 2 no
`03_treinar_avaliar.py`.

**Formato não reconhecido no passo 3.** Ajuste a `carregar_base_real()` para o
jeito que a base real está (CSV, JSON com outros campos, etc.).

**`dataset_top20_rate*.json` não encontrado no passo 5.** Sem os datasets de
treino o `dict_baseline.py` monta o Top-20 a partir da taxonomia, mas para a
comparação certa rode o passo 2 antes.

**Ollama não aceita o `format`.** O JSON estruturado precisa da versão 0.5 ou mais
nova (usei a 0.30.8). Atualize com `brew upgrade ollama`. O script cai para
`format="json"` se o schema falhar, mas o schema é mais confiável.

**`parse_fail` alto na checagem do passo 6.** Olhe os `_raw_failures` dentro do
`llm_baseline_results.json`. Com o `num_predict` folgado e o resgate por regex
já no script isso fica perto de zero (no sweep deu 0%).

**14B estourando os 16 GB.** Feche o resto, ou rode primeiro só o 7B
(`--models qwen2.5:7b`) e depois o 14B (`--models qwen2.5:14b`); como o run
continua de onde parou, não se perde o que já rodou.
