# Mortalidade por Causas Evitáveis no Brasil

**Trabalho de Conclusão de Curso — Ciência da Computação — IESB 2026/01**

**Alunos:** João Marcos Santos e Carvalho · Maria Clara Fontenele Silva  
**Orientador:** Prof. Cristiano Lehrer

---

## Sobre o Projeto

Plataforma de análise de dados para investigar a relação entre o Índice de Desenvolvimento Humano (IDH) e as taxas de mortalidade por causas evitáveis nos municípios brasileiros (2022–2024).

**Pergunta de pesquisa:** Em que medida indicadores socioeconômicos municipais (IDH, Gini, renda, saneamento) explicam as taxas de mortalidade por causas evitáveis no Brasil?

## Como Executar

### Opção 1 — Windows (recomendado)
Dê duplo clique em `iniciar.bat`.

### Opção 2 — Terminal
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Opção 3 — Rodar o pipeline completo (ETL do zero)
```bash
pip install -r requirements.txt
python pipeline.py
streamlit run app.py
```

> A pasta `models/` já contém os artefatos gerados. O pipeline só é necessário para regenerar os dados a partir das fontes originais.

## Estrutura

```
├── app.py           # Dashboard Streamlit (6 páginas)
├── pipeline.py      # Orquestrador do pipeline completo
├── src/
│   ├── etl.py       # Extract (PySUS + APIs) · Transform · Load
│   └── analise.py   # K-Means · OLS · Random Forest · XGBoost · Lift
├── models/          # Artefatos gerados (CSVs para o dashboard)
├── data/            # Dados brutos/processados (gerados pelo pipeline, não versionado)
└── requirements.txt
```

## Fontes de Dados

| Fonte | Conteúdo |
|---|---|
| SIM/DATASUS via PySUS | Microdados de óbitos com CID-10 (2022–2024) |
| IBGE Sidra | População municipal (Censo 2022), PIB |
| SICONFI/Tesouro Nacional | Despesa municipal em saúde |
| IPEA Data / Atlas DH | IDH-M, Gini, renda, saneamento (Censo 2010) |

## Metodologia

- **ETL:** coleta automatizada via PySUS e APIs REST, armazenamento em Parquet
- **Classificação:** Lista Brasileira de Causas Evitáveis — LBCE (Malta et al. 2007/2010), 344 prefixos CID-10, faixa 5–74 anos
- **Clusterização:** K-Means k=4 sobre 7 variáveis socioeconômicas (target excluído)
- **Regressão:** OLS painel com erros cluster-robust por município + Random Forest + XGBoost
- **Mineração:** Análise de lift por CID-10 por cluster

## Principais Achados

1. **Paradoxo corrigido:** o Norte tem a menor taxa bruta (158,9/100k) mas a maior padronizada (227,7/100k) — viés da estrutura etária corrigido por padronização direta
2. **Gini (+151)** é o coeficiente OLS mais forte — desigualdade supera pobreza absoluta
3. **4 perfis municipais** confirmados por ANOVA (p < 10⁻¹⁷): Norte/NE vulnerável, interior em transição, cidades médias e grandes centros
4. **Causas distintivas:** Chagas/esquistossomose nos municípios pobres; dengue nas cidades médias; DCV nos grandes centros

## Referências

- MALTA, D.C. et al. *Lista de causas de mortes evitáveis por intervenções do SUS.* Epidemiol. Serv. Saúde, 16(4):233–244, 2007.
- MALTA, D.C. et al. *Atualização da lista de causas de mortes evitáveis.* Epidemiol. Serv. Saúde, 19(2):173–176, 2010.
- PNUD/IPEA/FJP. *Atlas do Desenvolvimento Humano no Brasil.* Brasília, 2013.
- DATASUS/MS. *Sistema de Informações sobre Mortalidade (SIM).* Brasília: MS, 2024.
