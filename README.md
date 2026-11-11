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
Requer Python 3.12 (algumas dependências ainda não têm versão pronta para 3.14).
```bash
pip install -r requirements.txt
python pipeline.py
python gerar_figuras.py
streamlit run app.py
```

O download do SIM (81 arquivos) e as consultas ao SICONFI (uma por município e ano) levam cerca de 45 minutos na primeira execução. Tudo fica em cache em `data/`, e uma execução interrompida continua de onde parou.

> A pasta `models/` já contém os artefatos gerados. O pipeline só é necessário para regenerar os dados a partir das fontes originais.

## Estrutura

```
├── app.py           # Dashboard Streamlit (6 páginas)
├── pipeline.py      # Orquestrador do pipeline completo
├── gerar_figuras.py # Figuras do artigo a partir de models/ (saída: figuras/)
├── src/
│   ├── etl.py       # Extract (FTP DATASUS + APIs) · Transform · Load
│   └── analise.py   # K-Means · OLS · Random Forest · XGBoost · Lift
├── models/          # Artefatos gerados (CSVs para o dashboard)
├── data/            # Dados brutos/processados (gerados pelo pipeline, não versionado)
└── requirements.txt
```

## Fontes de Dados

| Fonte | Conteúdo |
|---|---|
| SIM/DATASUS (FTP público) | Microdados de óbitos com CID-10 (2022–2024) |
| IBGE Sidra | População municipal total e por faixa etária (Censo 2022), PIB |
| SICONFI/Tesouro Nacional | Despesa municipal em saúde (RREO Anexo 02, completo ou simplificado) |
| IPEA Data / Atlas DH | IDH-M, Gini, renda, saneamento (Censo 2010) |

## Metodologia

- **ETL:** coleta automatizada via FTP do DATASUS e APIs REST (IBGE, SICONFI, Ipeadata OData), armazenamento em Parquet
- **Classificação:** Lista Brasileira de Causas Evitáveis — LBCE (Malta et al. 2007/2010), 195 prefixos CID-10, faixa 5–74 anos
- **Padronização etária:** método direto, população-padrão Brasil (Censo 2022), faixas quinquenais de 5 a 74 anos
- **Clusterização:** K-Means k=4 sobre 7 variáveis socioeconômicas (target excluído)
- **Regressão:** desfecho = taxa padronizada; OLS painel com erros cluster-robust por município + Random Forest + XGBoost
- **Mineração:** Análise de lift por CID-10 por cluster (causas com pelo menos 50 óbitos)

## Principais Achados

1. **Paradoxo etário:** o Norte tem a menor taxa bruta (149,4/100k), que sobe para 188,8/100k após a padronização direta; o Sul cai de 244,8 para 204,2/100k — as diferenças regionais praticamente desaparecem
2. **Gini (+137)** é o coeficiente OLS mais forte; o IDHM e a despesa em saúde não são significativos
3. **4 perfis municipais** (K-Means k=4): Norte/NE vulnerável, interior em desenvolvimento, Sul/SE médio e grandes centros
4. **Causas distintivas (lift):** eclâmpsia, desnutrição grave e esquistossomose no cluster mais vulnerável; dengue no Sul/SE médio; perfil próximo à média nacional nos grandes centros

## Referências

- MALTA, D.C. et al. *Lista de causas de mortes evitáveis por intervenções do SUS.* Epidemiol. Serv. Saúde, 16(4):233–244, 2007.
- MALTA, D.C. et al. *Atualização da lista de causas de mortes evitáveis.* Epidemiol. Serv. Saúde, 19(2):173–176, 2010.
- PNUD/IPEA/FJP. *Atlas do Desenvolvimento Humano no Brasil.* Brasília, 2013.
- DATASUS/MS. *Sistema de Informações sobre Mortalidade (SIM).* Brasília: MS, 2024.
