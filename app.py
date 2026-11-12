"""
Dashboard TCC — Mortalidade por Causas Evitáveis no Brasil
IESB 2026/01 | Ciência da Computação

Alunos    : João Marcos Santos e Carvalho | Maria Clara Fontenele Silva
Orientador: Prof. Cristiano Lehrer
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Configuração da Página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="TCC | Mortalidade Evitável — IESB 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGO_PATH  = ROOT / "iesblogo.png"

# ─── CSS Acadêmico ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar escura / institucional */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #12294a 55%, #1a3a5c 100%);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small {
    color: #c8dff7 !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2e6da4 !important;
}
/* Cards métrica */
div[data-testid="metric-container"] {
    background: #f0f6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px !important;
}
/* Cabeçalho TCC */
.tcc-hero {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #2563eb 100%);
    color: white;
    padding: 28px 36px;
    border-radius: 12px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(10,22,40,0.25);
}
.tcc-hero h1 { color: #ffffff !important; font-size: 22px; margin: 0 0 6px 0; line-height: 1.3; }
.tcc-hero .subtitle { color: #93c5fd; font-size: 14px; margin: 2px 0; }
.tcc-hero .authors  { color: #e0f2fe; font-size: 13px; margin-top: 10px; }
/* Boxes */
.box-pergunta {
    background: #eff6ff; border-left: 5px solid #2563eb;
    padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 10px 0;
}
.box-hipotese {
    background: #f0fdf4; border-left: 5px solid #16a34a;
    padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 10px 0;
}
.box-achado {
    background: #fefce8; border-left: 5px solid #ca8a04;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0;
}
.box-limitacao {
    background: #fff7ed; border-left: 5px solid #ea580c;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0;
}
.box-futuro {
    background: #f5f3ff; border-left: 5px solid #7c3aed;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0;
}
/* Rodapé */
.footer-bar {
    background: #f1f5f9; border-top: 2px solid #cbd5e1;
    padding: 10px 20px; border-radius: 8px; margin-top: 30px;
    color: #475569; font-size: 12px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─── Constantes ───────────────────────────────────────────────────────────────
CLUSTER_LABELS = {
    0: "Cluster 0 — Norte/NE Vulnerável",
    1: "Cluster 1 — Interior em Desenvolvimento",
    2: "Cluster 2 — Sul/SE Médio",
    3: "Cluster 3 — Grandes Centros",
}
CLUSTER_CORES = {0: "#ef4444", 1: "#f97316", 2: "#3b82f6", 3: "#22c55e"}
REGIOES = {"N": "Norte", "NE": "Nordeste", "CO": "Centro-Oeste", "SE": "Sudeste", "S": "Sul"}
REGIAO_COORDS = {
    "N":  {"lat": -4.5,  "lon": -62.0},
    "NE": {"lat": -8.5,  "lon": -39.5},
    "CO": {"lat": -15.5, "lon": -52.0},
    "SE": {"lat": -20.0, "lon": -44.0},
    "S":  {"lat": -28.0, "lon": -52.0},
}
FEATURE_NOMES = {
    "const":                "Constante",
    "LOG_PIB_PER_CAPITA":   "PIB per capita (log R$)",
    "LOG_DESPESA_SAUDE_PC": "Despesa saúde p.c. (log R$)",
    "LOG_POPULACAO":        "Tamanho populacional (log)",
    "IDHM":                 "IDH Municipal",
    "GINI":                 "Índice de Gini (desigualdade)",
    "RDPC":                 "Renda domiciliar per capita",
    "PMPOB":                "% em extrema pobreza",
    "ESPVIDA":              "Esperança de vida (anos)",
    "T_ANALF15M":           "Analfabetismo 15+ (%)",
    "T_AGUA":               "Acesso à água encanada (%)",
    "REG_N":                "Região Norte",
    "REG_NE":               "Região Nordeste",
    "REG_S":                "Região Sul",
    "REG_SE":               "Região Sudeste",
    "ANO_2023":             "Ano 2023",
    "ANO_2024":             "Ano 2024",
}
DESCRICOES_CID = {
    "I64":"AVC (tipo não especificado)","I63":"Infarto cerebral (AVC isquêmico)",
    "I61":"Hemorragia intracerebral","I67":"Outras doenças cerebrovasculares",
    "I21":"Infarto agudo do miocárdio","I25":"Doença isquêmica crônica do coração",
    "I10":"Hipertensão essencial","I50":"Insuficiência cardíaca",
    "B57":"Doença de Chagas","B65":"Esquistossomose",
    "A09":"Diarreia e gastroenterite","A90":"Dengue",
    "A39":"Meningite meningocócica","B24":"HIV/AIDS",
    "A15":"Tuberculose respiratória","B18":"Hepatite viral crônica",
    "A41":"Septicemia","E11":"Diabetes mellitus tipo 2",
    "E14":"Diabetes mellitus NE","J18":"Pneumonia",
    "J43":"Enfisema pulmonar","J44":"DPOC",
    "J45":"Asma","C50":"Câncer de mama","C53":"Câncer do colo do útero",
    "C34":"Câncer de pulmão","C16":"Câncer de estômago","C18":"Câncer de cólon",
    "C61":"Câncer de próstata","K70":"Doença alcoólica do fígado",
    "K74":"Cirrose hepática","N18":"Insuficiência renal crônica",
    "O15":"Eclâmpsia","O14":"Pré-eclâmpsia",
}
NARRATIVAS_CLUSTER = {
    0: ("Municípios do Norte e Nordeste com os piores indicadores socioeconômicos. "
        "Eclâmpsia (O15), desnutrição grave (E43) e esquistossomose (B65) são as causas mais "
        "distintivas — mortes ligadas ao cuidado na gestação, à nutrição e ao saneamento."),
    1: ("Interior em desenvolvimento, sobretudo no Nordeste e no Norte. Esquistossomose (B65) e "
        "eclâmpsia (O15) marcam o perfil — parasitoses ligadas ao saneamento e complicações na gravidez."),
    2: ("Municípios de porte médio no Sul, Sudeste e Centro-Oeste. Dengue (A90) e enfisema (J43) "
        "são os principais diferenciais, com lifts moderados."),
    3: ("Grandes centros com os melhores indicadores. Nenhuma causa se destaca muito da média "
        "nacional: o perfil de causas é parecido com o do país, com AVC isquêmico (I63) e "
        "infecção meningocócica (A39) entre as mais frequentes em relação à média."),
}


# ─── Loaders ──────────────────────────────────────────────────────────────────
def _modelos_ok() -> list[str]:
    obrig = ["perfil_clusters.csv","metricas_modelos.csv","coeficientes_ols.csv",
             "importancia_features.csv","top_causas_cluster.csv","lbce_por_cluster.csv",
             "lift_causas_cluster.csv","taxa_regiao_comparacao.csv",
             "tendencia_temporal.csv","elbow_silhouette.csv"]
    return [f for f in obrig if not (MODELS_DIR / f).exists()]


@st.cache_data
def _csv(nome: str) -> pd.DataFrame:
    p = MODELS_DIR / nome
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_perfil() -> pd.DataFrame:
    df = _csv("perfil_clusters.csv")
    if df.empty:
        return df
    df["label"] = df["CLUSTER"].map(CLUSTER_LABELS)
    df["cor"]   = df["CLUSTER"].map(CLUSTER_CORES)
    for col in ("PIB_pc", "desp_saude_pc"):
        if col in df.columns:
            df[f"{col}_real"] = np.expm1(df[col].clip(lower=0))
    return df


def _br(x: float, casas: int = 1) -> str:
    """Número no formato brasileiro (vírgula decimal)."""
    return f"{x:.{casas}f}".replace(".", ",")


def _p_cientifico(p: float) -> str:
    """p-valor em notação científica legível, ex.: 2,10×10⁻¹⁶."""
    if p >= 0.001:
        return _br(p, 3)
    m, e = f"{p:.2e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m.replace('.', ',')}×10{sup}"


def _rotular_cid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["descricao"] = df["CID_CAT3"].map(DESCRICOES_CID).fillna(df["CID_CAT3"])
    return df


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=110)

    st.markdown("## TCC — IESB 2026/01")
    st.markdown(
        "**Ciência da Computação**\n\n"
        "**Alunos:**  \nJoão Marcos S. Carvalho  \nMaria Clara F. Silva\n\n"
        "**Orientador:**  \nProf. Cristiano Lehrer"
    )
    st.divider()

    pagina = st.radio(
        "Navegação",
        options=[
            "🎓 Apresentação",
            "📊 Panorama Nacional",
            "🗺️ Perfis Municipais",
            "🔬 Análise de Causas",
            "📈 Modelagem Estatística",
            "📋 Conclusões",
        ],
    )

    st.divider()
    st.caption(
        "**Fontes de dados:**  \n"
        "SIM/DATASUS · IBGE Sidra  \n"
        "SICONFI/STN · IPEA Data  \n\n"
        "**Período:** 2022–2024  \n"
        "**Cobertura:** 5.571 municípios"
    )

# ─── Verifica se os modelos existem ────────────────────────────────────────────
faltando = _modelos_ok()
if faltando and pagina != "🎓 Apresentação":
    st.error(
        "**Dados não encontrados.** Execute o pipeline para gerar os modelos:\n"
        "```\npython pipeline.py\n```\n"
        "Ou copie a pasta `models/` do projeto de referência para `TCC/models/`."
    )
    st.info("A página **Apresentação** está disponível sem dados.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Apresentação
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🎓 Apresentação":

    st.markdown("""
<div class="tcc-hero">
  <h1>Plataforma de Análise de Dados para Identificar Padrões de<br>
      Mortalidade por Causas Evitáveis no Brasil</h1>
  <p class="subtitle">Trabalho de Conclusão de Curso · Ciência da Computação · IESB · 2026/01</p>
  <p class="authors">
    <strong>Alunos:</strong> João Marcos Santos e Carvalho &nbsp;|&nbsp; Maria Clara Fontenele Silva<br>
    <strong>Orientador:</strong> Prof. Cristiano Lehrer
  </p>
</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Pergunta de Pesquisa")
        st.markdown("""
<div class="box-pergunta">
Em que medida indicadores socioeconômicos municipais — IDH, Gini, renda,
saneamento e gasto em saúde — explicam as taxas de mortalidade por causas
evitáveis nos municípios brasileiros entre 2022 e 2024?
</div>
""", unsafe_allow_html=True)

        st.markdown("### Hipótese")
        st.markdown("""
<div class="box-hipotese">
Municípios com piores indicadores socioeconômicos (menor IDH-M, maior Gini,
menor acesso a saneamento) apresentam taxas de mortalidade evitável
significativamente maiores, mesmo após controle pela estrutura etária da
população — evidenciando iniquidades em saúde reduzíveis por políticas
públicas de atenção básica.
</div>
""", unsafe_allow_html=True)

        st.markdown("### Objetivo Geral")
        st.info(
            "Desenvolver uma plataforma computacional para ingestão, processamento e análise "
            "da correlação entre o IDH e a mortalidade evitável no Brasil, integrando dados "
            "abertos do SIM/DATASUS, IBGE, SICONFI e IPEA."
        )

    with col_b:
        st.markdown("### Pipeline Técnico")
        st.markdown("""
| Etapa | Tecnologia | Saída |
|---|---|---|
| **Extract** | FTP do DATASUS + APIs REST | Microdados brutos (.parquet) |
| **Transform** | Pandas + LBCE | feature_matrix.parquet |
| **Load** | Apache Parquet | Formato colunar otimizado |
| **Clusterização** | K-Means (k=4) | perfil_clusters.csv |
| **Regressão** | OLS + RF + XGBoost | coeficientes_ols.csv |
| **Mineração** | Lift por CID-10 | lift_causas_cluster.csv |
| **Dashboard** | Streamlit + Plotly | Esta plataforma |
""")

        st.markdown("### Objetivos Específicos")
        st.markdown("""
1. Estruturar um **pipeline ETL** automatizado via FTP do DATASUS e APIs públicas
2. Armazenar os dados processados em **formato Parquet** (colunar, comprimido)
3. Aplicar **K-Means** para agrupar municípios com perfis socioeconômicos similares
4. Quantificar a associação via **regressão OLS** com erros cluster-robust
5. Identificar **causas CID-10 distintivas** por grupo via análise de lift
6. Disponibilizar os resultados nesta **plataforma interativa**
""")

    st.divider()

    st.markdown("### Fontes de Dados")
    st.markdown("""
| Fonte | Conteúdo | Acesso |
|---|---|---|
| **SIM/DATASUS** | Microdados de óbitos com CID-10 (2022–2024) | FTP público do DATASUS |
| **IBGE Sidra** | População municipal (Censo 2022), PIB municipal | API REST |
| **SICONFI/STN** | Despesa municipal em saúde (Função 10) | API REST |
| **IPEA/Atlas DH** | IDH-M, Gini, renda, saneamento, esperança de vida | ipeadatapy |
""")

    st.divider()

    st.markdown("### Lista Brasileira de Causas Evitáveis (LBCE)")
    st.markdown(
        "Classificação de Malta et al. (2007/2010) — 195 categorias CID-10 (3 caracteres), faixa etária **5–74 anos**. "
        "Causas externas (acidentes, homicídios) deliberadamente excluídas: determinantes "
        "são distintos das variáveis socioeconômicas do modelo."
    )
    lb = _csv("lbce_por_cluster.csv")

    def _faixa(col: str, casas: int) -> str:
        if lb.empty or col not in lb.columns:
            return "—"
        return f"{_br(lb[col].min(), casas)}–{_br(lb[col].max(), casas)}%"

    st.caption("Participação de cada grupo nas mortes evitáveis (faixa entre os 4 clusters, 2022–2024).")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.metric("Imunopreveníveis", _faixa("imunoprevencao", 2), "vacinação")
    col_l2.metric("D. Infecciosas", _faixa("doencas_infecciosas", 1), "diagnóstico/tratamento")
    col_l3.metric("D. Não Transmissíveis", _faixa("doencas_nao_transmissiveis", 1), "prevenção/atenção básica")
    col_l4.metric("Morte Materna", _faixa("morte_materna", 2), "gestação/parto")

    st.markdown("""
<div class="footer-bar">
TCC · IESB 2026/01 · Ciência da Computação ·
João Marcos Santos e Carvalho (2312130063) · Maria Clara Fontenele Silva (2312130230) ·
Orientador: Prof. Cristiano Lehrer
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — Panorama Nacional
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Panorama Nacional":

    st.markdown("""
<div class="tcc-hero">
  <h1>📊 Panorama Nacional da Mortalidade Evitável</h1>
  <p class="subtitle">Visão geral · Paradoxo da taxa bruta × padronizada · Evolução temporal</p>
</div>
""", unsafe_allow_html=True)

    if faltando:
        st.stop()

    perfil   = load_perfil()
    tendencia = _csv("tendencia_temporal.csv")
    regiao   = _csv("taxa_regiao_comparacao.csv")

    if not regiao.empty and "REGIAO" in regiao.columns:
        regiao["regiao_nome"] = regiao["REGIAO"].map(REGIOES)

    # KPIs
    total_mun    = int(perfil["n_municipios"].sum()) if not perfil.empty and "n_municipios" in perfil.columns else 0
    total_obitos = int(tendencia["obitos_total"].sum()) if not tendencia.empty and "obitos_total" in tendencia.columns else 0

    taxa_bruta_p = 0.0
    taxa_pad_p   = 0.0
    if not perfil.empty and "%_pop_brasil" in perfil.columns:
        if "taxa_bruta" in perfil.columns:
            taxa_bruta_p = float((perfil["taxa_bruta"] * perfil["%_pop_brasil"]).sum() / 100)
        if "taxa_padronizada" in perfil.columns:
            taxa_pad_p = float((perfil["taxa_padronizada"] * perfil["%_pop_brasil"]).sum() / 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municípios analisados", f"{total_mun:,}".replace(",","."))
    c2.metric("Óbitos evitáveis (3 anos)", f"{total_obitos:,}".replace(",","."))
    c3.metric("Taxa bruta média", f"{taxa_bruta_p:.1f} /100k", help="Sem correção etária")
    c4.metric("Taxa padronizada média", f"{taxa_pad_p:.1f} /100k", help="Corrigida pelo Censo 2022")

    st.divider()

    # ── Mapa regional ─────────────────────────────────────────────────────────
    st.subheader("Distribuição regional da mortalidade evitável padronizada")
    if not regiao.empty and "taxa_padronizada" in regiao.columns:
        ord_reg = regiao.set_index("REGIAO").reindex(["N","NE","CO","SE","S"]).reset_index()
        ord_reg["lat"]  = ord_reg["REGIAO"].map(lambda r: REGIAO_COORDS.get(r,{}).get("lat",0))
        ord_reg["lon"]  = ord_reg["REGIAO"].map(lambda r: REGIAO_COORDS.get(r,{}).get("lon",0))
        ord_reg["nome"] = ord_reg["REGIAO"].map(REGIOES)

        fig_map = px.scatter_geo(
            ord_reg, lat="lat", lon="lon",
            size="n_municipios" if "n_municipios" in ord_reg.columns else None,
            color="taxa_padronizada",
            color_continuous_scale="RdYlGn_r",
            hover_name="nome",
            hover_data={"taxa_padronizada":":.1f","taxa_bruta":":.1f",
                        "lat":False,"lon":False},
            text="nome",
            labels={"taxa_padronizada":"Taxa pad. /100k"},
            size_max=60,
        )
        fig_map.update_traces(textposition="top center", textfont_size=12)
        fig_map.update_geos(
            scope="south america", showcountries=True,
            countrycolor="lightgray", showland=True, landcolor="#f8fafc",
            showcoastlines=True, coastlinecolor="gray", fitbounds="locations",
        )
        fig_map.update_layout(height=420, coloraxis_colorbar_title="Taxa pad.<br>/100k")
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "Tamanho das bolhas proporcional ao número de municípios. "
            "Vermelho = maior mortalidade evitável padronizada."
        )

    st.divider()

    # ── Paradoxo bruta vs padronizada ─────────────────────────────────────────
    st.subheader("O Paradoxo: Taxa Bruta vs. Padronizada por Região")
    st.markdown(
        "Municípios desenvolvidos têm populações mais envelhecidas. Como doenças crônicas "
        "matam próximo aos 70 anos, o Sul aparecia com maior mortalidade pela taxa bruta — "
        "o contrário do esperado. A padronização etária direta corrige esse viés."
    )

    if not regiao.empty and "taxa_bruta" in regiao.columns and "taxa_padronizada" in regiao.columns:
        ord_reg2 = regiao.set_index("REGIAO").reindex(["N","NE","CO","SE","S"]).reset_index()
        ord_reg2["regiao_nome"] = ord_reg2["REGIAO"].map(REGIOES)

        fig_par = go.Figure()
        fig_par.add_trace(go.Bar(
            name="Taxa bruta /100k", x=ord_reg2["regiao_nome"], y=ord_reg2["taxa_bruta"],
            marker_color="#94a3b8", text=ord_reg2["taxa_bruta"].round(1), textposition="outside",
        ))
        fig_par.add_trace(go.Bar(
            name="Taxa padronizada /100k", x=ord_reg2["regiao_nome"], y=ord_reg2["taxa_padronizada"],
            marker_color="#2563eb", text=ord_reg2["taxa_padronizada"].round(1), textposition="outside",
        ))
        fig_par.update_layout(
            barmode="group", height=360,
            yaxis_title="Óbitos evitáveis por 100k hab.",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_par, use_container_width=True)
        rg = regiao.set_index("REGIAO")
        menor_b, maior_b = rg["taxa_bruta"].idxmin(), rg["taxa_bruta"].idxmax()
        d_menor = rg.loc[menor_b, "taxa_padronizada"] - rg.loc[menor_b, "taxa_bruta"]
        d_maior = rg.loc[maior_b, "taxa_padronizada"] - rg.loc[maior_b, "taxa_bruta"]
        st.info(
            f"**{REGIOES[menor_b]}:** taxa bruta {_br(rg.loc[menor_b, 'taxa_bruta'])} (a menor do país) → "
            f"padronizada {_br(rg.loc[menor_b, 'taxa_padronizada'])} ({'+' if d_menor >= 0 else '−'}{_br(abs(d_menor))}).  \n"
            f"**{REGIOES[maior_b]}:** taxa bruta {_br(rg.loc[maior_b, 'taxa_bruta'])} (a maior) → "
            f"padronizada {_br(rg.loc[maior_b, 'taxa_padronizada'])} ({'+' if d_maior >= 0 else '−'}{_br(abs(d_maior))}).  \n"
            f"Após a padronização, as regiões ficam próximas: de {_br(rg['taxa_padronizada'].min())} a "
            f"{_br(rg['taxa_padronizada'].max())} óbitos por 100 mil habitantes."
        )

    st.divider()

    col_t, col_c = st.columns(2)

    with col_t:
        st.subheader("Evolução temporal (2022–2024)")
        if not tendencia.empty and "taxa_media" in tendencia.columns:
            fig_tend = go.Figure()
            fig_tend.add_trace(go.Scatter(
                x=tendencia["ANO"].astype(str), y=tendencia["taxa_media"],
                mode="lines+markers+text",
                marker=dict(size=14, color="#2563eb"),
                line=dict(width=3, color="#2563eb"),
                text=tendencia["taxa_media"].round(1),
                textposition="top center",
                fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            ))
            fig_tend.update_layout(height=290, yaxis_title="Taxa padronizada /100k")
            st.plotly_chart(fig_tend, use_container_width=True)

    with col_c:
        st.subheader("Taxa padronizada por cluster")
        if not perfil.empty and "taxa_padronizada" in perfil.columns:
            fig_cl = px.bar(
                perfil, x="label", y="taxa_padronizada", color="label",
                color_discrete_map={v: CLUSTER_CORES[k] for k, v in CLUSTER_LABELS.items()},
                labels={"taxa_padronizada":"Taxa padronizada /100k","label":""},
                text_auto=".1f",
            )
            fig_cl.update_layout(showlegend=False, height=290)
            fig_cl.update_traces(textposition="outside")
            st.plotly_chart(fig_cl, use_container_width=True)

    # ── Scatter IDH × mortalidade ─────────────────────────────────────────────
    if not perfil.empty and "IDHM" in perfil.columns and "taxa_padronizada" in perfil.columns:
        st.divider()
        st.subheader("IDH Municipal × Taxa padronizada por cluster")
        fig_sc = px.scatter(
            perfil, x="IDHM", y="taxa_padronizada",
            size="n_municipios" if "n_municipios" in perfil.columns else None,
            color="label",
            color_discrete_map={v: CLUSTER_CORES[k] for k, v in CLUSTER_LABELS.items()},
            text="label",
            labels={"IDHM":"IDH Municipal","taxa_padronizada":"Taxa padronizada /100k",
                    "label":"Cluster","n_municipios":"Nº municípios"},
        )
        fig_sc.update_traces(textposition="top center", textfont_size=10)
        fig_sc.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Cada ponto representa um cluster. Tamanho proporcional ao número de municípios.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Perfis Municipais
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🗺️ Perfis Municipais":

    st.markdown("""
<div class="tcc-hero">
  <h1>🗺️ Perfis Municipais — Clusterização K-Means</h1>
  <p class="subtitle">4 grupos socioeconômicos identificados sem uso da variável-alvo</p>
</div>
""", unsafe_allow_html=True)

    if faltando:
        st.stop()

    perfil   = load_perfil()
    lbce     = _csv("lbce_por_cluster.csv")
    comp_reg = _csv("composicao_regional_cluster.csv")
    elbow    = _csv("elbow_silhouette.csv")

    if not lbce.empty and "CLUSTER" in lbce.columns:
        lbce["label"] = lbce["CLUSTER"].map(CLUSTER_LABELS)
    if not comp_reg.empty and "CLUSTER" in comp_reg.columns:
        comp_reg["label"] = comp_reg["CLUSTER"].map(CLUSTER_LABELS)

    st.markdown(
        "**K-Means (k=4)** aplicado sobre 7 variáveis socioeconômicas padronizadas (StandardScaler). "
        "A taxa de mortalidade **não entrou** no clustering — os perfis emergem exclusivamente "
        "do contexto socioeconômico, evitando circularidade na análise."
    )

    cluster_sel = st.selectbox(
        "Selecione um cluster para detalhar",
        options=list(CLUSTER_LABELS.keys()),
        format_func=lambda x: CLUSTER_LABELS[x],
    )

    if not perfil.empty and "CLUSTER" in perfil.columns:
        row = perfil[perfil["CLUSTER"] == cluster_sel]
        if not row.empty:
            row = row.iloc[0]
            cor = CLUSTER_CORES[cluster_sel]

            cols_kpi = st.columns(6)
            _kpi = lambda col, lbl, fmt="": cols_kpi[0]  # placeholder
            metricas_kpi = [
                ("n_municipios",    "Municípios",        lambda v: f"{int(v):,}".replace(",",".")),
                ("%_pop_brasil",    "% pop. Brasil",     lambda v: f"{v}%"),
                ("IDHM",           "IDH-M",             lambda v: f"{v:.3f}"),
                ("GINI",           "Índice de Gini",    lambda v: f"{v:.3f}"),
                ("taxa_padronizada","Taxa padronizada",  lambda v: f"{v:.1f} /100k"),
                ("ESPVIDA",        "Esp. de vida",      lambda v: f"{v:.1f} anos"),
            ]
            for i, (campo, nome, fmt) in enumerate(metricas_kpi):
                if campo in perfil.columns:
                    cols_kpi[i].metric(nome, fmt(row[campo]))

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Indicadores socioeconômicos do cluster")
        if not perfil.empty:
            row_p = perfil[perfil["CLUSTER"] == cluster_sel]
            if not row_p.empty:
                rp = row_p.iloc[0]
                ind_map = {
                    "PIB per capita real (R$)":   rp.get("PIB_pc_real", np.nan),
                    "Desp. saúde p.c. (R$)":     rp.get("desp_saude_pc_real", np.nan),
                    "Analfabetismo 15+ (%)":      rp.get("ANALF15M", np.nan),
                    "Acesso à água (%)" :          rp.get("T_AGUA", np.nan),
                    "Esperança de vida (anos)":   rp.get("ESPVIDA", np.nan),
                }
                df_ind = pd.DataFrame({"Indicador": list(ind_map.keys()),
                                       "Valor": list(ind_map.values())}).dropna()
                if not df_ind.empty:
                    fig_ind = px.bar(df_ind, x="Valor", y="Indicador", orientation="h",
                                     color_discrete_sequence=[CLUSTER_CORES[cluster_sel]],
                                     text_auto=".1f")
                    fig_ind.update_layout(height=280, showlegend=False)
                    st.plotly_chart(fig_ind, use_container_width=True)

    with col_b:
        st.subheader("Composição geográfica")
        if not comp_reg.empty and "CLUSTER" in comp_reg.columns:
            row_cr = comp_reg[comp_reg["CLUSTER"] == cluster_sel]
            if not row_cr.empty:
                regioes_disp = [r for r in ["N","NE","CO","SE","S"] if r in row_cr.columns]
                vals = [float(row_cr.iloc[0][r]) for r in regioes_disp]
                nomes = [REGIOES[r] for r in regioes_disp]
                df_pie = pd.DataFrame({"Região": nomes, "% municípios": vals})
                df_pie = df_pie[df_pie["% municípios"] > 0]
                if not df_pie.empty:
                    fig_pie = px.pie(df_pie, names="Região", values="% municípios",
                                     color_discrete_sequence=px.colors.qualitative.Set1, hole=0.4)
                    fig_pie.update_layout(height=280)
                    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Composição regional de todos os clusters ────────────────────────────
    if not comp_reg.empty:
        st.subheader("Composição regional de todos os clusters")
        comp_melt = comp_reg.melt(
            id_vars=["CLUSTER","label"] if "label" in comp_reg.columns else ["CLUSTER"],
            value_vars=[r for r in ["N","NE","CO","SE","S"] if r in comp_reg.columns],
            var_name="REGIAO", value_name="pct",
        )
        if "label" not in comp_melt.columns:
            comp_melt["label"] = comp_melt["CLUSTER"].map(CLUSTER_LABELS)
        comp_melt["regiao_nome"] = comp_melt["REGIAO"].map(REGIOES)

        fig_reg = px.bar(
            comp_melt, x="label", y="pct", color="regiao_nome", barmode="stack",
            color_discrete_sequence=["#0ea5e9","#f97316","#8b5cf6","#22c55e","#ef4444"],
            labels={"label":"","pct":"% dos municípios","regiao_nome":"Região"},
            text_auto=".0f",
        )
        fig_reg.update_layout(height=360)
        st.plotly_chart(fig_reg, use_container_width=True)

    st.divider()

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Composição LBCE por cluster")
        if not lbce.empty and "CLUSTER" in lbce.columns:
            row_l = lbce[lbce["CLUSTER"] == cluster_sel]
            if not row_l.empty:
                rl = row_l.iloc[0]
                nomes_lbce = ["D. Infecciosas","D. Não Transmissíveis","Imunopreveníveis","Morte Materna"]
                cols_lbce  = ["doencas_infecciosas","doencas_nao_transmissiveis",
                              "imunoprevencao","morte_materna"]
                vals_lbce  = [float(rl.get(c, 0)) for c in cols_lbce]
                fig_lbce = px.pie(names=nomes_lbce, values=vals_lbce,
                                  color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
                fig_lbce.update_layout(height=290)
                st.plotly_chart(fig_lbce, use_container_width=True)

    with col_d:
        st.subheader("Radar comparativo — todos os clusters")
        cats   = ["taxa_padronizada","IDHM","GINI","T_AGUA","ESPVIDA"]
        labels = ["Taxa padronizada","IDH-M","Gini","Água (%)","Esp. de vida"]
        cats_ok = [c for c in cats if c in perfil.columns]
        if len(cats_ok) >= 3:
            fig_rad = go.Figure()
            for _, rr in perfil.iterrows():
                idx = int(rr["CLUSTER"])
                def norm(col):
                    mn, mx = perfil[col].min(), perfil[col].max()
                    return float((rr[col] - mn) / (mx - mn + 1e-9))
                vals_r = [norm(c) for c in cats_ok] ; vals_r += [vals_r[0]]
                lbl_r  = [labels[cats.index(c)] for c in cats_ok] + [labels[cats.index(cats_ok[0])]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals_r, theta=lbl_r,
                    name=rr.get("label", CLUSTER_LABELS[idx]),
                    fill="toself", line_color=CLUSTER_CORES[idx], opacity=0.6,
                ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                height=290,
            )
            st.plotly_chart(fig_rad, use_container_width=True)

    # ── Como k=4 foi escolhido ───────────────────────────────────────────────
    with st.expander("Como o k=4 foi escolhido? — Método do Cotovelo e Silhouette"):
        if not elbow.empty:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                fig_el = go.Figure()
                fig_el.add_trace(go.Scatter(
                    x=elbow["k"], y=elbow["inertia"],
                    mode="lines+markers", marker=dict(size=8, color="#2563eb"),
                    line=dict(width=2), name="Inertia (WCSS)",
                ))
                fig_el.add_vline(x=4, line_dash="dash", line_color="#ef4444",
                                 annotation_text="k=4 escolhido")
                fig_el.update_layout(height=260, title="Método do Cotovelo",
                                     xaxis_title="k", yaxis_title="Inertia (WCSS)")
                st.plotly_chart(fig_el, use_container_width=True)

            with col_e2:
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=elbow["k"], y=elbow["silhouette"],
                    mode="lines+markers", marker=dict(size=8, color="#f97316"),
                    line=dict(width=2), name="Silhouette",
                ))
                fig_sil.add_vline(x=4, line_dash="dash", line_color="#ef4444",
                                  annotation_text="k=4 escolhido")
                fig_sil.update_layout(height=260, title="Silhouette Score",
                                      xaxis_title="k", yaxis_title="Silhouette médio")
                st.plotly_chart(fig_sil, use_container_width=True)

            st.markdown(
                "**k=4 foi escolhido pelo equilíbrio entre separação estatística e "
                "interpretabilidade geográfica** — 4 perfis mapeiam as 4 realidades "
                "de desenvolvimento municipal no Brasil: Norte/NE vulnerável, "
                "interior em transição, cidades médias Centro-Sul e grandes centros."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — Análise de Causas
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔬 Análise de Causas":

    st.markdown("""
<div class="tcc-hero">
  <h1>🔬 Padrões de Causas de Óbito por Cluster</h1>
  <p class="subtitle">Top CID-10 por cluster · Análise de Lift (distintividade epidemiológica)</p>
</div>
""", unsafe_allow_html=True)

    if faltando:
        st.stop()

    top_causas = _csv("top_causas_cluster.csv")
    lift_df    = _csv("lift_causas_cluster.csv")

    if not lift_df.empty and "CLUSTER" in lift_df.columns:
        lift_df["label"] = lift_df["CLUSTER"].map(CLUSTER_LABELS)

    tab1, tab2 = st.tabs(["📋 Top causas por cluster", "📐 Lift — Distintividade epidemiológica"])

    with tab1:
        st.markdown(
            "Proporção de cada CID-10 no total de óbitos evitáveis do cluster. "
            "Doenças cardiovasculares e diabetes dominam em todos — transição epidemiológica brasileira."
        )
        col_f, col_g = st.columns([1, 2])
        with col_f:
            cl_tc  = st.selectbox("Cluster", list(CLUSTER_LABELS.keys()),
                                  format_func=lambda x: CLUSTER_LABELS[x], key="tc")
            n_top  = st.slider("Nº de causas", 5, 15, 10)
        with col_g:
            if not top_causas.empty:
                df_tc = top_causas[top_causas["CLUSTER"] == cl_tc].head(n_top).copy()
                df_tc = _rotular_cid(df_tc)
                if not df_tc.empty:
                    fig_tc = px.bar(
                        df_tc.sort_values("pct"), x="pct", y="descricao",
                        orientation="h",
                        color_discrete_sequence=[CLUSTER_CORES[cl_tc]],
                        text="pct",
                        labels={"pct":"% dos óbitos do cluster","descricao":""},
                        hover_data={"CID_CAT3":True,"n":True,"descricao":False},
                    )
                    fig_tc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_tc.update_layout(height=max(300, n_top * 36), showlegend=False)
                    st.plotly_chart(fig_tc, use_container_width=True)

    with tab2:
        st.markdown(
            "**Lift** = frequência da causa neste cluster ÷ frequência nacional.  \n"
            "Lift = 2,0 → a causa é **2× mais frequente** aqui do que na média nacional. "
            "Revela o que torna cada cluster único epidemiologicamente."
        )
        cl_lift = st.selectbox("Cluster", list(CLUSTER_LABELS.keys()),
                               format_func=lambda x: CLUSTER_LABELS[x], key="lift")
        st.info(NARRATIVAS_CLUSTER[cl_lift])

        if not lift_df.empty:
            df_l = lift_df[lift_df["CLUSTER"] == cl_lift].copy()
            df_l = df_l.sort_values("lift", ascending=False).head(10)
            df_l = _rotular_cid(df_l)
            df_l["label_bar"] = df_l["CID_CAT3"] + " — " + df_l["descricao"]

            fig_lift = px.bar(
                df_l.sort_values("lift"), x="lift", y="label_bar",
                orientation="h",
                color="lift", color_continuous_scale="RdYlGn",
                text="lift",
                labels={"lift":"Lift (razão vs. média nacional)","label_bar":""},
            )
            fig_lift.add_vline(x=1.0, line_dash="dash", line_color="gray",
                               annotation_text="Média nacional (lift = 1)")
            fig_lift.update_traces(texttemplate="%{text:.2f}×", textposition="outside")
            fig_lift.update_layout(height=420, coloraxis_showscale=False)
            st.plotly_chart(fig_lift, use_container_width=True)
            st.caption("Top 10 causas com maior lift no cluster selecionado.")

    # ── Assinatura epidemiológica (resumo) ──────────────────────────────────
    st.divider()
    st.subheader("Assinatura epidemiológica por cluster")
    st.markdown(
        "**Resultado-chave da mineração:** a variação no *tipo* de causa é o que diferencia "
        "os clusters — não apenas a quantidade de óbitos."
    )
    topo = (lift_df.sort_values("lift", ascending=False).groupby("CLUSTER").head(1)
            .set_index("CLUSTER") if not lift_df.empty else pd.DataFrame())
    resumo_data = {
        "Cluster": list(CLUSTER_LABELS.values()),
        "Causa mais distintiva (lift)": [
            (f"{DESCRICOES_CID.get(topo.loc[c, 'CID_CAT3'], topo.loc[c, 'CID_CAT3'])} "
             f"({topo.loc[c, 'CID_CAT3']}) · {_br(topo.loc[c, 'lift'], 2)}×") if c in topo.index else "—"
            for c in CLUSTER_LABELS
        ],
        "Interpretação": [
            "Mortes ligadas à gestação, à nutrição e ao saneamento",
            "Parasitoses ligadas ao saneamento básico",
            "Arboviroses (dengue) em cidades médias",
            "Perfil de causas próximo da média nacional",
        ],
    }
    st.dataframe(pd.DataFrame(resumo_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — Modelagem Estatística
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📈 Modelagem Estatística":

    st.markdown("""
<div class="tcc-hero">
  <h1>📈 Modelagem Estatística e Preditiva</h1>
  <p class="subtitle">OLS painel (cluster-robust) · Random Forest · XGBoost · ANOVA de validação</p>
</div>
""", unsafe_allow_html=True)

    if faltando:
        st.stop()

    metricas = _csv("metricas_modelos.csv")
    coefs    = _csv("coeficientes_ols.csv")
    imp      = _csv("importancia_features.csv")

    if not coefs.empty:
        coefs["feature_label"] = coefs["feature"].map(FEATURE_NOMES).fillna(coefs["feature"])
    if not imp.empty:
        imp["feature_label"] = imp["feature"].map(FEATURE_NOMES).fillna(imp["feature"])

    # ── Desempenho dos modelos ──────────────────────────────────────────────
    st.subheader("Desempenho dos modelos (hold-out 20% dos municípios)")
    if not metricas.empty:
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], metricas.iterrows()):
            col.metric(row["modelo"], f"R² = {row['R2']:.3f}",
                       f"RMSE {row['RMSE']:.1f} | MAE {row['MAE']:.1f}")

    st.warning(
        f"**R² entre {_br(metricas['R2'].min(), 3)} e {_br(metricas['R2'].max(), 3)} — esperado e não invalidante:** "
        "variáveis socioeconômicas capturam "
        "padrões estruturais mas não fatores locais (qualidade da rede de saúde, "
        "sub-registro de óbitos, comportamentos). O que importa é a **direção, magnitude "
        "e significância dos coeficientes**."
    )

    if not metricas.empty:
        fig_met = px.bar(
            metricas.melt(id_vars="modelo", value_vars=["RMSE","MAE"]),
            x="modelo", y="value", color="variable", barmode="group",
            labels={"value":"Erro (óbitos/100k)","variable":"Métrica","modelo":""},
            color_discrete_sequence=["#ef4444","#f97316"], text_auto=".1f",
        )
        fig_met.update_layout(height=240)
        st.plotly_chart(fig_met, use_container_width=True)

    st.divider()

    # ── Validação estatística dos clusters (ANOVA) ──────────────────────────
    st.subheader("Validação estatística dos clusters — ANOVA one-way")
    st.markdown(
        "H₀: as médias de mortalidade são iguais entre os 4 clusters. "
        "Rejeitar H₀ confirma que o K-Means separou grupos com mortalidade distinta."
    )
    anova = _csv("anova_clusters.csv")
    if not anova.empty:
        av = anova.set_index("medida")
        va1, va2 = st.columns(2)
        for col, chave, nome in ((va1, "taxa_bruta", "Taxa bruta"), (va2, "taxa_padronizada", "Taxa padronizada")):
            r = av.loc[chave]
            col.metric(f"ANOVA — {nome}", f"F = {_br(r['F'], 2)}", f"p = {_p_cientifico(r['p_value'])}",
                       help=f"Graus de liberdade: {int(r['gl_entre'])} e {int(r['gl_dentro'])} (médias por município)")
        pf_ = load_perfil()
        st.info(
            "Os p-valores são muito pequenos: os 4 clusters têm taxas médias **estatisticamente diferentes**, "
            "mesmo que a taxa **não tenha entrado** na clusterização. Na taxa padronizada, porém, a diferença é "
            f"pequena em valores absolutos: de {_br(pf_['taxa_padronizada'].min())} a "
            f"{_br(pf_['taxa_padronizada'].max())} óbitos por 100 mil habitantes."
        )

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Coeficientes OLS — impacto na taxa padronizada")
        if not coefs.empty:
            df_coef = coefs[
                (coefs["feature"] != "const") &
                (~coefs["feature"].str.startswith("ANO_", na=False)) &
                (~coefs["feature"].str.startswith("REG_", na=False))
            ].copy()
            df_coef["cor"] = df_coef["coef"].apply(lambda x: "#ef4444" if x > 0 else "#22c55e")

            fig_coef = go.Figure()
            df_s = df_coef.sort_values("coef")
            fig_coef.add_trace(go.Bar(
                x=df_s["coef"], y=df_s["feature_label"], orientation="h",
                marker_color=df_s["cor"].tolist(),
                text=df_s.get("sig", pd.Series("", index=df_s.index)).fillna(""),
                textposition="outside",
                customdata=df_s["coef"].round(3),
                hovertemplate="<b>%{y}</b><br>Coef: %{customdata}<extra></extra>",
            ))
            fig_coef.add_vline(x=0, line_color="gray", line_dash="dash")
            fig_coef.update_layout(height=400, showlegend=False,
                                   xaxis_title="Efeito na taxa padronizada /100k")
            st.plotly_chart(fig_coef, use_container_width=True)
            st.caption("🔴 aumenta mortalidade · 🟢 reduz mortalidade · *** p<0,001 · ** p<0,01 · * p<0,05")

        with st.expander("Interpretação dos principais coeficientes"):
            cf = coefs.set_index("feature")

            def _c(f):
                r = cf.loc[f]
                sig = r["sig"].replace("*", r"\*") if isinstance(r["sig"], str) and r["sig"] else "não signif."
                return r["coef"], sig

            def _s(x, casas=1):  # número com sinal: +1,0 / −1,0
                return ("+" if x >= 0 else "−") + _br(abs(x), casas)

            g, gs = _c("GINI"); e, es = _c("ESPVIDA"); po, ps = _c("LOG_POPULACAO")
            d, ds = _c("LOG_DESPESA_SAUDE_PC"); rd, rs = _c("RDPC"); i, is_ = _c("IDHM")
            st.markdown(f"""
- **Gini ({_s(g)}; {gs}):** maior efeito — um aumento de 0,1 no índice → {_s(g / 10)} óbitos/100k
- **Esperança de vida ({_s(e)}; {es}):** cada ano a mais → {_s(e)} óbitos/100k
- **Tamanho da população, log ({_s(po)}; {ps}):** municípios maiores têm taxas maiores (possivelmente melhor registro de óbitos)
- **Renda per capita ({_s(rd, 2)}; {rs}):** cada R$ 100 a mais → {_s(rd * 100)} óbitos/100k
- **IDH-M ({_s(i)}; {is_}) e despesa em saúde, log ({_s(d)}; {ds}):** sem efeito significativo quando se comparam municípios em condições semelhantes
""")

    with col_b:
        st.subheader("Importância de features — modelos de ML")
        modelo_sel = st.radio("Modelo", ["Random Forest","XGBoost"], horizontal=True)
        col_imp = "imp_rf" if modelo_sel == "Random Forest" else "imp_xgb"

        if not imp.empty and col_imp in imp.columns:
            df_imp = imp.sort_values(col_imp, ascending=False).copy()
            fig_imp = px.bar(
                df_imp.sort_values(col_imp), x=col_imp, y="feature_label",
                orientation="h", color=col_imp,
                color_continuous_scale="Blues", text_auto=".3f",
                labels={col_imp:"Importância (MDI)","feature_label":""},
            )
            fig_imp.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("MDI = Mean Decrease in Impurity. Coerente com coeficientes OLS.")

    # ── IC 95% ──────────────────────────────────────────────────────────────
    if not coefs.empty and "ci_low" in coefs.columns:
        st.divider()
        st.subheader("Intervalos de confiança 95% — coeficientes OLS significativos")
        df_sig = coefs[
            coefs.get("sig", pd.Series("", index=coefs.index)).fillna("").str.len() > 0
        ].copy().sort_values("coef")

        if not df_sig.empty:
            df_sig["feature_label"] = df_sig["feature"].map(FEATURE_NOMES).fillna(df_sig["feature"])
            fig_ci = go.Figure()
            for _, r in df_sig.iterrows():
                cor = "#ef4444" if r["coef"] > 0 else "#22c55e"
                lbl = r["feature_label"]
                fig_ci.add_trace(go.Scatter(
                    x=[r["ci_low"], r["ci_high"]], y=[lbl, lbl],
                    mode="lines", line=dict(color=cor, width=4), showlegend=False,
                ))
                fig_ci.add_trace(go.Scatter(
                    x=[r["coef"]], y=[lbl],
                    mode="markers", marker=dict(color=cor, size=10),
                    showlegend=False,
                    hovertemplate=f"<b>{lbl}</b><br>Coef: {r['coef']:.3f}<br>IC 95%: [{r['ci_low']:.3f}, {r['ci_high']:.3f}]<extra></extra>",
                ))
            fig_ci.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_ci.update_layout(height=400, xaxis_title="Coeficiente (IC 95%)")
            st.plotly_chart(fig_ci, use_container_width=True)
            st.caption("Barras que cruzam o zero → não significativo ao nível 5%. Erros-padrão robustos agrupados por município.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 6 — Conclusões
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📋 Conclusões":

    st.markdown("""
<div class="tcc-hero">
  <h1>📋 Síntese e Conclusões</h1>
  <p class="subtitle">Achados principais · Limitações · Contribuições · Trabalhos futuros</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Retomando a Pergunta de Pesquisa")
    st.markdown("""
<div class="box-pergunta">
<strong>Em que medida indicadores socioeconômicos municipais explicam as taxas de mortalidade
por causas evitáveis no Brasil entre 2022 e 2024?</strong>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Achados Principais")

    rg = _csv("taxa_regiao_comparacao.csv").set_index("REGIAO")
    cf = _csv("coeficientes_ols.csv").set_index("feature")
    pf_ = load_perfil()
    met = _csv("metricas_modelos.csv")
    av = _csv("anova_clusters.csv").set_index("medida")
    achados = [
        ("Paradoxo da taxa bruta",
         f"Pela taxa bruta, o Norte aparecia com a menor mortalidade ({_br(rg.loc['N', 'taxa_bruta'])}/100k). "
         f"Após a padronização etária direta, sobe para {_br(rg.loc['N', 'taxa_padronizada'])}/100k, e o Sul cai "
         f"de {_br(rg.loc['S', 'taxa_bruta'])} para {_br(rg.loc['S', 'taxa_padronizada'])}/100k: "
         "as diferenças entre as regiões praticamente desaparecem."),
        ("A desigualdade pesa mais que o nível de desenvolvimento",
         f"O Gini ({_br(cf.loc['GINI', 'coef'])}) é o coeficiente OLS de maior efeito (p < 0,001). "
         f"O IDH-M (p = {_br(cf.loc['IDHM', 'p_value'], 3)}) e o gasto em saúde "
         f"(p = {_br(cf.loc['LOG_DESPESA_SAUDE_PC', 'p_value'], 3)}) não foram significativos."),
        ("4 perfis municipais",
         f"O K-Means (k=4) separou grupos com taxas médias estatisticamente diferentes (ANOVA, "
         f"p = {_p_cientifico(av.loc['taxa_padronizada', 'p_value'])}), mas a diferença na taxa padronizada "
         f"é pequena: de {_br(pf_['taxa_padronizada'].min())} a {_br(pf_['taxa_padronizada'].max())}/100k."),
        ("Assinatura epidemiológica por cluster",
         "Eclâmpsia, desnutrição grave e esquistossomose marcam os municípios mais vulneráveis; "
         "a dengue diferencia as cidades médias; nos grandes centros o perfil de causas é próximo "
         "da média nacional."),
        ("Poder explicativo limitado",
         f"Os modelos explicam cerca de {_br(met['R2'].min() * 100, 0)}% a {_br(met['R2'].max() * 100, 0)}% "
         f"das diferenças entre municípios (R² de {_br(met['R2'].min(), 3)} a {_br(met['R2'].max(), 3)}): "
         "fatores não medidos, como a rede de saúde local e o sub-registro, também pesam."),
    ]

    for titulo, texto in achados:
        st.markdown(f"""
<div class="box-achado">
<strong>{titulo}</strong><br>{texto}
</div>
""", unsafe_allow_html=True)

    st.divider()
    col_lim, col_fut = st.columns(2)

    with col_lim:
        st.markdown("### Limitações")
        limitacoes = [
            ("Sub-registro heterogêneo",
             "Municípios remotos tendem a registrar menos óbitos no SIM, então as taxas do "
             "Cluster 0 podem estar subestimadas. Não corrigível sem dados de captura-recaptura."),
            ("Atlas DH defasado",
             "IDH-M e Gini baseados no Censo 2010 (12–14 anos de defasagem). "
             "Atlas com Censo 2022 ainda não publicado pelo PNUD/IPEA."),
            ("Falácia ecológica",
             "Análise ao nível municipal — inferências não se transferem ao nível individual."),
            ("Denominador estático",
             "Estrutura etária do Censo 2022 usada para todos os anos (2022–2024)."),
            ("Despesa em saúde",
             "O coeficiente não foi significativo; como o gasto também responde à carga de doença "
             "(causalidade reversa), seu efeito é difícil de medir com dados observacionais."),
            ("Associação, não causalidade",
             "O estudo é observacional: os coeficientes indicam associações, não efeitos causais."),
        ]
        for titulo, texto in limitacoes:
            st.markdown(f"""
<div class="box-limitacao">
<strong>{titulo}:</strong> {texto}
</div>
""", unsafe_allow_html=True)

    with col_fut:
        st.markdown("### Trabalhos Futuros")
        futuros = [
            "Incorporar rede de atenção à saúde (leitos/hab, médicos/hab, cobertura APS via CNES/DAB)",
            "Aplicar padronização indireta (SMR) para municípios pequenos (pop < 5k)",
            "Tratar sub-registro com modelos de captura-recaptura em áreas remotas",
            "Atualizar Atlas DH com Censo 2022 assim que disponível",
            "Adotar efeitos fixos municipais para variáveis que variam no tempo (despesa, PIB)",
            "Integrar dados da COVID-19 para análise do impacto pandêmico na mortalidade evitável",
        ]
        for f in futuros:
            st.markdown(f"""
<div class="box-futuro">• {f}</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Contribuições do Trabalho")
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.success(
        "**Pipeline reprodutível**  \nIntegração de 4 fontes públicas heterogêneas "
        "via FTP do DATASUS e APIs REST, com cache e retomada automática."
    )
    col_c2.success(
        "**Evidência quantitativa**  \nGini identificado como o determinante "
        "mais forte (p < 0,001) da mortalidade evitável municipal."
    )
    col_c3.success(
        "**Plataforma aberta**  \nDashboard interativo para gestores públicos "
        "identificarem perfis municipais e orientarem políticas focadas."
    )

    st.divider()
    st.markdown("### Referências")
    st.markdown("""
- MALTA, D.C. et al. *Lista de causas de mortes evitáveis por intervenções do SUS.* Epidemiol. Serv. Saúde, 16(4):233–244, 2007.
- MALTA, D.C. et al. *Atualização da lista de causas de mortes evitáveis.* Epidemiol. Serv. Saúde, 19(2):173–176, 2010.
- PNUD/IPEA/FJP. *Atlas do Desenvolvimento Humano no Brasil.* Brasília, 2013.
- DATASUS/MS. *Sistema de Informações sobre Mortalidade (SIM).* Brasília: MS, 2024.
- IBGE. *Censo Demográfico 2022.* Rio de Janeiro: IBGE, 2023.
- BREIMAN, L. *Random Forests.* Machine Learning, 45(1):5–32, 2001.
- CHEN, T.; GUESTRIN, C. *XGBoost: A Scalable Tree Boosting System.* KDD, 2016.
- MACQUEEN, J. *Some methods for classification and analysis of multivariate observations.* Berkeley Symposium on Mathematical Statistics and Probability, 5., 1967. v. 1, p. 281–297.
- WORLD HEALTH ORGANIZATION. *World Health Statistics 2023.* Geneva: WHO, 2023.
""")

    st.markdown("""
<div class="footer-bar">
TCC · IESB 2026/01 · Ciência da Computação ·
<strong>João Marcos Santos e Carvalho</strong> (2312130063) ·
<strong>Maria Clara Fontenele Silva</strong> (2312130230) ·
Orientador: <strong>Prof. Cristiano Lehrer</strong>
</div>
""", unsafe_allow_html=True)
