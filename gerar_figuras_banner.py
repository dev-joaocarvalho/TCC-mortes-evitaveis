"""
Gera as figuras do banner (pôster) a partir dos CSVs em models/ (saída: figuras/banner/).

Cada figura é desenhada na proporção exata do espaço que ocupa no banner, com
letras grandes para leitura à distância.

Uso:
    python gerar_figuras_banner.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent
M = ROOT / "models"
OUT = ROOT / "figuras" / "banner"
OUT.mkdir(parents=True, exist_ok=True)

CORES = {0: "#ef4444", 1: "#f97316", 2: "#3b82f6", 3: "#22c55e"}
NOME = {0: "Cluster 0<br>Norte/NE Vulnerável", 1: "Cluster 1<br>Interior em Desenvolvimento",
        2: "Cluster 2<br>Sul/SE Médio", 3: "Cluster 3<br>Grandes Centros"}
BASE = dict(template="plotly_white", separators=",.",
            font=dict(family="Arial", size=26, color="#111827"))


def salvar(fig, nome, w, h):
    fig.write_image(OUT / nome, width=w, height=h, scale=2)
    print("  ", OUT / nome, f"({w}x{h})")


pf = pd.read_csv(M / "perfil_clusters.csv")

# Taxa padronizada por cluster (espaço no banner: 1027 x 380 pt)
f1 = go.Figure(go.Bar(x=pf["CLUSTER"].map(NOME), y=pf["taxa_padronizada"],
                      marker_color=[CORES[c] for c in pf["CLUSTER"]],
                      text=pf["taxa_padronizada"], texttemplate="%{text:.1f}",
                      textposition="outside", textfont=dict(size=30), cliponaxis=False))
f1.update_layout(yaxis=dict(title="Óbitos evitáveis por 100 mil<br>hab. (taxa padronizada)",
                            range=[0, pf["taxa_padronizada"].max() * 1.18], title_font=dict(size=22)),
                 margin=dict(l=10, r=10, t=20, b=10), showlegend=False, **BASE)
f1.update_xaxes(tickfont=dict(size=23))
salvar(f1, "banner_taxa_cluster.png", 1300, round(1300 * 380 / 1027))

# Radar (espaço: 649 x 447 pt; a legenda dos clusters já é uma imagem separada no banner)
cats = ["taxa_padronizada", "IDHM", "GINI", "T_AGUA", "ESPVIDA"]
labs = ["Taxa<br>padronizada", "IDHM", "Gini", "Água (%)", "Esperança<br>de vida"]
f2 = go.Figure()
for _, r in pf.iterrows():
    v = [float((r[c] - pf[c].min()) / (pf[c].max() - pf[c].min() + 1e-9)) for c in cats]
    f2.add_trace(go.Scatterpolar(r=v + v[:1], theta=labs + labs[:1], fill="toself",
                                 line=dict(color=CORES[int(r["CLUSTER"])], width=3.5), opacity=0.55))
f2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=16), angle=36,
                                            tickvals=[0, 0.5, 1], ticktext=["0", "0,5", "1"]),
                            angularaxis=dict(tickfont=dict(size=26)),
                            # radar à esquerda e mais baixo: o canto superior direito fica livre
                            # para a imagem da legenda dos clusters, que fica sobre ele no banner
                            domain=dict(x=[0, 0.66], y=[0, 0.9])),
                 showlegend=False, margin=dict(l=120, r=10, t=10, b=90), **BASE)
salvar(f2, "banner_radar.png", 900, round(900 * 447 / 649))


# Fluxo do trabalho (espaço: 844 x 698 pt)
def caixa(fig, x0, x1, y0, y1, texto, cor, borda, fonte=24):
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=cor,
                  line=dict(color=borda, width=2.5), layer="below")
    fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=texto, showarrow=False,
                       font=dict(size=fonte), align="center", xanchor="center", yanchor="middle")


def seta(fig, x0, y0, x1, y1):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=3, arrowcolor="#374151", text="")


f3 = go.Figure()
f3.update_layout(xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]),
                 margin=dict(l=4, r=4, t=4, b=4), plot_bgcolor="white", **BASE)
CZ, AZ, VD, LJ = "#f3f4f6", "#dbeafe", "#dcfce7", "#ffedd5"
caixa(f3, 0.0, 1.0, 0.86, 1.0, "<b>Fontes de dados abertos</b><br>SIM/DATASUS · IBGE · Siconfi · Ipeadata", CZ, "#9ca3af")
etl = [(0.0, 0.32, "<b>1. Extração</b><br>FTP e APIs<br><i>requests, datasus-dbc</i>"),
       (0.34, 0.66, "<b>2. Transformação</b><br>LBE · taxas bruta<br>e padronizada"),
       (0.68, 1.0, "<b>3. Carga</b><br>matriz município<br>× ano · <i>Parquet</i>")]
for i, (x0, x1, t) in enumerate(etl):
    caixa(f3, x0, x1, 0.58, 0.80, t, AZ, "#60a5fa", fonte=22)
    if i < 2:
        seta(f3, x1, 0.69, etl[i + 1][0], 0.69)
seta(f3, 0.5, 0.86, 0.5, 0.80)
ana = [(0.0, 0.32, "<b>Regressão</b><br>OLS · Random<br>Forest · XGBoost"),
       (0.34, 0.66, "<b>Clusterização</b><br>K-Means<br>(4 grupos)"),
       (0.68, 1.0, "<b>Mineração</b><br>de causas<br>(lift CID-10)")]
seta(f3, 0.84, 0.58, 0.84, 0.50)
f3.add_shape(type="line", x0=0.16, x1=0.84, y0=0.50, y1=0.50, line=dict(color="#374151", width=3))
for x0, x1, t in ana:
    seta(f3, (x0 + x1) / 2, 0.50, (x0 + x1) / 2, 0.44)
    caixa(f3, x0, x1, 0.22, 0.44, t, VD, "#4ade80", fonte=22)
f3.add_annotation(x=0.5, y=0.18, text="<i>Python 3.12 · pandas · statsmodels · scikit-learn · xgboost</i>",
                  showarrow=False, font=dict(size=20), xanchor="center", yanchor="middle")
seta(f3, 0.5, 0.15, 0.5, 0.115)
caixa(f3, 0.08, 0.92, 0.0, 0.115, "<b>Resultados e dashboard interativo</b><br><i>Plotly · Streamlit</i>", LJ, "#fb923c", fonte=22)
salvar(f3, "banner_fluxo.png", 1200, round(1200 * 698 / 844))
