"""
Gera as Figuras 1–5 do artigo a partir dos CSVs em models/ (saída: figuras/).

As figuras são desenhadas na largura da coluna do artigo (500 px ≈ 8 cm), de modo
que os textos fiquem com cerca de 8–9 pt depois de impressos. Exportadas em 3x
para ficarem nítidas.

Uso:
    python gerar_figuras.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
M = ROOT / "models"
OUT = ROOT / "figuras"
OUT.mkdir(exist_ok=True)

LARGURA = 500                      # px = largura da coluna do artigo
FONTE = 19                         # px ≈ 8,5 pt impresso
VERMELHO, VERDE, AZUL, LARANJA = "#dc2626", "#16a34a", "#2563eb", "#f97316"
CORES = {0: "#ef4444", 1: "#f97316", 2: "#3b82f6", 3: "#22c55e"}
CLUSTER_NOME = {0: "Cluster 0 – Norte/NE Vulnerável", 1: "Cluster 1 – Interior em Desenvolvimento",
                2: "Cluster 2 – Sul/SE Médio", 3: "Cluster 3 – Grandes Centros"}
CLUSTER_2LINHAS = {0: "Cluster 0<br>Norte/NE<br>Vulnerável", 1: "Cluster 1<br>Interior em<br>Desenvolvimento",
                   2: "Cluster 2<br>Sul/SE<br>Médio", 3: "Cluster 3<br>Grandes<br>Centros"}
VARIAVEL = {
    "GINI": "Desigualdade (Gini)", "IDHM": "IDHM",
    "LOG_POPULACAO": "Tamanho da população (log)",
    "LOG_DESPESA_SAUDE_PC": "Gasto em saúde por hab. (log)",
    "LOG_PIB_PER_CAPITA": "PIB por habitante (log)",
    "T_ANALF15M": "Analfabetismo (%)", "T_AGUA": "Acesso à água (%)",
    "RDPC": "Renda por pessoa (R$)", "PMPOB": "Extrema pobreza (%)",
    "ESPVIDA": "Esperança de vida (anos)",
    "REG_N": "Região Norte", "REG_NE": "Região Nordeste",
    "REG_S": "Região Sul", "REG_SE": "Região Sudeste",
    "ANO_2023": "Ano 2023", "ANO_2024": "Ano 2024",
}
BASE = dict(template="plotly_white", separators=",.",
            font=dict(family="Arial", size=FONTE, color="#111827"))


def salvar(fig, nome, altura):
    fig.write_image(OUT / nome, width=LARGURA, height=altura, scale=3)
    print("  ", OUT / nome, f"({LARGURA}x{altura})")


# ── Figura 1 — desempenho dos modelos ─────────────────────────────────────────
met = pd.read_csv(M / "metricas_modelos.csv")
modelos = met["modelo"].str.replace("Random Forest", "Random<br>Forest")
f1 = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.13,
                   subplot_titles=("R² (maior = melhor)", "Erro (menor = melhor)"))
f1.add_trace(go.Bar(x=modelos, y=met["R2"], marker_color=AZUL, name="R²", showlegend=False,
                    text=met["R2"], texttemplate="%{text:.3f}", textposition="outside"), 1, 1)
for col, cor in (("RMSE", VERMELHO), ("MAE", LARANJA)):
    f1.add_trace(go.Bar(x=modelos, y=met[col], marker_color=cor, name=col,
                        text=met[col], texttemplate="%{text:.1f}", textposition="outside",
                        textfont=dict(size=15)), 1, 2)
f1.update_yaxes(range=[0, met["R2"].max() * 1.3], row=1, col=1, tickformat=".2f")
f1.update_yaxes(range=[0, met["RMSE"].max() * 1.2], row=1, col=2)
f1.update_layout(barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                 legend=dict(orientation="h", x=0.62, y=-0.2, xanchor="center"), **BASE)
f1.update_annotations(font_size=FONTE)
f1.update_xaxes(tickangle=0, tickfont=dict(size=14))
salvar(f1, "figura1_metricas.png", 420)

# ── Figura 2 — coeficientes OLS (legenda dentro da figura) ────────────────────
co = pd.read_csv(M / "coeficientes_ols.csv")
co = co[co["feature"] != "const"].sort_values("coef")
rot = co["feature"].map(VARIAVEL) + " " + co["sig"].fillna("")
f2 = go.Figure(go.Bar(
    x=co["coef"], y=rot, orientation="h",
    marker_color=[VERMELHO if c > 0 else VERDE for c in co["coef"]],
    cliponaxis=False))
f2.add_vline(x=0, line_color="gray", line_width=1)
for c, y in zip(co["coef"], rot):          # valor escrito à direita (barra positiva) ou do zero (negativa)
    txt = (f"{c:+.1f}" if abs(c) >= 1 else f"{c:+.2f}").replace(".", ",").replace("-", "−")
    f2.add_annotation(x=max(c, 0), y=y, text=txt, showarrow=False, xanchor="left", xshift=4,
                      font=dict(size=15))
f2.update_layout(
    title=dict(text=f"<span style='color:{VERMELHO}'>■</span> aumenta a mortalidade   "
                    f"<span style='color:{VERDE}'>■</span> reduz<br>"
                    "*** p<0,001   ** p<0,01   * p<0,05",
               x=0.5, xanchor="center", xref="container", y=0.975, font=dict(size=16)),
    xaxis=dict(title="Efeito (óbitos/100 mil)", range=[co["coef"].min() * 1.15, co["coef"].max() * 1.35]),
    margin=dict(l=10, r=10, t=80, b=10), showlegend=False, **BASE)
salvar(f2, "figura2_coeficientes_ols.png", 640)

# ── Figura 3 — importância das variáveis no Random Forest ─────────────────────
imp = pd.read_csv(M / "importancia_features.csv").sort_values("imp_rf")
f3 = go.Figure(go.Bar(x=imp["imp_rf"], y=imp["feature"].map(VARIAVEL), orientation="h",
                      marker=dict(color=imp["imp_rf"], colorscale="Blues", cmin=-0.03),
                      text=imp["imp_rf"], texttemplate="%{text:.3f}", textposition="outside",
                      textfont=dict(size=15), cliponaxis=False))
f3.update_layout(xaxis=dict(title="Importância no modelo", range=[0, imp["imp_rf"].max() * 1.22],
                            tickformat=".2f"),
                 margin=dict(l=10, r=10, t=10, b=10), **BASE)
salvar(f3, "figura3_importancia_rf.png", 600)

# ── Figura 4 — taxa padronizada por cluster ───────────────────────────────────
pf = pd.read_csv(M / "perfil_clusters.csv")
pf4 = pf.sort_values("CLUSTER", ascending=False)
f4 = go.Figure(go.Bar(y=pf4["CLUSTER"].map(CLUSTER_2LINHAS), x=pf4["taxa_padronizada"], orientation="h",
                      marker_color=[CORES[c] for c in pf4["CLUSTER"]],
                      text=pf4["taxa_padronizada"], texttemplate="%{text:.1f}",
                      textposition="outside", cliponaxis=False))
f4.update_layout(xaxis=dict(title="Óbitos evitáveis por 100 mil hab.<br>(taxa padronizada)",
                            range=[0, pf["taxa_padronizada"].max() * 1.2]),
                 margin=dict(l=10, r=10, t=10, b=10), showlegend=False, **BASE)
f4.update_yaxes(tickfont=dict(size=17))
salvar(f4, "figura4_taxa_cluster.png", 400)

# ── Figura 5 — radar (escala 0 a 1 entre clusters), legenda dentro ────────────
cats = ["taxa_padronizada", "IDHM", "GINI", "T_AGUA", "ESPVIDA"]
labs = ["Taxa<br>padronizada", "IDHM", "Gini", "Água (%)", "Esperança<br>de vida"]
f5 = go.Figure()
for _, r in pf.iterrows():
    v = [float((r[c] - pf[c].min()) / (pf[c].max() - pf[c].min() + 1e-9)) for c in cats]
    f5.add_trace(go.Scatterpolar(r=v + v[:1], theta=labs + labs[:1], fill="toself",
                                 line=dict(color=CORES[int(r["CLUSTER"])], width=2.5),
                                 opacity=0.55, name=CLUSTER_NOME[int(r["CLUSTER"])]))
f5.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=12), angle=36,
                                            tickvals=[0, 0.5, 1], ticktext=["0", "0,5", "1"]),
                            angularaxis=dict(tickfont=dict(size=17))),
                 legend=dict(orientation="v", x=0.5, xanchor="center", y=-0.08, yanchor="top",
                             font=dict(size=16)),
                 margin=dict(l=75, r=125, t=20, b=10), **BASE)
salvar(f5, "figura5_radar.png", 600)
