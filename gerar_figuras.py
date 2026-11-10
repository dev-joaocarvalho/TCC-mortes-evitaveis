"""
Gera as Figuras 1–5 do artigo a partir dos CSVs em models/ (saída: figuras/).

Uso:
    python gerar_figuras.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent
M = ROOT / "models"
OUT = ROOT / "figuras"
OUT.mkdir(exist_ok=True)

CORES = {0: "#ef4444", 1: "#f97316", 2: "#3b82f6", 3: "#22c55e"}
ROTULOS = {0: "cluster 0 - Norte/NE Vulnerável", 1: "cluster 1 - Interior em Desenvolvimento",
           2: "cluster 2 - Sul/SE Médio", 3: "cluster 3 - Grandes Centros"}
BASE = dict(template="plotly_white", font=dict(family="Arial", size=15),
            margin=dict(l=10, r=10, t=40, b=10))


def salvar(fig, nome, w, h):
    fig.write_image(OUT / nome, width=w, height=h, scale=2)
    print("  ", OUT / nome)


# Figura 1 — métricas dos modelos (R², RMSE, MAE)
met = pd.read_csv(M / "metricas_modelos.csv")
long = met.melt(id_vars="modelo", value_vars=["R2", "RMSE", "MAE"], var_name="Métrica", value_name="Valor")
f1 = px.bar(long, x="modelo", y="Valor", color="Métrica", barmode="group", text_auto=".2f",
            color_discrete_map={"R2": "#3b82f6", "RMSE": "#ef4444", "MAE": "#f97316"},
            labels={"modelo": ""})
titulo = "   |   ".join(f"{r.modelo}: R² = {r.R2:.3f}" for r in met.itertuples())
f1.update_layout(title=dict(text=titulo, font=dict(size=15)), **BASE)
salvar(f1, "figura1_metricas.png", 840, 650)

# Figura 2 — coeficientes OLS (vermelho aumenta, verde reduz; asteriscos = significância)
co = pd.read_csv(M / "coeficientes_ols.csv")
co = co[co["feature"] != "const"].sort_values("coef")
rot = co["feature"] + " " + co["sig"].fillna("")   # asteriscos ao lado do nome
f2 = go.Figure(go.Bar(x=co["coef"], y=rot, orientation="h",
                      marker_color=["#ef4444" if c > 0 else "#22c55e" for c in co["coef"]]))
f2.add_vline(x=0, line_color="gray", line_dash="dash")
f2.update_layout(xaxis_title="Coeficiente", showlegend=False, **BASE)
salvar(f2, "figura2_coeficientes_ols.png", 450, 750)

# Figura 3 — importância das variáveis no Random Forest
imp = pd.read_csv(M / "importancia_features.csv").sort_values("imp_rf")
f3 = px.bar(imp, x="imp_rf", y="feature", orientation="h", color="imp_rf",
            color_continuous_scale="Blues", text_auto=".3f",
            labels={"imp_rf": "Importância", "feature": ""})
f3.update_layout(coloraxis_showscale=False, **BASE)
salvar(f3, "figura3_importancia_rf.png", 780, 830)

# Figura 4 — taxa padronizada por cluster
pf = pd.read_csv(M / "perfil_clusters.csv")
pf["rotulo"] = pf["CLUSTER"].map(ROTULOS)
f4 = go.Figure(go.Bar(x=pf["rotulo"], y=pf["taxa_padronizada"],
                      marker_color=[CORES[c] for c in pf["CLUSTER"]],
                      text=pf["taxa_padronizada"].round(1), textposition="outside"))
f4.update_layout(yaxis_title="Taxa padronizada /100k", showlegend=False,
                 yaxis_range=[0, pf["taxa_padronizada"].max() * 1.15], **BASE)
salvar(f4, "figura4_taxa_cluster.png", 1250, 441)

# Figura 5 — radar (min–max 0 a 1 entre clusters), mesma lógica do dashboard
cats = ["taxa_padronizada", "IDHM", "GINI", "T_AGUA", "ESPVIDA"]
labs = ["Taxa padronizada", "IDH-M", "Gini", "Água (%)", "Esp. de vida"]
f5 = go.Figure()
for _, r in pf.iterrows():
    v = [float((r[c] - pf[c].min()) / (pf[c].max() - pf[c].min() + 1e-9)) for c in cats]
    f5.add_trace(go.Scatterpolar(r=v + v[:1], theta=labs + labs[:1], fill="toself",
                                 line_color=CORES[int(r["CLUSTER"])], opacity=0.6,
                                 name=ROTULOS[int(r["CLUSTER"])]))
f5.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, **BASE)
salvar(f5, "figura5_radar.png", 880, 610)
