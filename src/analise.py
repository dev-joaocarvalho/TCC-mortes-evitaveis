"""
Análise — Clusterização, Regressão e Mineração de Causas
TCC | Mortalidade por Causas Evitáveis no Brasil | IESB 2026/01
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
import statsmodels.api as sm

try:
    from xgboost import XGBRegressor  # type: ignore
    _XGBOOST = True
except ImportError:
    _XGBOOST = False

ROOT       = Path(__file__).parent.parent
DATA_PROC  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Features usadas na clusterização (sem o target)
FEATURES_CLUSTER = ["IDHM", "GINI", "LOG_PIB_PER_CAPITA", "LOG_DESPESA_SAUDE_PC",
                    "T_ANALF15M", "T_AGUA", "ESPVIDA"]

# Mínimo de óbitos de uma causa no cluster (janela 2022–2024) para entrar no lift
LIFT_MIN_OBITOS = 50

# Desfecho da regressão: taxa padronizada por idade (método direto, Censo 2022)
TARGET = "TAXA_PADRONIZADA_100K"

# Features da regressão
FEATURES_REG = ["LOG_PIB_PER_CAPITA", "LOG_DESPESA_SAUDE_PC", "LOG_POPULACAO",
                "IDHM", "GINI", "RDPC", "PMPOB", "ESPVIDA", "T_ANALF15M", "T_AGUA"]


# ─── Utilitários ──────────────────────────────────────────────────────────────
def _carregar_fm() -> pd.DataFrame:
    path = DATA_PROC / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "feature_matrix.parquet não encontrado. Execute pipeline.py primeiro."
        )
    return pd.read_parquet(path)


def _cols_disponiveis(fm: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in fm.columns]


# ─── K-Means ──────────────────────────────────────────────────────────────────
def clusterizar(fm: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Aplica K-Means k=4 sobre features socioeconômicas padronizadas."""
    print("  [Cluster] Aplicando K-Means...")
    feats = _cols_disponiveis(fm, FEATURES_CLUSTER)
    if len(feats) < 2:
        print("  [Cluster] Features insuficientes para clusterização.")
        fm["CLUSTER"] = 0
        return fm

    base = fm.groupby("CODMUN6")[feats].mean().reset_index().dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(base[feats])

    # Elbow + silhouette (k de 2 a 8)
    elbow_rows = []
    for ki in range(2, 9):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if ki > 1 else 0.0
        elbow_rows.append({"k": ki, "inertia": km.inertia_, "silhouette": round(sil, 4)})
    pd.DataFrame(elbow_rows).to_csv(MODELS_DIR / "elbow_silhouette.csv", index=False)

    km4 = KMeans(n_clusters=k, random_state=42, n_init=10)
    base["CLUSTER"] = km4.fit_predict(X)

    # Reordena clusters por IDH crescente (0 = mais vulnerável)
    if "IDHM" in feats:
        ordem = base.groupby("CLUSTER")["IDHM"].mean().sort_values().index.tolist()
        mapa  = {c: i for i, c in enumerate(ordem)}
        base["CLUSTER"] = base["CLUSTER"].map(mapa)

    # Municípios sem todas as features (ex.: código ignorado "xx0000", municípios
    # criados após 2010 sem dados do Atlas, sem RREO) ficam fora — antes eram
    # atribuídos ao Cluster 0 por fillna(0), inflando o cluster mais vulnerável.
    fm = fm.merge(base[["CODMUN6", "CLUSTER"]], on="CODMUN6", how="inner")
    fm["CLUSTER"] = fm["CLUSTER"].astype(int)

    print(f"  [Cluster] Distribuição: {fm['CLUSTER'].value_counts().sort_index().to_dict()}")
    return fm


def gerar_perfil_clusters(fm: pd.DataFrame) -> None:
    """Salva CSV com perfil socioeconômico médio por cluster."""
    print("  [Cluster] Gerando perfil dos clusters...")

    cols_perfil = ["TAXA_EVITAVEL_100K", "TAXA_PADRONIZADA_100K", "IDHM", "GINI", "T_AGUA", "ESPVIDA",
                   "T_ANALF15M", "LOG_PIB_PER_CAPITA", "LOG_DESPESA_SAUDE_PC"]
    cols_ok = _cols_disponiveis(fm, cols_perfil)

    perfil = fm.groupby("CLUSTER")[cols_ok].mean().reset_index()
    perfil.columns = [c if c == "CLUSTER" else c for c in perfil.columns]

    # Renomeia para compatibilidade com o dashboard
    rename = {
        "LOG_PIB_PER_CAPITA":   "PIB_pc",
        "LOG_DESPESA_SAUDE_PC": "desp_saude_pc",
        "T_ANALF15M":           "ANALF15M",
        "TAXA_EVITAVEL_100K":   "taxa_bruta",
        "TAXA_PADRONIZADA_100K": "taxa_padronizada",   # padronização direta (etl.py)
    }
    perfil = perfil.rename(columns=rename)

    # Número de municípios por cluster
    n_mun = fm.groupby("CLUSTER")["CODMUN6"].nunique().reset_index(name="n_municipios")
    perfil = perfil.merge(n_mun, on="CLUSTER")

    # % população Brasil
    if "POPULACAO" in fm.columns:
        pop_total = fm.drop_duplicates("CODMUN6")["POPULACAO"].sum()
        pop_cl = fm.drop_duplicates("CODMUN6").groupby("CLUSTER")["POPULACAO"].sum().reset_index()
        pop_cl["%_pop_brasil"] = (pop_cl["POPULACAO"] / pop_total * 100).round(1)
        perfil = perfil.merge(pop_cl[["CLUSTER", "%_pop_brasil"]], on="CLUSTER")
    else:
        perfil["%_pop_brasil"] = 0.0

    perfil.to_csv(MODELS_DIR / "perfil_clusters.csv", index=False)
    print(f"  [Cluster] perfil_clusters.csv salvo.")

    # Composição LBCE por cluster
    cols_lbce = ["imunoprevencao", "doencas_infecciosas", "doencas_nao_transmissiveis", "morte_materna"]
    cols_lbce_ok = _cols_disponiveis(fm, cols_lbce)
    if cols_lbce_ok:
        lbce = fm.groupby("CLUSTER")[cols_lbce_ok].sum().reset_index()
        total = lbce[cols_lbce_ok].sum(axis=1)
        for c in cols_lbce_ok:
            lbce[c] = (lbce[c] / total * 100).round(2)
        lbce.to_csv(MODELS_DIR / "lbce_por_cluster.csv", index=False)

    # Composição regional por cluster
    if "REGIAO" in fm.columns:
        comp = fm.drop_duplicates("CODMUN6").groupby(["CLUSTER", "REGIAO"]).size().unstack(fill_value=0)
        comp = (comp.div(comp.sum(axis=1), axis=0) * 100).round(1).reset_index()
        for r in ["N","NE","CO","SE","S"]:
            if r not in comp.columns:
                comp[r] = 0.0
        comp.to_csv(MODELS_DIR / "composicao_regional_cluster.csv", index=False)

    # Tendência temporal
    tend = fm.groupby("ANO").agg(
        n_municipios=("CODMUN6", "nunique"),
        obitos_total=("OBITOS_EVITAVEIS", "sum"),
        taxa_media=("TAXA_PADRONIZADA_100K", "mean"),     # dashboard rotula como padronizada
        taxa_bruta_media=("TAXA_EVITAVEL_100K", "mean"),
    ).reset_index()
    tend.to_csv(MODELS_DIR / "tendencia_temporal.csv", index=False)

    # Taxa por região
    if "REGIAO" in fm.columns:
        reg = fm.groupby("REGIAO").agg(
            n_municipios=("CODMUN6", "nunique"),
            taxa_bruta=("TAXA_EVITAVEL_100K", "mean"),
            taxa_padronizada=("TAXA_PADRONIZADA_100K", "mean"),
        ).reset_index().round(1)
        reg.to_csv(MODELS_DIR / "taxa_regiao_comparacao.csv", index=False)


# ─── Mineração de Causas ───────────────────────────────────────────────────────
def mineracao_causas(fm: pd.DataFrame) -> None:
    """Top CID-10 por cluster e lift (frequência no cluster ÷ frequência nacional)."""
    print("  [Miner] Analisando causas de óbito por cluster...")

    top_cid_path = DATA_PROC / "top_cid_municipio.parquet"
    if not top_cid_path.exists():
        print("  [Miner] top_cid_municipio.parquet não encontrado, pulando mineração.")
        return

    cid_mun = pd.read_parquet(top_cid_path)
    # Join cluster
    cluster_map = fm.drop_duplicates("CODMUN6")[["CODMUN6", "CLUSTER"]]
    cid = cid_mun.merge(cluster_map, on="CODMUN6", how="left").dropna(subset=["CLUSTER"])
    cid["CLUSTER"] = cid["CLUSTER"].astype(int)

    total_global = cid["n_cid"].sum()
    freq_global = cid.groupby("CID_CAT3")["n_cid"].sum() / total_global

    rows_top, rows_lift = [], []
    for cl in sorted(cid["CLUSTER"].unique()):
        sub  = cid[cid["CLUSTER"] == cl]
        tot  = sub["n_cid"].sum()
        freq = sub.groupby("CID_CAT3")["n_cid"].sum().sort_values(ascending=False)

        # Top causas (% no cluster)
        for cid_code, n in freq.head(15).items():
            rows_top.append({"CLUSTER": cl, "CID_CAT3": cid_code,
                             "n": int(n), "pct": round(n / tot * 100, 2)})

        # Lift (sobre-representação vs. média nacional)
        for cid_code, n in freq.items():
            if n < LIFT_MIN_OBITOS:   # causas raras geram lift instável
                continue
            pct_cl = n / tot
            pct_gl = freq_global.get(cid_code, 1e-9)
            rows_lift.append({"CLUSTER": cl, "CID_CAT3": cid_code,
                               "n": int(n), "lift": round(pct_cl / pct_gl, 3)})

    pd.DataFrame(rows_top).to_csv(MODELS_DIR / "top_causas_cluster.csv", index=False)
    lift_df = pd.DataFrame(rows_lift).sort_values(["CLUSTER", "lift"], ascending=[True, False])
    lift_df.to_csv(MODELS_DIR / "lift_causas_cluster.csv", index=False)
    print("  [Miner] top_causas_cluster.csv e lift_causas_cluster.csv salvos.")

    # Termos distintivos (abreviado)
    lift_df.groupby("CLUSTER").head(5)[["CLUSTER","CID_CAT3","lift"]].to_csv(
        MODELS_DIR / "termos_distintivos_cluster.csv", index=False
    )


# ─── Regressão ────────────────────────────────────────────────────────────────
def regressao(fm: pd.DataFrame) -> None:
    """OLS painel (cluster-robust por município) + RF + XGBoost."""
    print("  [Reg] Treinando modelos de regressão...")

    target = TARGET
    feats  = _cols_disponiveis(fm, FEATURES_REG)
    if not feats or target not in fm.columns:
        print("  [Reg] Dados insuficientes para regressão.")
        return

    df_reg = fm[feats + [target, "CODMUN6", "ANO", "REGIAO"]].dropna()
    if len(df_reg) < 100:
        print(f"  [Reg] Apenas {len(df_reg)} linhas disponíveis — pulando regressão.")
        return

    # Dummies de região e ano
    reg_dummies = pd.get_dummies(df_reg["REGIAO"], prefix="REG", drop_first=True)
    ano_dummies = pd.get_dummies(df_reg["ANO"].astype(str), prefix="ANO", drop_first=True)
    X_full = pd.concat([df_reg[feats], reg_dummies, ano_dummies], axis=1).astype(float)
    y      = df_reg[target].astype(float)
    grupos = df_reg["CODMUN6"]

    # Train/test split (GroupShuffleSplit — municípios inteiros em cada split)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X_full, y, grupos))
    X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    g_tr = grupos.iloc[tr_idx]

    # ── OLS cluster-robust ──────────────────────────────────────────────────
    try:
        ols = sm.OLS(y_tr, sm.add_constant(X_tr)).fit(
            cov_type="cluster", cov_kwds={"groups": g_tr}
        )
        pred_ols = ols.predict(sm.add_constant(X_te, has_constant="add"))
        r2_ols   = float(1 - ((y_te - pred_ols)**2).sum() / ((y_te - y_te.mean())**2).sum())
        rmse_ols = float(np.sqrt(((y_te - pred_ols)**2).mean()))
        mae_ols  = float((y_te - pred_ols).abs().mean())

        # Coeficientes
        def sig(p):
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return ""

        coefs = pd.DataFrame({
            "feature": ols.params.index,
            "coef":    ols.params.values.round(4),
            "std_err": ols.bse.values.round(4),
            "t":       ols.tvalues.values.round(4),
            "p_value": ols.pvalues.values,
            "ci_low":  ols.conf_int()[0].values.round(4),
            "ci_high": ols.conf_int()[1].values.round(4),
        })
        coefs["sig"] = coefs["p_value"].apply(sig)
        coefs.to_csv(MODELS_DIR / "coeficientes_ols.csv", index=False)
    except Exception as e:
        print(f"  [Reg] OLS falhou: {e}")
        r2_ols, rmse_ols, mae_ols = 0.0, 0.0, 0.0

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    r2_rf   = float(1 - ((y_te - pred_rf)**2).sum() / ((y_te - y_te.mean())**2).sum())
    rmse_rf = float(np.sqrt(((y_te - pred_rf)**2).mean()))
    mae_rf  = float((y_te - pred_rf).abs().mean())

    # ── XGBoost ─────────────────────────────────────────────────────────────
    r2_xgb = rmse_xgb = mae_xgb = 0.0
    if _XGBOOST:
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           early_stopping_rounds=30, random_state=42,
                           verbosity=0)
        xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred_xgb = xgb.predict(X_te)
        r2_xgb   = float(1 - ((y_te - pred_xgb)**2).sum() / ((y_te - y_te.mean())**2).sum())
        rmse_xgb = float(np.sqrt(((y_te - pred_xgb)**2).mean()))
        mae_xgb  = float((y_te - pred_xgb).abs().mean())

    # ── Salva métricas ───────────────────────────────────────────────────────
    metricas = pd.DataFrame([
        {"modelo": "OLS",           "R2": round(r2_ols,3),  "RMSE": round(rmse_ols,1),  "MAE": round(mae_ols,1)},
        {"modelo": "Random Forest", "R2": round(r2_rf,3),   "RMSE": round(rmse_rf,1),   "MAE": round(mae_rf,1)},
        {"modelo": "XGBoost",       "R2": round(r2_xgb,3),  "RMSE": round(rmse_xgb,1),  "MAE": round(mae_xgb,1)},
    ])
    metricas.to_csv(MODELS_DIR / "metricas_modelos.csv", index=False)

    # ── Importância de features ──────────────────────────────────────────────
    imp = pd.DataFrame({"feature": X_full.columns, "imp_rf": rf.feature_importances_})
    if _XGBOOST:
        imp["imp_xgb"] = xgb.feature_importances_
    else:
        imp["imp_xgb"] = 0.0
    imp.to_csv(MODELS_DIR / "importancia_features.csv", index=False)

    print(f"  [Reg] OLS R²={r2_ols:.3f} | RF R²={r2_rf:.3f} | XGB R²={r2_xgb:.3f}")


# ─── Ponto de entrada ──────────────────────────────────────────────────────────
def executar_analise() -> None:
    """Executa clusterização, regressão e mineração de causas."""
    print("\n── Análise ──────────────────────────────────────────────────")

    fm = _carregar_fm()
    fm = clusterizar(fm)
    gerar_perfil_clusters(fm)
    mineracao_causas(fm)
    regressao(fm)

    print("── Análise concluída ────────────────────────────────────────\n")
