"""
ETL — Extração, Transformação e Carga
TCC | Mortalidade por Causas Evitáveis no Brasil | IESB 2026/01

Fontes:
  - SIM/DATASUS via PySUS (óbitos com CID-10)
  - IBGE Sidra API   (população municipal, PIB)
  - SICONFI API      (despesa municipal em saúde)
  - IPEA Data        (IDH-M, Gini, renda, saneamento)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ─── Diretórios ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for d in (DATA_RAW, DATA_PROC, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

ANOS = [2022, 2023, 2024]
UFS  = [
    "AC","AL","AM","AP","BA","CE","DF","ES","GO",
    "MA","MG","MS","MT","PA","PB","PE","PI","PR",
    "RJ","RN","RO","RR","RS","SC","SE","SP","TO",
]

# ─── Lista Brasileira de Causas Evitáveis (Malta et al. 2007/2010) ────────────
# 3 chars de prefixo CID-10 → grupo LBCE (faixa 5–74 anos; causas externas excluídas)
_LBCE_RAW = {
    # Imunoprevenção
    "A33":"imunoprevencao","A35":"imunoprevencao","A36":"imunoprevencao",
    "A37":"imunoprevencao","A80":"imunoprevencao","B05":"imunoprevencao",
    "B06":"imunoprevencao","B16":"imunoprevencao","B26":"imunoprevencao",
    "A82":"imunoprevencao","B19":"imunoprevencao",
    # Doenças infecciosas
    "A00":"doencas_infecciosas","A01":"doencas_infecciosas","A02":"doencas_infecciosas",
    "A03":"doencas_infecciosas","A04":"doencas_infecciosas","A06":"doencas_infecciosas",
    "A07":"doencas_infecciosas","A08":"doencas_infecciosas","A09":"doencas_infecciosas",
    "A15":"doencas_infecciosas","A16":"doencas_infecciosas","A17":"doencas_infecciosas",
    "A18":"doencas_infecciosas","A19":"doencas_infecciosas","A39":"doencas_infecciosas",
    "A41":"doencas_infecciosas","A90":"doencas_infecciosas","A91":"doencas_infecciosas",
    "B20":"doencas_infecciosas","B21":"doencas_infecciosas","B22":"doencas_infecciosas",
    "B23":"doencas_infecciosas","B24":"doencas_infecciosas","B50":"doencas_infecciosas",
    "B51":"doencas_infecciosas","B52":"doencas_infecciosas","B53":"doencas_infecciosas",
    "B54":"doencas_infecciosas","B57":"doencas_infecciosas","B65":"doencas_infecciosas",
    "B18":"doencas_infecciosas","J00":"doencas_infecciosas","J01":"doencas_infecciosas",
    "J02":"doencas_infecciosas","J03":"doencas_infecciosas","J06":"doencas_infecciosas",
    "J18":"doencas_infecciosas","J20":"doencas_infecciosas","J22":"doencas_infecciosas",
    "G00":"doencas_infecciosas","G03":"doencas_infecciosas",
    # Doenças não transmissíveis
    "I10":"doencas_nao_transmissiveis","I11":"doencas_nao_transmissiveis",
    "I20":"doencas_nao_transmissiveis","I21":"doencas_nao_transmissiveis",
    "I22":"doencas_nao_transmissiveis","I23":"doencas_nao_transmissiveis",
    "I24":"doencas_nao_transmissiveis","I25":"doencas_nao_transmissiveis",
    "I50":"doencas_nao_transmissiveis","I60":"doencas_nao_transmissiveis",
    "I61":"doencas_nao_transmissiveis","I62":"doencas_nao_transmissiveis",
    "I63":"doencas_nao_transmissiveis","I64":"doencas_nao_transmissiveis",
    "I65":"doencas_nao_transmissiveis","I66":"doencas_nao_transmissiveis",
    "I67":"doencas_nao_transmissiveis","I69":"doencas_nao_transmissiveis",
    "E10":"doencas_nao_transmissiveis","E11":"doencas_nao_transmissiveis",
    "E12":"doencas_nao_transmissiveis","E13":"doencas_nao_transmissiveis",
    "E14":"doencas_nao_transmissiveis","E40":"doencas_nao_transmissiveis",
    "E41":"doencas_nao_transmissiveis","E43":"doencas_nao_transmissiveis",
    "E46":"doencas_nao_transmissiveis","C00":"doencas_nao_transmissiveis",
    "C01":"doencas_nao_transmissiveis","C02":"doencas_nao_transmissiveis",
    "C15":"doencas_nao_transmissiveis","C16":"doencas_nao_transmissiveis",
    "C18":"doencas_nao_transmissiveis","C19":"doencas_nao_transmissiveis",
    "C20":"doencas_nao_transmissiveis","C22":"doencas_nao_transmissiveis",
    "C25":"doencas_nao_transmissiveis","C33":"doencas_nao_transmissiveis",
    "C34":"doencas_nao_transmissiveis","C44":"doencas_nao_transmissiveis",
    "C50":"doencas_nao_transmissiveis","C53":"doencas_nao_transmissiveis",
    "C54":"doencas_nao_transmissiveis","C61":"doencas_nao_transmissiveis",
    "C67":"doencas_nao_transmissiveis","F10":"doencas_nao_transmissiveis",
    "J40":"doencas_nao_transmissiveis","J41":"doencas_nao_transmissiveis",
    "J42":"doencas_nao_transmissiveis","J43":"doencas_nao_transmissiveis",
    "J44":"doencas_nao_transmissiveis","J45":"doencas_nao_transmissiveis",
    "J46":"doencas_nao_transmissiveis","K70":"doencas_nao_transmissiveis",
    "K74":"doencas_nao_transmissiveis","N18":"doencas_nao_transmissiveis",
    "G40":"doencas_nao_transmissiveis","G41":"doencas_nao_transmissiveis",
    # Morte materna
    **{f"O{n:02d}":"morte_materna" for n in list(range(0,10))+list(range(10,20))+
       list(range(20,30))+list(range(30,50))+list(range(60,80))+list(range(85,100))},
}

def classificar_cid(cid: str) -> Optional[str]:
    """Classifica um código CID-10 segundo a LBCE (prefixo de 3 chars)."""
    c = str(cid).strip().upper().replace(".", "")
    return _LBCE_RAW.get(c[:3])


# ─── EXTRACT — SIM via PySUS ───────────────────────────────────────────────────
def _baixar_uf_ano_pysus(uf: str, ano: int, cache_dir: Path) -> pd.DataFrame:
    """Tenta baixar microdados SIM de uma UF/ano via PySUS (com cache local)."""
    cache = cache_dir / f"sim_{uf}_{ano}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    df = None

    # Tentativa 1 — API moderna pysus >= 0.4
    try:
        from pysus.ftp.databases.sim import SIM as SIMdb  # type: ignore
        sim = SIMdb()
        files = sim.get_files(group="DORES", year=ano, state=uf)
        if files:
            item = files[0] if isinstance(files, (list, tuple)) else files
            content = sim.download(item)
            df = content.to_dataframe() if hasattr(content, "to_dataframe") else content
    except Exception:
        pass

    # Tentativa 2 — API legada pysus < 0.4
    if df is None:
        try:
            from pysus.online_data import SIM as SIMold  # type: ignore
            result = SIMold.download(uf, ano)
            df = result.to_dataframe() if hasattr(result, "to_dataframe") else result
        except Exception:
            pass

    if df is not None and not df.empty:
        df.to_parquet(cache)

    return df if df is not None else pd.DataFrame()


def extrair_sim() -> pd.DataFrame:
    """Extrai todos os microdados SIM (2022–2024) via PySUS com cache."""
    print("  [SIM] Baixando microdados via PySUS...")
    cache_dir = DATA_RAW / "sim"
    cache_dir.mkdir(exist_ok=True)

    parquet_final = DATA_PROC / "sim_evitaveis.parquet"
    if parquet_final.exists():
        print("  [SIM] Cache final encontrado, carregando...")
        return pd.read_parquet(parquet_final)

    tarefas = [(uf, ano) for uf in UFS for ano in ANOS]
    dfs = []
    total = len(tarefas)
    for i, (uf, ano) in enumerate(tarefas, 1):
        print(f"  [SIM] {uf}/{ano} ({i}/{total})...", end="\r")
        df = _baixar_uf_ano_pysus(uf, ano, cache_dir)
        if not df.empty:
            df["_UF"] = uf
            df["_ANO"] = ano
            dfs.append(df)

    if not dfs:
        print("\n  [SIM] AVISO: Nenhum dado baixado. Verifique a instalação do pysus.")
        return pd.DataFrame()

    sim = pd.concat(dfs, ignore_index=True)
    sim.to_parquet(parquet_final)
    print(f"\n  [SIM] {len(sim):,} registros brutos salvos.")
    return sim


# ─── TRANSFORM — Limpeza e LBCE ───────────────────────────────────────────────
def _decode_idade(valor) -> float:
    """Decodifica campo IDADE do SIM: formato XYY (X=unidade, YY=valor)."""
    try:
        s = str(int(valor)).zfill(3)
        unidade, vv = s[0], int(s[1:])
        if unidade == "3":
            return vv / 12        # meses → anos
        if unidade == "4":
            return float(vv)      # 0–99 anos
        if unidade == "5":
            return float(vv + 100)
    except Exception:
        pass
    return np.nan


def transformar_sim(sim: pd.DataFrame) -> pd.DataFrame:
    """Aplica LBCE, filtra faixa 5–74 anos e agrega por município × ano."""
    if sim.empty:
        return pd.DataFrame()

    print("  [Transform] Decodificando idades...")
    sim["IDADE_ANOS"] = sim["IDADE"].apply(_decode_idade)

    # Faixa etária da LBCE
    sim5_74 = sim[(sim["IDADE_ANOS"] >= 5) & (sim["IDADE_ANOS"] < 75)].copy()

    print("  [Transform] Classificando CIDs pela LBCE...")
    sim5_74["GRUPO_LBCE"] = sim5_74["CAUSABAS"].apply(classificar_cid)
    evitaveis = sim5_74[sim5_74["GRUPO_LBCE"].notna()].copy()

    # Código municipal: SIM usa 6 dígitos
    evitaveis["CODMUN6"] = evitaveis["CODMUNRES"].astype(str).str.zfill(7).str[:6]
    evitaveis["CID_CAT3"] = evitaveis["CAUSABAS"].astype(str).str[:3].str.upper()

    # Agregação por município × ano
    agg = (
        evitaveis.groupby(["CODMUN6", "_ANO"])
        .agg(
            OBITOS_EVITAVEIS=("GRUPO_LBCE", "count"),
            imunoprevencao=("GRUPO_LBCE", lambda x: (x == "imunoprevencao").sum()),
            doencas_infecciosas=("GRUPO_LBCE", lambda x: (x == "doencas_infecciosas").sum()),
            doencas_nao_transmissiveis=("GRUPO_LBCE", lambda x: (x == "doencas_nao_transmissiveis").sum()),
            morte_materna=("GRUPO_LBCE", lambda x: (x == "morte_materna").sum()),
        )
        .reset_index()
        .rename(columns={"_ANO": "ANO"})
    )

    # Top causas individuais por município × ano
    top_cid = (
        evitaveis.groupby(["CODMUN6", "_ANO", "CID_CAT3"])
        .size()
        .reset_index(name="n_cid")
        .rename(columns={"_ANO": "ANO"})
    )
    top_cid.to_parquet(DATA_PROC / "top_cid_municipio.parquet")

    print(f"  [Transform] {len(evitaveis):,} óbitos evitáveis → {len(agg):,} municípios-ano.")
    return agg


# ─── EXTRACT — IBGE Sidra ─────────────────────────────────────────────────────
def _sidra_get(url: str, campo_valor: str, nome_col: str) -> pd.DataFrame:
    """Consulta genérica à API do IBGE Sidra."""
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        data = r.json()
        rows = []
        for item in data[0]["resultados"][0]["series"]:
            cod = item["localidade"]["id"]
            val = item["serie"].get(list(item["serie"].keys())[0])
            rows.append({"CODMUN7": str(cod), nome_col: float(val) if val else np.nan})
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  [IBGE Sidra] Erro em {nome_col}: {e}")
        return pd.DataFrame()


def extrair_populacao() -> pd.DataFrame:
    print("  [IBGE] Baixando população municipal (Censo 2022)...")
    url = ("https://servicodados.ibge.gov.br/api/v3/agregados/4714"
           "/periodos/2022/variaveis/93?localidades=N6[all]")
    return _sidra_get(url, "2022", "POPULACAO")


def extrair_pib() -> pd.DataFrame:
    print("  [IBGE] Baixando PIB municipal (2021)...")
    url = ("https://servicodados.ibge.gov.br/api/v3/agregados/5938"
           "/periodos/2021/variaveis/37?localidades=N6[all]")
    return _sidra_get(url, "2021", "PIB_RS_MIL")


# ─── EXTRACT — SICONFI (despesa em saúde) ─────────────────────────────────────
def extrair_despesa_saude() -> pd.DataFrame:
    """Baixa despesa municipal em saúde via SICONFI (função orçamentária 10)."""
    print("  [SICONFI] Baixando despesa em saúde...")
    cache = DATA_PROC / "despesa_saude.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    rows = []
    for ano in ANOS:
        url = (
            f"https://apidatalake.tesouro.gov.br/ords/siconfi/tt/rreo"
            f"?an_exercicio={ano}&in_periodicidade=A&nr_periodo=6"
            f"&co_tipo_demonstrativo=RREO&no_anexo=RREO-Anexo%2002"
            f"&co_esfera=M&co_poder=E&co_funcao=10"
        )
        try:
            r = requests.get(url, timeout=120)
            data = r.json()
            items = data.get("items", data) if isinstance(data, dict) else data
            for item in (items if isinstance(items, list) else []):
                cod = str(item.get("co_municipio_ibge", "")).zfill(7)[:6]
                val = item.get("vl_despesa_liquidada") or item.get("vl_despesa_empenhada")
                if cod and val:
                    rows.append({"CODMUN6": cod, "ANO": ano, "DESPESA_SAUDE_RS": float(val)})
        except Exception as e:
            print(f"  [SICONFI] Erro ano {ano}: {e}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["CODMUN6","ANO","DESPESA_SAUDE_RS"])
    df.to_parquet(cache)
    return df


# ─── EXTRACT — IPEA Data (IDH, Gini, renda, saneamento) ──────────────────────
_IPEA_SERIES = {
    "IDHM":      "ADH_IDHM",
    "GINI":      "ADH_GINI",
    "RDPC":      "ADH_RDPC",
    "PMPOB":     "ADH_PMPOB",
    "ESPVIDA":   "ADH_ESPVIDA",
    "T_ANALF15M":"ADH_T_ANALF15M",
    "T_AGUA":    "ADH_T_AGUA",
    "IDHM_E":   "ADH_IDHM_E",
    "IDHM_L":   "ADH_IDHM_L",
    "IDHM_R":   "ADH_IDHM_R",
}


def extrair_ipea() -> pd.DataFrame:
    """Baixa indicadores do Atlas DH (Censo 2010) via ipeadatapy."""
    print("  [IPEA] Baixando indicadores Atlas DH 2010...")
    cache = DATA_PROC / "ipea_atlas.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    dfs = []
    try:
        import ipeadatapy as ipea  # type: ignore

        for col, series_id in _IPEA_SERIES.items():
            try:
                df = ipea.timeseries(series_id)
                # Normaliza colunas (podem variar por versão do ipeadatapy)
                df.columns = [c.strip().upper() for c in df.columns]
                # Filtra ano 2010
                for date_col in ("YEAR", "DATE", "RAW DATE"):
                    if date_col in df.columns:
                        df = df[df[date_col].astype(str).str.startswith("2010")]
                        break
                # Pega código do município e valor
                for code_col in ("CODE", "TERRITORY CODE", "IBGE"):
                    if code_col in df.columns:
                        df = df.rename(columns={code_col: "CODMUN7"})
                        break
                for val_col in ("VALUE", "VALOR"):
                    if val_col in df.columns:
                        df = df.rename(columns={val_col: col})
                        break
                if "CODMUN7" in df.columns and col in df.columns:
                    dfs.append(df[["CODMUN7", col]])
                    print(f"    {series_id} ({col}): {len(df)} municípios")
            except Exception as e:
                print(f"    Aviso {col}: {e}")
    except ImportError:
        print("  [IPEA] ipeadatapy não instalado — tente: pip install ipeadatapy")

    if not dfs:
        df_vazio = pd.DataFrame(columns=["CODMUN7"] + list(_IPEA_SERIES.keys()))
        df_vazio.to_parquet(cache)
        return df_vazio

    df_final = dfs[0]
    for d in dfs[1:]:
        df_final = df_final.merge(d, on="CODMUN7", how="outer")

    df_final["CODMUN7"] = df_final["CODMUN7"].astype(str).str.zfill(7)
    df_final["CODMUN6"] = df_final["CODMUN7"].str[:6]
    df_final.to_parquet(cache)
    return df_final


# ─── LOAD — Feature Matrix ─────────────────────────────────────────────────────
def construir_feature_matrix(
    obitos: pd.DataFrame,
    pop: pd.DataFrame,
    pib: pd.DataFrame,
    desp: pd.DataFrame,
    ipea: pd.DataFrame,
) -> pd.DataFrame:
    """Faz join de todas as fontes e engenharia de features."""
    print("  [Load] Construindo feature matrix...")

    if obitos.empty:
        print("  [Load] AVISO: sem dados de óbitos — impossível construir feature matrix.")
        return pd.DataFrame()

    # Normaliza códigos municipais
    for df in (pop, pib, ipea):
        if "CODMUN7" in df.columns:
            df["CODMUN6"] = df["CODMUN7"].astype(str).str.zfill(7).str[:6]

    fm = obitos.copy()

    # Join população (repetida para cada ano)
    if not pop.empty:
        fm = fm.merge(pop[["CODMUN6", "POPULACAO"]], on="CODMUN6", how="left")

    # Join PIB
    if not pib.empty:
        fm = fm.merge(pib[["CODMUN6", "PIB_RS_MIL"]], on="CODMUN6", how="left")

    # Join despesa saúde (ano a ano)
    if not desp.empty:
        fm = fm.merge(desp, on=["CODMUN6", "ANO"], how="left")

    # Join IPEA
    if not ipea.empty:
        cols_ipea = ["CODMUN6"] + [c for c in _IPEA_SERIES if c in ipea.columns]
        fm = fm.merge(ipea[cols_ipea], on="CODMUN6", how="left")

    # Região (primeiros 2 dígitos do código de UF → macrorregião)
    uf_para_regiao = {
        "11":"N","12":"N","13":"N","14":"N","15":"N","16":"N","17":"N",
        "21":"NE","22":"NE","23":"NE","24":"NE","25":"NE","26":"NE",
        "27":"NE","28":"NE","29":"NE",
        "31":"SE","32":"SE","33":"SE","35":"SE",
        "41":"S","42":"S","43":"S",
        "50":"CO","51":"CO","52":"CO","53":"CO",
    }
    fm["UF_COD"] = fm["CODMUN6"].str[:2]
    fm["REGIAO"] = fm["UF_COD"].map(uf_para_regiao).fillna("CO")

    # Feature engineering
    fm["PIB_PER_CAPITA"]    = (fm["PIB_RS_MIL"] * 1_000) / fm["POPULACAO"].replace(0, np.nan)
    fm["DESPESA_SAUDE_PC"]  = fm.get("DESPESA_SAUDE_RS", pd.Series(np.nan, index=fm.index)) / fm["POPULACAO"].replace(0, np.nan)
    fm["TAXA_EVITAVEL_100K"] = (fm["OBITOS_EVITAVEIS"] / fm["POPULACAO"].replace(0, np.nan)) * 100_000

    # Log-transforms
    for c in ("PIB_PER_CAPITA", "DESPESA_SAUDE_PC", "POPULACAO"):
        if c in fm.columns:
            fm[f"LOG_{c}"] = np.log1p(fm[c].clip(lower=0))

    fm.to_parquet(DATA_PROC / "feature_matrix.parquet")
    print(f"  [Load] Feature matrix: {len(fm):,} linhas × {len(fm.columns)} colunas.")
    return fm


# ─── Ponto de entrada ──────────────────────────────────────────────────────────
def executar_etl() -> pd.DataFrame:
    """Executa o pipeline ETL completo."""
    print("\n── ETL ──────────────────────────────────────────────────────")

    sim_raw   = extrair_sim()
    obitos    = transformar_sim(sim_raw)
    pop       = extrair_populacao()
    pib       = extrair_pib()
    desp      = extrair_despesa_saude()
    df_ipea   = extrair_ipea()
    fm        = construir_feature_matrix(obitos, pop, pib, desp, df_ipea)

    print("── ETL concluído ────────────────────────────────────────────\n")
    return fm
