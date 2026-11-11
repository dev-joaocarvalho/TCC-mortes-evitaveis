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


# ─── EXTRACT — SIM via FTP do DATASUS ─────────────────────────────────────────
# Obs.: a versão anterior usava PySUS, mas a API do pacote mudou (pysus 2.x) e a
# versão antiga falha no Windows. Os mesmos arquivos DO{UF}{ANO}.dbc são baixados
# diretamente do FTP público do DATASUS.
SIM_FTP_HOST = "ftp.datasus.gov.br"
SIM_FTP_DIR  = "/dissemin/publicos/SIM/CID10/DORES"
SIM_COLUNAS  = ["CAUSABAS", "IDADE", "CODMUNRES"]


def _ler_dbf_colunas(path: Path, colunas: list[str]) -> pd.DataFrame:
    """Leitor DBF (dBase III) vetorizado: lê só as colunas pedidas."""
    raw = path.read_bytes()
    n_rec  = int.from_bytes(raw[4:8], "little")
    h_len  = int.from_bytes(raw[8:10], "little")
    r_len  = int.from_bytes(raw[10:12], "little")
    campos, pos, off = {}, 32, 1          # 1º byte do registro = flag de exclusão
    while raw[pos] != 0x0D:
        nome = raw[pos:pos+11].split(b"\x00")[0].decode("latin-1").strip()
        tam  = raw[pos+16]
        campos[nome] = (off, tam)
        off += tam
        pos += 32
    dados = np.frombuffer(raw, dtype=np.uint8, count=n_rec * r_len, offset=h_len).reshape(n_rec, r_len)
    ativos = dados[:, 0] != ord("*")
    out = {}
    for c in colunas:
        o, t = campos[c]
        col = dados[ativos, o:o+t].copy().view(f"S{t}").ravel()
        out[c] = pd.Series(col).str.decode("latin-1").str.strip()
    return pd.DataFrame(out)


def _baixar_uf_ano_ftp(uf: str, ano: int, cache_dir: Path) -> pd.DataFrame:
    """Baixa DO{UF}{ANO}.dbc do FTP do DATASUS, descompacta e lê (com cache)."""
    cache = cache_dir / f"sim_{uf}_{ano}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    import ftplib
    import datasus_dbc  # descompactador .dbc → .dbf

    nome = f"DO{uf}{ano}.dbc"
    dbc  = cache_dir / nome
    dbf  = cache_dir / nome.replace(".dbc", ".dbf")
    for tentativa in range(4):
        try:
            with ftplib.FTP(SIM_FTP_HOST, timeout=120) as ftp:
                ftp.login()
                ftp.cwd(SIM_FTP_DIR)
                with open(dbc, "wb") as fh:
                    ftp.retrbinary(f"RETR {nome}", fh.write)
            break
        except Exception as e:
            print(f"\n  [SIM] {nome}: tentativa {tentativa+1} falhou ({e})")
            time.sleep(10 * (tentativa + 1))
    else:
        raise RuntimeError(f"Não foi possível baixar {nome} do FTP do DATASUS")

    datasus_dbc.decompress(str(dbc), str(dbf))
    df = _ler_dbf_colunas(dbf, SIM_COLUNAS)
    df.to_parquet(cache)
    dbc.unlink(missing_ok=True)
    dbf.unlink(missing_ok=True)
    return df


def extrair_sim() -> pd.DataFrame:
    """Extrai todos os microdados SIM (2022–2024) do FTP do DATASUS com cache."""
    print("  [SIM] Baixando microdados do FTP do DATASUS...")
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
        df = _baixar_uf_ano_ftp(uf, ano, cache_dir)
        if not df.empty:
            df["_UF"] = uf
            df["_ANO"] = ano
            dfs.append(df)

    if len(dfs) < total:
        raise RuntimeError(f"[SIM] Apenas {len(dfs)}/{total} arquivos UF/ano obtidos.")

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

    # Código municipal: o SIM já usa 6 dígitos (o zfill(7)[:6] anterior cortava o
    # último dígito e fundia municípios diferentes)
    evitaveis["CODMUN6"] = evitaveis["CODMUNRES"].astype(str).str.strip().str.zfill(6).str[:6]
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

    # Óbitos por faixa etária quinquenal (5–9 … 70–74) — base da padronização direta
    evitaveis["FAIXA"] = (evitaveis["IDADE_ANOS"] // 5 * 5).astype(int)
    por_faixa = (
        evitaveis.groupby(["CODMUN6", "_ANO", "FAIXA"])
        .size()
        .reset_index(name="OBITOS")
        .rename(columns={"_ANO": "ANO"})
    )
    por_faixa.to_parquet(DATA_PROC / "obitos_faixa_municipio.parquet")

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


def extrair_populacao_idade() -> pd.DataFrame:
    """População municipal por faixa quinquenal 5–74 anos (Censo 2022, Sidra 9514)."""
    print("  [IBGE] Baixando população por faixa etária (Censo 2022, tabela 9514)...")
    cache = DATA_PROC / "populacao_faixa.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    import re
    meta = requests.get("https://servicodados.ibge.gov.br/api/v3/agregados/9514/metadados",
                        timeout=90).json()
    idade = next(c for c in meta["classificacoes"] if c["id"] == 287)
    faixas = {}
    for cat in idade["categorias"]:
        m = re.fullmatch(r"(\d+) a (\d+) anos", cat["nome"])
        if m and int(m.group(2)) - int(m.group(1)) == 4 and 5 <= int(m.group(1)) <= 70:
            faixas[int(m.group(1))] = cat["id"]
    assert len(faixas) == 14, f"Esperadas 14 faixas (5–74), obtidas {sorted(faixas)}"

    dfs = []
    for ini, cat_id in sorted(faixas.items()):
        url = ("https://servicodados.ibge.gov.br/api/v3/agregados/9514/periodos/2022"
               f"/variaveis/93?localidades=N6[all]&classificacao=2[6794]|287[{cat_id}]|286[113635]")
        df = _sidra_get(url, "2022", "POP")
        if df.empty:
            raise RuntimeError(f"[IBGE] Falha na faixa {ini}–{ini+4}")
        df["FAIXA"] = ini
        dfs.append(df)
    pop = pd.concat(dfs, ignore_index=True)
    pop["CODMUN6"] = pop["CODMUN7"].str.zfill(7).str[:6]
    pop.to_parquet(cache)
    return pop


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

    # A API do SICONFI exige id_ente: sem ele a consulta retorna 0 itens.
    # Consulta-se o RREO Anexo 02 (despesa por função) do 6º bimestre de cada
    # município, filtrando a conta "Saúde" (função 10) e usando a despesa
    # liquidada até o bimestre (total anual).
    municipios = requests.get(
        "https://servicodados.ibge.gov.br/api/v1/localidades/municipios", timeout=120
    ).json()
    ids = [str(m["id"]) for m in municipios]
    raw_dir = DATA_RAW / "siconfi"
    raw_dir.mkdir(exist_ok=True)
    filtro = requests.utils.quote(json.dumps({"conta": "Saúde"}))

    def consultar(id_ente: str, ano: int) -> dict:
        # Municípios < 50 mil hab. podem publicar o "RREO Simplificado" (LRF, art. 63);
        # tenta o RREO completo e, se não houver a conta Saúde, o simplificado.
        houve_erro = False
        for tipo in ("RREO", "RREO%20Simplificado"):
            d = _consultar_tipo(id_ente, ano, tipo)
            if d["valor"] == "ERRO":
                houve_erro = True
            elif d["valor"] is not None:
                return d
        if houve_erro:                       # não conclui "sem dado" se alguma consulta falhou
            return {"id": id_ente, "ano": ano, "valor": "ERRO"}
        return {"id": id_ente, "ano": ano, "valor": None, "tipos_testados": 2}

    def _consultar_tipo(id_ente: str, ano: int, tipo: str) -> dict:
        url = ("https://apidatalake.tesouro.gov.br/ords/siconfi/tt/rreo"
               f"?an_exercicio={ano}&nr_periodo=6&co_tipo_demonstrativo={tipo}"
               f"&no_anexo=RREO-Anexo%2002&co_esfera=M&id_ente={id_ente}&q={filtro}")
        for tentativa in range(6):
            try:
                r = requests.get(url, timeout=90)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                itens = r.json().get("items", [])
                val = None
                for col in ("DESPESAS LIQUIDADAS ATÉ O BIMESTRE", "DESPESAS EMPENHADAS ATÉ O BIMESTRE"):
                    for it in itens:
                        if (it.get("cod_conta") == "RREO2TotalDespesas"
                                and str(it.get("coluna", "")).startswith(col)):
                            val = float(it["valor"])
                            break
                    if val is not None:
                        break
                return {"id": id_ente, "ano": ano, "valor": val}
            except Exception:
                time.sleep(3 * (tentativa + 1))
        return {"id": id_ente, "ano": ano, "valor": "ERRO"}

    rows, total_erros = [], 0
    for ano in ANOS:
        arq = raw_dir / f"rreo_saude_{ano}.jsonl"
        feitos = {}
        if arq.exists():
            for linha in arq.read_text(encoding="utf-8").splitlines():
                d = json.loads(linha)
                if d["valor"] not in ("ERRO", None) or d.get("tipos_testados") == 2:
                    feitos[d["id"]] = d
        pendentes = [i for i in ids if i not in feitos]
        print(f"  [SICONFI] {ano}: {len(feitos)} em cache, {len(pendentes)} a consultar...")
        with open(arq, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(consultar, i, ano) for i in pendentes]
            for k, fut in enumerate(as_completed(futs), 1):
                d = fut.result()
                fh.write(json.dumps(d) + "\n")
                fh.flush()
                if d["valor"] != "ERRO":
                    feitos[d["id"]] = d
                if k % 500 == 0:
                    print(f"  [SICONFI] {ano}: {k}/{len(pendentes)}", end="\r")
        erros = len(ids) - len(feitos)
        total_erros += erros
        if erros:
            print(f"\n  [SICONFI] AVISO {ano}: {erros} municípios com erro (rode de novo para retomar).")
        for d in feitos.values():
            if d["valor"] is not None:
                rows.append({"CODMUN6": d["id"][:6], "ANO": ano, "DESPESA_SAUDE_RS": d["valor"]})

    df = pd.DataFrame(rows, columns=["CODMUN6", "ANO", "DESPESA_SAUDE_RS"])
    print(f"  [SICONFI] {len(df):,} registros município-ano com despesa em saúde.")
    if total_erros == 0:          # só fixa o cache final se não houve falhas
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

    # O ipeadatapy atual não devolve mais o código do município (a coluna CODE é o
    # código da série), por isso a consulta vai direto à API OData do Ipeadata,
    # que traz TERCODIGO (código IBGE) e NIVNOME (nível territorial).
    dfs = []
    for col, series_id in _IPEA_SERIES.items():
        url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{series_id}')"
        for tentativa in range(4):
            try:
                vals = requests.get(url, timeout=180).json()["value"]
                break
            except Exception as e:
                print(f"    Aviso {series_id}: tentativa {tentativa+1} ({e})")
                time.sleep(10)
        else:
            raise RuntimeError(f"[IPEA] Falha ao baixar {series_id}")
        df = pd.DataFrame(vals)
        df = df[(df["NIVNOME"] == "Municípios") & df["VALDATA"].astype(str).str.startswith("2010")]
        df = df.rename(columns={"TERCODIGO": "CODMUN7", "VALVALOR": col})[["CODMUN7", col]]
        dfs.append(df)
        print(f"    {series_id} ({col}): {len(df)} municípios")

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
    pop_faixa: pd.DataFrame = pd.DataFrame(),
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

    # Taxa padronizada por idade — método direto.
    # População-padrão: Brasil, Censo 2022 (soma dos municípios por faixa).
    #   taxa_pad = Σ_f (óbitos_f / pop_f) · Ppad_f / Ppad_total · 100.000
    # Ppad_total é a população total do Brasil (todas as idades), de modo que a
    # taxa padronizada fica na mesma escala da taxa bruta (óbitos 5–74 / pop total).
    if not pop_faixa.empty:
        ob_f = pd.read_parquet(DATA_PROC / "obitos_faixa_municipio.parquet")
        ob_f = ob_f[ob_f["FAIXA"].between(5, 70)]
        pf = pop_faixa[["CODMUN6", "FAIXA", "POP"]]
        pad = pf.groupby("FAIXA")["POP"].sum()
        pop_br_total = pop.drop_duplicates("CODMUN6")["POPULACAO"].sum()
        peso = (pad / pop_br_total).rename("PESO").reset_index()

        # todas as combinações município × ano × faixa (faixas sem óbito contam 0)
        grade = fm[["CODMUN6", "ANO"]].merge(pf, on="CODMUN6", how="inner")
        grade = grade.merge(ob_f, on=["CODMUN6", "ANO", "FAIXA"], how="left").fillna({"OBITOS": 0})
        grade = grade.merge(peso, on="FAIXA")
        grade = grade[grade["POP"] > 0]
        grade["CONTRIB"] = grade["OBITOS"] / grade["POP"] * grade["PESO"] * 100_000
        taxa_pad = grade.groupby(["CODMUN6", "ANO"])["CONTRIB"].sum().rename("TAXA_PADRONIZADA_100K")
        fm = fm.merge(taxa_pad.reset_index(), on=["CODMUN6", "ANO"], how="left")

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
    pop_faixa = extrair_populacao_idade()
    pib       = extrair_pib()
    desp      = extrair_despesa_saude()
    df_ipea   = extrair_ipea()
    for nome, d in (("população", pop), ("população por idade", pop_faixa), ("PIB", pib)):
        if d.empty:
            raise RuntimeError(f"[ETL] Falha ao obter {nome} do IBGE.")
    fm        = construir_feature_matrix(obitos, pop, pib, desp, df_ipea, pop_faixa)

    print("── ETL concluído ────────────────────────────────────────────\n")
    return fm
