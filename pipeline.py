"""
Pipeline principal do TCC.
Executa ETL completo + análise e gera os artefatos para o dashboard.

Uso:
    python pipeline.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def banner():
    print("=" * 62)
    print("  TCC — Mortalidade por Causas Evitáveis no Brasil")
    print("  IESB 2026/01")
    print("  Alunos  : João Marcos S. Carvalho & Maria Clara F. Silva")
    print("  Orientador: Prof. Cristiano Lehrer")
    print("=" * 62)


def main():
    banner()
    t0 = time.time()

    print("\n[1/2] ETL — Extração, Transformação e Carga")
    print("      (download SIM via PySUS + IBGE + SICONFI + IPEA)\n")
    from src.etl import executar_etl
    executar_etl()

    print("\n[2/2] Análise — Clusterização · Regressão · Mineração\n")
    from src.analise import executar_analise
    executar_analise()

    elapsed = int(time.time() - t0)
    print(f"\n{'='*62}")
    print(f"  Pipeline concluído em {elapsed//60}min {elapsed%60}s.")
    print(f"  Execute o dashboard:")
    print(f"    streamlit run app.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
