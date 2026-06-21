@echo off
title TCC IESB 2026

echo.
echo  ============================================================
echo   TCC - Mortalidade por Causas Evitaveis no Brasil
echo   IESB 2026/01 - Ciencia da Computacao
echo   Alunos    : Joao Marcos e Maria Clara
echo   Orientador: Prof. Cristiano Lehrer
echo  ============================================================
echo.

REM 1. Copia models/ se a pasta estiver vazia
if not exist "models\perfil_clusters.csv" (
    echo [Setup] Copiando dados do projeto de referencia...
    if exist "analise-mortalidade-evitavel-brasil\models\" (
        xcopy /E /I /Y "analise-mortalidade-evitavel-brasil\models\*" "models\" >nul
        echo [Setup] OK - models copiados.
    ) else (
        echo [Aviso] Dados nao encontrados. Execute: python pipeline.py
    )
)

REM 2. Copia logo IESB se necessario
if not exist "iesblogo.png" (
    if exist "analise-mortalidade-evitavel-brasil\iesblogo.png" (
        copy "analise-mortalidade-evitavel-brasil\iesblogo.png" "iesblogo.png" >nul
    )
)

REM 3. Verifica dependencias
echo [Setup] Verificando dependencias...
python -c "import streamlit, pandas, plotly" 2>nul
if errorlevel 1 (
    echo [Setup] Instalando dependencias...
    pip install -r requirements.txt
) else (
    echo [Setup] Dependencias OK.
)

REM 4. Inicia o dashboard
echo.
echo [Dashboard] Abrindo em http://localhost:8501
echo.
python -m streamlit run app.py

pause
