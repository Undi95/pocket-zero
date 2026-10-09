@echo off
rem pocket-zero, avec l'environnement virtuel du dossier :
rem   run.bat train   [options]   entraîne le bot (AlphaZero), reprend là où il s'est arrêté
rem   run.bat search  [options]   recherche de decks sur toutes les cartes avec le bot entraîné
rem   run.bat update              récupère la dernière version (git pull + binaires)
rem   run.bat help                aide détaillée
setlocal
cd /d "%~dp0"
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (echo Lance d'abord setup.bat & exit /b 1)
set CMD=%1
shift
set ARGS=
:loop
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto loop
:run
if /i "%CMD%"=="train"  "%PY%" -m zero.loop %ARGS% & exit /b
if /i "%CMD%"=="search" "%PY%" -m zero.search %ARGS% & exit /b
if /i "%CMD%"=="update" (git pull && powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1") & exit /b
"%PY%" -m zero.loop --help
echo.
"%PY%" -m zero.search --help
