@echo off
rem pocket-zero, avec l'environnement virtuel du dossier :
rem   run.bat train   [options]   entraîne le bot (AlphaZero), reprend là où il s'est arrêté
rem   run.bat search  [options]   recherche de decks sur toutes les cartes avec le bot entraîné
rem   run.bat solo    [options]   tournoi des decks du mode solo (Poké Ball à Master Ball)
rem   run.bat ranked  [options]   saison de ranked simulée (règles officielles, méta réelle)
rem   run.bat ui                  decklists avec les images officielles (navigateur)
rem   run.bat meta                met à jour les decks méta des autres sources
rem   run.bat update              récupère la dernière version (git pull + binaires)
rem   run.bat help                aide détaillée
rem (fins de ligne Windows CRLF obligatoires pour cmd : voir .gitattributes)
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
if /i "%CMD%"=="solo"   "%PY%" -m zero.solo %ARGS% & exit /b
if /i "%CMD%"=="ranked" "%PY%" -m zero.ranked %ARGS% & exit /b
if /i "%CMD%"=="ui"     "%PY%" -m zero.ui %ARGS% & exit /b
if /i "%CMD%"=="meta"   "%PY%" lab\tools\meta_sources.py %ARGS% & exit /b
if /i "%CMD%"=="update" (git pull && powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1") & exit /b
"%PY%" -m zero.loop --help
echo.
"%PY%" -m zero.search --help
