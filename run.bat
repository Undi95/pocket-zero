@echo off
rem pocket-zero, avec l'environnement virtuel du dossier / with this folder's virtual environment:
rem   run.bat train   [options]   entraîne le bot / trains the bot (AlphaZero), resumes where it stopped
rem   run.bat search  [options]   recherche de decks / deck search over every card, played by the bot
rem   run.bat forge   [options]   améliore des decks / improves decks, change by change (paired games)
rem   run.bat ranked  [options]   saison de ranked simulée / simulated ranked season (official rules)
rem   run.bat solo    [options]   decks du mode solo / solo mode decks (Poke Ball to Master Ball)
rem   run.bat ui      [options]   interface / web UI: decklists + live training (/train), FR/EN
rem   run.bat meta                met à jour la méta maintenant / refreshes the meta lists now
rem   run.bat update              dernière version / latest version (git pull, binaries, network)
rem   run.bat help                aide détaillée / detailed help
rem Guide : docs\GUIDE.fr.md (français) / docs\GUIDE.en.md (English)
rem (fins de ligne Windows CRLF obligatoires pour cmd / CRLF line endings required by cmd: see .gitattributes)
setlocal
cd /d "%~dp0"
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (echo Lance d'abord setup.bat / run setup.bat first & exit /b 1)
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
if /i "%CMD%"=="forge"  "%PY%" -m zero.forge %ARGS% & exit /b
if /i "%CMD%"=="ranked" "%PY%" -m zero.ranked %ARGS% & exit /b
if /i "%CMD%"=="ui"     "%PY%" -m zero.ui %ARGS% & exit /b
if /i "%CMD%"=="meta"   "%PY%" lab\tools\meta_sources.py %ARGS% & exit /b
if /i "%CMD%"=="update" (git pull && powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1") & exit /b
chcp 65001 >nul
findstr /b "rem" "%~f0"
echo.
"%PY%" -m zero.loop --help
echo.
"%PY%" -m zero.search --help
