@echo off
rem Installe pocket-zero dans ce dossier (environnement virtuel + binaires). Voir setup.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
pause
