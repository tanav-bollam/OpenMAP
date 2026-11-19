@echo off
rem Starts the priority AST run (2 extra seeds + one 10-epoch run, ~7 hours). Keep this window open and the laptop awake.
cd /d "%~dp0"
if not exist logs mkdir logs
echo Started %date% %time%. Progress: logs\ast_seeds.log. Do not close this window.
"C:\Program Files\Git\bin\bash.exe" scripts/run_ast_seeds.sh >> logs\ast_seeds.log 2>&1
echo Finished %date% %time%. Check logs\ast_seeds.log
pause
