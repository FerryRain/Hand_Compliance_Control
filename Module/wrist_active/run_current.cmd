@echo off
setlocal
rem Default to PalmReach-6, the retained six-direction controller and deployed object profile.
for %%D in ("%~dp0..\..") do set "WRIST_ACTIVE_REPO=%%~fD"
set "PYTHONPATH=%WRIST_ACTIVE_REPO%\.runtime;%WRIST_ACTIVE_REPO%\.runtime\gpis_deps;%WRIST_ACTIVE_REPO%"
set "PYTHONUTF8=1"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
"%WRIST_ACTIVE_REPO%\.venv\Scripts\python.exe" -B -u -m Module.wrist_active.current_cli %*
exit /b %ERRORLEVEL%
