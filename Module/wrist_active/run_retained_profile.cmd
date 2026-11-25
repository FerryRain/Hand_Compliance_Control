@echo off
setlocal
for %%D in ("%~dp0..\..") do set "WRIST_ACTIVE_REPO=%%~fD"
set "PYTHONPATH=%WRIST_ACTIVE_REPO%\.runtime;%WRIST_ACTIVE_REPO%\.runtime\gpis_deps;%WRIST_ACTIVE_REPO%"
set "PYTHONUTF8=1"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
"%WRIST_ACTIVE_REPO%\.venv\Scripts\python.exe" -B -u -m Module.wrist_active.palm_cracker_parameters %*
exit /b %ERRORLEVEL%
