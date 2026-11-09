@echo off
rem After run_iclabel_siena.cmd has finished: does P6 itself remove the flagged brain
rem components? Runs alone (Infomax depends on the execution context, docs/ICA_REPRO.md).
cd /d C:\Projects\python\eeg-fusion-benchmark
set LOG=results_v2\iclabel
:wait
if exist %LOG%\siena_batch_done.txt goto run
timeout /t 300 /nobreak > nul
goto wait
:run
for %%S in (PN00 PN12 PN14 PN10 PN17) do (
  .venv-ica\Scripts\python.exe -W ignore -m src.iclabel_p6_match --subject %%S >> %LOG%\p6_match_run.log 2>&1
)
echo all done >> %LOG%\p6_match_done.txt
