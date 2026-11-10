@echo off
rem Does P6 itself remove the flagged brain components? Runs alone, after the Siena batch
rem (Infomax depends on the execution context, docs/ICA_REPRO.md). Short recordings first.
rem --force: the first run (2026-11-10 night) had a bug that set every removed_share to 0.
cd /d C:\Projects\python\eeg-fusion-benchmark
set PY=.venv-ica\Scripts\python.exe -W ignore -m src.iclabel_p6_match --force
set LOG=results_v2\iclabel\p6_match_run.log
%PY% --subject PN00 --records PN00-2.edf PN00-5.edf >> %LOG% 2>&1
%PY% --subject PN12 --records PN12-1.2.edf >> %LOG% 2>&1
%PY% --subject PN17 --records PN17-1.edf PN17-2.edf >> %LOG% 2>&1
%PY% --subject PN14 --records PN14-1.edf PN14-2.edf PN14-3.edf >> %LOG% 2>&1
%PY% --subject PN10 --records PN10-1.edf PN10-4.5.6.edf >> %LOG% 2>&1
echo all done >> results_v2\iclabel\p6_match_done2.txt
