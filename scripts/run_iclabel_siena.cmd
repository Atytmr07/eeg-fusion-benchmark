@echo off
rem ICLabel overlap check on further Siena patients, one per seizure profile (advisor, 2026-11-07).
rem Each patient: most seizures per recording hour within its group (subject_info.csv only).
rem Resumable: finished recordings are skipped on a rerun.
cd /d C:\Projects\python\eeg-fusion-benchmark
set PY=.venv-ica\Scripts\python.exe -W ignore -m src.iclabel_check
set LOG=results_v2\iclabel

%PY% --subject PN12 --selection "Left temporal IAS: most seizures per recording hour among the left-sided patients listed with 29 electrodes in subject_info.csv. Chosen before any result." >> %LOG%\PN12_run.log 2>&1
%PY% --subject PN14 --selection "WIAS: the only patient with focal seizures without impaired awareness. Chosen before any result." >> %LOG%\PN14_run.log 2>&1
%PY% --subject PN10 --selection "Frontal, bilateral FBTC: the only patient with focal to bilateral tonic-clonic seizures. Chosen before any result." >> %LOG%\PN10_run.log 2>&1
%PY% --subject PN17 --selection "Second right temporal IAS: most seizures per recording hour among right-sided patients listed with 29 electrodes in subject_info.csv other than PN00 with at least two seizures. Chosen before any result." >> %LOG%\PN17_run.log 2>&1
echo all done >> %LOG%\siena_batch_done.txt
