# Note to amendment A12 (written after job submission, before any non-smoke result existed)

The frozen file docs/preregistration_v1_amendment_A12.md (sha256 ef2fdb0d...) was meant to contain the following sentence under "What has been seen"; the update of the file failed on the cluster (the file was already read-only) and was noticed after submission. The text is recorded here instead, unchanged in substance:

"Smoke runs (20 questions each, code test only) before the freeze: the A9 3B adapter scored 20/20 on orig, 16/20 on pres and 16/20 on rand (n = 20, so uninformative, but it signals that the filler may cost accuracy and V may fail; V is kept as written)."

The M1 alarm analysis (CPU, scripts/m1_alarm.py, hashed in the A12 freeze) was run right after the freeze and before the GPU jobs finished; its result is in analysis/m1_alarm_results.md.
