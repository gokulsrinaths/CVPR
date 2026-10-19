# Amendment A28 to pre-registration v1: option-length and position audit across public multiple-choice benchmarks (exploratory, text only)

Appended 2026-10-05. v1 and A0-A27 unchanged. **Disclosure: this audit was run before this amendment was written**, so it is exploratory, not confirmatory; the amendment freezes the script, the data sources and the results before anything is claimed from them. No model and no image is involved.

## Question
Which public multiple-choice benchmarks carry a nonvisual option-length or position cue of the kind that makes Med-CMR text-decisive?

## Method (`scripts/cue_audit.py`, output `analysis/cue_audit_results.json`)
For every question: option text is stripped of its leading letter; length is the number of characters. The *shortest-option rule* picks the shortest option, splitting ties uniformly (credit 1/ties); its accuracy is compared with the chance level (mean of 1/options). The *longest-option rule* is the mirror. The most common position of the correct option is reported with its share. No threshold, test or decision rule is attached.

Sources: Med-CMR challenge training selection (14,398 five-option MCQs); PMC-VQA train.csv (176,934) and test_clean.csv (2,000); OmniMedVQA open-access QA files (88,996; 2-4 options, hence chance 26.9%); A-OKVQA validation (1,145); MedXpertQA-MM test (2,000; Hugging Face, TsinghuaC3I); MMMU validation, five health subjects, multiple-choice only (145). GMAI-MMBench is gated and was not obtained.

## Results (shortest-option rule accuracy / chance; correct option most often at)
| Benchmark | n | rule | chance | position |
|---|---|---|---|---|
| Med-CMR | 14,398 | 91.9 | 20.0 | A 20.7% |
| PMC-VQA train | 176,934 | 30.3 | 25.0 | B 30.6% |
| PMC-VQA test | 2,000 | 24.4 | 25.0 | B 31.9% |
| OmniMedVQA | 88,996 | 32.7 | 26.9 | B 26.9% |
| A-OKVQA | 1,145 | 23.6 | 25.0 | C 25.8% |
| MedXpertQA-MM | 2,000 | 18.4 | 20.0 | E 21.2% |
| MMMU health | 145 | 29.1 | 26.5 | A 31.0% |

Excess over chance (points): Med-CMR +71.9, OmniMedVQA +5.8, PMC-VQA train +5.3, MMMU health +2.6 (n=145), PMC-VQA test -0.6, A-OKVQA -1.4, MedXpertQA-MM -1.6.

## Reading (fixed before it is used in the paper)
The paper states that, among the seven sets audited, only Med-CMR carries a severe length cue; that PMC-VQA has a weak position skew (correct option B in 31% of questions) and a training-set length rule slightly above chance that is absent in its clean test set; and that OmniMedVQA's 32.7% must be read against its 26.9% chance level. It does not state that other benchmarks are free of cues, because only two cue types were examined, and does not state any claim about image use for them from this audit.

## Limits (stated in advance)
Two cue types only (length, position); character length ignores token length and lexical overlap with the question; MMMU is small; GMAI-MMBench and the closed-ended VQA-RAD-style sets were not examined; exploratory.

## Amendments log
(A0-A28.)
