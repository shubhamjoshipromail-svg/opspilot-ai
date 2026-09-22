# Experiment log

Each entry states a hypothesis, the method, and the criterion that would count
as success *before* the run. Results are recorded whether or not they support
the hypothesis — the Jev benchmark is kept precisely because it failed.

## Protocol

- Development and model selection happen on **validation**. The **test** split
  is evaluated once, at the end of a line of work, and that number is locked.
- Every reported metric names its split.
- Headline metrics carry bootstrap 95% CIs (`final_eval.py`). A difference
  smaller than the CI width is not a result.
- Paired model comparisons use McNemar on identical rows, not two independent
  accuracy numbers.
- Calibration is reported alongside accuracy. A model whose confidence is used
  for routing must have its confidence measured.

## Completed

| # | Experiment | Outcome |
|---|---|---|
| 1 | TF-IDF + LogReg baseline | 0.4409 (10-class, test) |
| 2 | ModernBERT-base, original taxonomy | 0.5229 (split unrecorded) |
| 3 | `clean_v1` taxonomy redesign | 0.7313 val / **0.7398 test** |
| 4 | [Zero-shot decision model](jev_vs_modernbert.md) | **Negative.** 0.5458, below the 0.5946 majority baseline |
| 5 | Calibration measurement | ECE 0.0803 test — gate is sound |

## Queued

Ordered by expected value per hour of work.

### E6 — Class weighting for minority recall

**Hypothesis.** Macro-F1 (0.5759) lags accuracy (0.7398) because the model
under-predicts minority queues: macro recall 0.5174 against macro precision
0.6821. Inverse-frequency weighting should raise minority recall at a small
cost to overall accuracy.

**Method.** `train_modernbert.py --class-weighting sqrt_balanced
--run-name clean_v1_sqrtbal`. The flag exists and has never been used; the
published run used `none`. Compare on validation, then lock on test.

**Success.** Macro-F1 improves by more than the CI width (>0.03) with accuracy
falling less than 0.02. If accuracy drops further, the trade is not worth it
for a routing product where the majority queue dominates volume.

**Why first.** Highest-value single run. One GPU session, one flag, and it
targets the weakest measured dimension.

### E7 — Per-class confidence thresholds

**Hypothesis.** A single global gate is unsafe because calibration is uneven:
`returns_and_exchanges` has ECE 0.1948 against the global 0.0803. Per-class
thresholds should raise coverage at fixed accuracy.

**Method.** Fit a threshold per predicted class on validation targeting equal
precision (say 0.85), then evaluate coverage on test against the single-gate
baseline of 62.5% at 85.0%.

**Success.** Coverage improves by 3pp or more at equal or better accuracy.

**Why.** Directly moves the number the product is judged on — how many tickets
a human never has to look at.

### E8 — Label-noise ceiling

**Hypothesis.** The dominant error is bidirectional confusion between
`customer_general` and `technical_product_support`. If human annotators also
disagree on those tickets, every model here is near a dataset-imposed ceiling
and further modeling is wasted effort.

**Method.** Sample 100 tickets: 50 from that confusion cell, 50 correctly
classified as controls. Label them blind against the queue definitions, then
compute Cohen's kappa against the dataset labels. The 229 tickets where the
zero-shot model beat the fine-tuned one are a sharper sample than random.

**Success.** This experiment cannot fail usefully. High kappa means the labels
are sound and modeling has headroom; low kappa means the ceiling is the data,
which reframes every number in this repository.

**Why.** No GPU, no cost, ~1 hour, and it determines whether E6 and E7 are
worth running at all.

### E9 — Metadata ablation

**Hypothesis.** The limitations section asserts that priority and routing need
metadata beyond ticket text. `model_text` (which appends `type` and `tags`)
should outperform `subject + body`. This is currently an untested assumption.

**Method.** Train identical configs on both inputs. Watch for leakage: `tags`
may be assigned post-triage, in which case improvement is not usable at
inference. Inspect tag timing before trusting any gain.

**Success.** >0.02 accuracy gain that survives the leakage check.

### E10 — Isolate taxonomy from architecture

**Hypothesis.** The 0.5229 to 0.7398 improvement mixes a taxonomy change with
a model change. Running TF-IDF under `clean_v1` separates them.

**Method.** `train_baseline.py` with the `clean_v1` label map. Compare the
four cells: {TF-IDF, ModernBERT} x {10-class, clean_v1}.

**Success.** A clean attribution of the gain. If TF-IDF also jumps to ~0.68,
the headline story is "taxonomy design beat architecture", which is both more
honest and more interesting than the current framing.

**Why.** Closes the one remaining provenance gap, and the result is
publishable either way.

### E11 — Escalation detection without labels

**Hypothesis.** `routing.py` detects escalation risk with a keyword list, which
misses paraphrase ("I have spoken to my attorney") and false-fires on negation
("I am not angry"). No supervised alternative is possible because the dataset
has no observed escalation outcomes — the existing risk labels are derived from
those same keywords.

**Method.** Hand-label 200 tickets for escalation intent. Compare the keyword
rule against a zero-shot `Noul`. Report precision/recall on the minority class,
not accuracy.

**Success.** Recall improves at equal or better precision.

**Why.** The one place in this system where zero-shot has a structural
advantage: the absence of training data is the reason to use it. See the
placement analysis in [jev_vs_modernbert.md](jev_vs_modernbert.md).

## Infrastructure debt

Not experiments, but they gate how fast the above can run:

- **Pin dependencies.** `requirements.txt` is unpinned; results are not
  reproducible across environments.
- **One-command reproduction.** A `make reproduce` that regenerates the dataset,
  runs evaluation, and rewrites the results tables from metrics JSON, so no
  number in the docs is hand-copied.
- **Record the split in every metrics file.** The 10-class run's missing split
  is the direct cost of not doing this.
- **A data card** for the dataset, covering provenance, licence, known label
  quality issues, and the leakage controls applied.
