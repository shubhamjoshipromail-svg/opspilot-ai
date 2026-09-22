# Can a zero-shot decision model replace a fine-tuned encoder?

**Answer: not for queue routing. But it belongs somewhere the encoder cannot go.**

Date: 2026-09-20 · Harness: `verticals/ticket-intelligence/ml/ticket_intelligence/jev_benchmark/`

---

## The question

OpsPilot routes tickets with a fine-tuned ModernBERT. Fine-tuning costs GPU
time, needs 16,622 labeled examples, and has to be redone whenever the queue
taxonomy changes. TypeSafe's Jev is a System One model that returns typed
decisions and calibrated probabilities with no training at all.

If zero-shot matched fine-tuned, the training pipeline becomes optional. Worth
measuring rather than assuming.

## Setup

| | |
|---|---|
| Task | `parent_queue`, 7-class `clean_v1` taxonomy |
| Input | subject + blank line + body |
| Split | Validation, n=3,562, `random_state=42` |
| Jev question | One `Choice` per ticket, criteria written per queue |
| Excluded | `reference_answer` (post-resolution, would leak) |

Criteria were written specifically to separate the queues that the ModernBERT
confusion matrix showed collapsing into one another, not as generic one-line
labels.

**Deliberate asymmetry:** ModernBERT saw 16,622 in-domain examples. Jev saw
none. That is the comparison being made, not a flaw in it.

## Results

Both systems run on the same 3,562 validation rows, row by row.

| Metric | Jev (zero-shot) | ModernBERT (fine-tuned) |
|---|---:|---:|
| Accuracy | 0.5458 | **0.7294** |
| Macro-F1 | 0.3289 | 0.5630 |
| ECE (calibration) | 0.2896 | **0.0841** |
| Latency p50 | 337ms | 55ms (local MPS, batched) |
| Cost | $0.1285 per 3,562 tickets | GPU training + hosting |
| Failures | 0 | 0 |

**Paired result:** ModernBERT is exclusively correct on 883 tickets, Jev on
229. McNemar exact p = 5.5e-91. The gap is not sampling noise.

The local run reproduces 0.7294 against the 0.7313 recorded in the published
`trainer_state.json`, a 0.2pp difference from tokenizer and batching details.

**Majority-class baseline: 0.5946.** Always guessing
`technical_product_support` beats Jev here. That is the number that settles it.
An 18-point gap to a fine-tuned model is unsurprising; failing to clear a
constant classifier is disqualifying for this task.

### Per-class recall

| Queue | Recall | Share of data |
|---|---:|---:|
| technical_product_support | 0.715 | 59.5% |
| billing_and_payments | 0.471 | 10.2% |
| service_outages_and_maintenance | 0.400 | 3.9% |
| sales_and_pre_sales | 0.376 | 3.1% |
| customer_general | 0.250 | 16.5% |
| human_resources | 0.087 | 1.9% |
| returns_and_exchanges | 0.051 | 4.9% |

Macro-F1 of 0.329 against accuracy of 0.546 is the signature of a model
carried by one large class while failing the small ones.

### Calibration did not hold

Calibrated confidence is the headline claim for System One models, and it is
the capability OpsPilot's routing layer depends on. It did not hold here.

| Confidence bin | n | Mean confidence | Actual accuracy | Gap |
|---|---:|---:|---:|---:|
| 0.5–0.6 | 280 | 0.557 | 0.321 | −0.236 |
| 0.7–0.8 | 330 | 0.757 | 0.446 | −0.312 |
| 0.8–0.9 | 352 | 0.856 | 0.489 | −0.367 |
| 0.9–1.0 | 1,968 | 0.984 | 0.650 | −0.333 |

Systematic overconfidence across every bin. Confidence gating does not rescue
it: routing only the most confident half still yields 65.7% accuracy, below
the fine-tuned model's overall score.

The fine-tuned model, measured the same way on the same rows, has ECE 0.0841 —
3.4x better. Its confidence can be thresholded; at a 0.80 gate it auto-routes
61.8% of tickets at 86.5% accuracy.

This is the more consequential finding. An accuracy gap costs you correctness;
a calibration gap costs you the ability to tell when you are wrong, which is
what the entire human-review design depends on.

## Why it lost

The dominant error is **bidirectional**: 305 `customer_general` predicted as
`technical_product_support`, and 279 the reverse. Symmetric confusion at that
scale is not a prompt-wording problem.

Rewriting the `customer_general` criteria to be sharper moved accuracy by
exactly zero — recall on that class rose, and `technical_product_support` fell
by the same amount. Errors were traded, not removed.

The likely explanation: **a fine-tuned model learns this dataset's filing
conventions.** Whether a product complaint requesting a remedy is filed under
`customer_service` or `technical_support` is an arbitrary organizational
choice, learnable from 16,622 examples and not inferable from any description
of the queue. Part of the 18-point gap measures convention memorization rather
than judgment quality.

This also implies a ceiling that constrains every model here, including the
fine-tuned one.

## What this does and does not show

**Shows:** for a fixed taxonomy with abundant labels, fine-tuning wins
decisively on accuracy, and a zero-shot model's confidence should not be
trusted as a routing gate without measuring its calibration on your own data.

**Does not show:** that Jev is a weak model. One task, one criteria design,
zero-shot, against a specialist trained on the exact distribution. A different
task, or richer criteria, could land differently.

## Where a decision model does belong

The honest conclusion is not "do not use Jev." It is that queue routing was the
wrong job for it. Four places in this system where it fits better, ordered by
expected value:

### 1. Escalation risk detection — the strongest case

`routing.py` currently detects escalation with a keyword list:

```text
legal, lawsuit, chargeback, fraud, regulator, complaint, angry, cancel,
refund, charged twice, manager, escalate, urgent, account locked, ...
```

This is brittle in both directions. It misses "I have spoken to my attorney"
and fires on "I am not angry, just confused."

A supervised classifier cannot replace it, because **the dataset has no
observed escalation outcomes** — the existing risk labels are derived from
these same keywords, so training on them only relearns the rules.

That is exactly the gap zero-shot fills. A `Noul` asking *"is this customer
likely to escalate to a complaint, chargeback or legal action?"* needs no
labels. This is where the absence of training data is a reason to use Jev, not
a reason to avoid it.

### 2. Disagreement detection as a review trigger

Run both models. Where ModernBERT and Jev disagree, route to a human. This
needs no accuracy improvement from either model — it converts a second opinion
into review-queue prioritization, which is what the product actually needs.
Worth measuring: is disagreement a better predictor of error than ModernBERT's
own confidence?

### 3. Cold start for new queues

When a queue is added, the fine-tuned model has zero examples and must be
retrained. Jev needs only a description. Zero-shot at 55% beats no classifier
at all while labels accumulate, then hand over once there are enough.

### 4. Label-noise auditing

Use disagreements as a sampling strategy for finding bad labels. If a human
adjudicator sides with Jev on a meaningful share of the
`customer_general`/`technical_product_support` conflicts, the dataset's ceiling
is a labeling artifact — which reframes every accuracy number in this repo.

## Next steps

1. **Blind-label 50 disagreements.** Cheapest, highest information. Determines
   whether the label-noise hypothesis holds, which affects every reported
   number here. One hour, no cost.
2. ~~Run both on the same split for a paired comparison.~~ **Done** —
   `run_modernbert.py` pulls the published checkpoint and scores the same rows.
3. ~~Measure ModernBERT's ECE.~~ **Done** — 0.0841. The confidence gate in
   `routing.py` is sound, and could reasonably be raised from 0.65 to 0.80.
4. **Prototype the escalation `Noul`** against hand-labeled escalation cases.
   The most promising of the four placements below, and it competes against a
   keyword list rather than against a trained transformer.

## Reproducing

```bash
cd verticals/ticket-intelligence/ml/ticket_intelligence/jev_benchmark
echo "TYPESAFE_API_KEY=sk-..." > .env          # gitignored
python3 run_jev.py --taxonomy clean_v1 --limit 50    # smoke test
python3 run_jev.py --taxonomy clean_v1 --input ../../../data/processed/val.csv --out outputs/jev_val.csv
python3 score_jev.py --predictions outputs/jev_val.csv
```

Predictions are written in the schema `threshold_analysis.py` already reads, so
existing tooling works on them unchanged. Runs are resumable and stdlib-only;
the ModernBERT side (`run_modernbert.py`) needs `torch` and `transformers`.
