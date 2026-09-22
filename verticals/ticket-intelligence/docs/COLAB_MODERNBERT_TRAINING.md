# Colab ModernBERT Training

This guide trains the OpsPilot Ticket Intelligence v2 transformer benchmark on Google Colab GPU.

The first v2 target is `parent_queue` routing:

```text
input = subject + "\n\n" + body
label = normalized category / parent_queue
```

`answer` and `reference_answer` are never used as model input because they are post-resolution fields and would leak information.

## 1. Runtime

In Colab:

1. Open a notebook.
2. Select `Runtime > Change runtime type`.
3. Choose `GPU`.

Check GPU:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

## 2. Clone Repo

```bash
cd /content
git clone -b codex/ticket-intelligence https://github.com/shubhamjoshipromail-svg/opspilot-ai.git
cd /content/opspilot-ai
```

If you uploaded the repo manually, `cd` into that uploaded repo folder instead.

## 3. Install Dependencies

```bash
pip install -q torch transformers datasets evaluate accelerate sentencepiece scikit-learn pandas matplotlib
```

ModernBERT may require a recent `transformers` version. If loading `answerdotai/ModernBERT-base` fails because of package support, upgrade:

```bash
pip install -q --upgrade transformers accelerate
```

## 4. Make Sure Splits Exist

The repo includes fixed split CSVs. To regenerate them from the public Hugging Face dataset:

```bash
python scripts/build_ticket_dataset.py
cd verticals/ticket-intelligence
python ml/ticket_intelligence/create_splits.py --input ../../data/processed/tickets_en_normalized.csv
```

Expected split files:

```text
verticals/ticket-intelligence/data/processed/train.csv
verticals/ticket-intelligence/data/processed/val.csv
verticals/ticket-intelligence/data/processed/test.csv
```

## 5. Smoke Test

Run a small sample first. This checks dataset loading, tokenization, labels, training loop, and output writing.

```bash
cd /content/opspilot-ai/verticals/ticket-intelligence

python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-base \
  --run-name smoke_modernbert_parent_queue \
  --epochs 1 \
  --batch-size 4 \
  --max-length 256 \
  --sample-size 500
```

If ModernBERT fails due to package or compute limits, use the fallback:

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name microsoft/deberta-v3-small \
  --run-name smoke_deberta_small_parent_queue \
  --epochs 1 \
  --batch-size 4 \
  --max-length 256 \
  --sample-size 500
```

## 6. Full Parent Queue Training

```bash
cd /content/opspilot-ai/verticals/ticket-intelligence

python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-base \
  --run-name modernbert_base_parent_queue \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

## 7. Model Bakeoff Commands

Use unique `--run-name` values so each run writes separate artifacts and outputs.

### 1. ModernBERT-base baseline

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-base \
  --run-name modernbert_base_parent_queue \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

### 2. ModernBERT-base + sqrt class weights

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-base \
  --run-name modernbert_base_parent_queue_sqrt_weights \
  --class-weighting sqrt_balanced \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

Use `--class-weighting balanced` if you want stronger inverse-frequency weighting.

### 3. ModernBERT-large + class weights

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-large \
  --run-name modernbert_large_parent_queue_sqrt_weights \
  --class-weighting sqrt_balanced \
  --epochs 3 \
  --batch-size 4 \
  --max-length 512 \
  --learning-rate 2e-5
```

Large models may need smaller batch sizes on free Colab GPUs.

### 4. DeBERTa-v3-large + class weights

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name microsoft/deberta-v3-large \
  --run-name deberta_v3_large_parent_queue_sqrt_weights \
  --class-weighting sqrt_balanced \
  --epochs 3 \
  --batch-size 4 \
  --max-length 512 \
  --learning-rate 2e-5
```

Fallback smaller DeBERTa:

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name microsoft/deberta-v3-small \
  --run-name deberta_v3_small_parent_queue_sqrt_weights \
  --class-weighting sqrt_balanced \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

### 5. clean_v1 taxonomy run

This merges overlapping labels into cleaner routing groups:

- `technical_support`, `it_support`, `product_support` -> `technical_product_support`
- `customer_service`, `general_inquiry` -> `customer_general`
- other labels remain separate

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-base \
  --run-name modernbert_base_parent_queue_clean_v1_sqrt_weights \
  --label-map clean_v1 \
  --class-weighting sqrt_balanced \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

Use `distilbert-base-uncased` only if the above fail.

## 8. Threshold Analysis

After a run creates predictions, analyze confidence thresholds for auto-routing vs human review:

```bash
python ml/ticket_intelligence/threshold_analysis.py \
  --predictions ml/ticket_intelligence/outputs/modernbert_base_parent_queue_predictions.csv
```

For a named run:

```bash
python ml/ticket_intelligence/threshold_analysis.py \
  --predictions ml/ticket_intelligence/outputs/modernbert_base_parent_queue_sqrt_weights_predictions.csv
```

If prediction files include `top1_top2_margin`, you can require a minimum margin too:

```bash
python ml/ticket_intelligence/threshold_analysis.py \
  --predictions ml/ticket_intelligence/outputs/modernbert_base_parent_queue_sqrt_weights_predictions.csv \
  --margin-threshold 0.10
```

Outputs:

```text
ml/ticket_intelligence/outputs/{run_name}_coverage_by_threshold.csv
ml/ticket_intelligence/outputs/{run_name}_coverage_by_threshold.png
```

## 9. Build Leaderboard

After training:

```bash
python ml/ticket_intelligence/compare_models.py
```

Outputs:

```text
ml/ticket_intelligence/outputs/model_leaderboard.csv
ml/ticket_intelligence/outputs/model_leaderboard.json
```

## 10. Expected Outputs

Model artifact:

```text
ml/ticket_intelligence/artifacts/{run_name}/
```

Reports:

```text
ml/ticket_intelligence/outputs/{run_name}_metrics.json
ml/ticket_intelligence/outputs/{run_name}_classification_report.csv
ml/ticket_intelligence/outputs/{run_name}_confusion_matrix.png
ml/ticket_intelligence/outputs/{run_name}_predictions.csv
ml/ticket_intelligence/outputs/{run_name}_error_analysis.csv
ml/ticket_intelligence/outputs/{run_name}_run_summary.md
```

If you do not pass `--run-name`, the legacy default remains:

```text
modernbert_parent_queue_*
```

## 11. Zip And Download

```bash
zip -r modernbert_parent_queue_results.zip \
  ml/ticket_intelligence/artifacts/modernbert_parent_queue \
  ml/ticket_intelligence/outputs/modernbert_parent_queue_* \
  ml/ticket_intelligence/outputs/model_leaderboard.*
```

Optional Google Drive export:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cp modernbert_parent_queue_results.zip /content/drive/MyDrive/
```

## Notes

- Use the fixed train/validation/test splits for fair comparison with v1.
- Validation is used for model selection and early stopping.
- Test is evaluated once at the end.
- This is a transformer-based ticket-routing benchmark, not a production-ready routing system.
- Do not use synthetic data for real model results.
