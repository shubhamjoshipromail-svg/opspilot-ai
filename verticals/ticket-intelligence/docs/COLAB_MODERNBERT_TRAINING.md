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
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

Fallbacks if needed:

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name microsoft/deberta-v3-base \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name microsoft/deberta-v3-small \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

Use `distilbert-base-uncased` only if the above fail.

## 7. Build Leaderboard

After training:

```bash
python ml/ticket_intelligence/compare_models.py
```

Outputs:

```text
ml/ticket_intelligence/outputs/model_leaderboard.csv
ml/ticket_intelligence/outputs/model_leaderboard.json
```

## 8. Expected Outputs

Model artifact:

```text
ml/ticket_intelligence/artifacts/modernbert_parent_queue/
```

Reports:

```text
ml/ticket_intelligence/outputs/modernbert_parent_queue_metrics.json
ml/ticket_intelligence/outputs/modernbert_parent_queue_classification_report.csv
ml/ticket_intelligence/outputs/modernbert_parent_queue_confusion_matrix.png
ml/ticket_intelligence/outputs/modernbert_parent_queue_predictions.csv
ml/ticket_intelligence/outputs/modernbert_parent_queue_error_analysis.csv
ml/ticket_intelligence/outputs/modernbert_parent_queue_run_summary.md
```

## 9. Zip And Download

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
