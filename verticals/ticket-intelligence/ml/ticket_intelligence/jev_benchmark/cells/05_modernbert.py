"""## 5. Fine-tuned ModernBERT

Pulls the Hub checkpoint and runs it on the identical rows. The Hub repo has no
Inference Provider attached, so the weights run locally on the Colab GPU."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_MODEL = 'shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1'
BATCH_SIZE = 32
MAX_LENGTH = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
if device == 'cpu':
    print('WARNING: no GPU. Runtime > Change runtime type > T4 GPU for a faster run.')

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).to(device).eval()
id2label = model.config.id2label
mb_labels = [id2label[i] for i in sorted(id2label)]
print(f'{len(mb_labels)} labels: {mb_labels}')
assert set(mb_labels) == set(LABELS), 'Label space differs from the Jev run'

# Same rows Jev saw, in the same order.
mb_subset = subset.loc[subset[ID_COL].astype(str).isin(set(jev['ticket_id']))]
texts = mb_subset['_text'].tolist()
mb_rows = []
started_all = time.perf_counter()

for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]
    encoded = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors='pt').to(device)
    batch_started = time.perf_counter()
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1).cpu()
    per_item_ms = (time.perf_counter() - batch_started) * 1000 / len(batch)

    for offset, row_probs in enumerate(probs):
        idx = start + offset
        source = mb_subset.iloc[idx]
        true_label = to_clean_v1(str(source[LABEL_COL]))
        ordered, _ = torch.sort(row_probs, descending=True)
        top1, top2 = float(ordered[0]), float(ordered[1])
        predicted = mb_labels[int(torch.argmax(row_probs))]
        record = {'ticket_id': str(source[ID_COL]), 'true_label': true_label,
                  'predicted_label': predicted, 'confidence': top1,
                  'top_1_probability': top1, 'top_2_probability': top2,
                  'top1_top2_margin': top1 - top2,
                  'correct': int(predicted == true_label),
                  'latency_ms': round(per_item_ms, 2)}
        for label_idx, label in enumerate(mb_labels):
            record[f'prob_{label}'] = float(row_probs[label_idx])
        mb_rows.append(record)

mb = pd.DataFrame(mb_rows)
elapsed = time.perf_counter() - started_all
print(f"\nModernBERT: {len(mb)} tickets, acc={mb['correct'].mean():.4f}")
print(f'{elapsed:.1f}s wall, {len(mb)/elapsed:.1f} tickets/s on {device}')
