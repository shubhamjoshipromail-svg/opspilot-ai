"""## 6. Compare

Accuracy is the headline, but the decisive number is accuracy at a fixed human
review budget: given that a person can only look at N% of tickets, which system
routes the rest more accurately?

McNemar tests whether the accuracy gap is real or sampling noise."""

from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import binomtest

paired = jev.merge(mb, on='ticket_id', suffixes=('_jev', '_mb'))
assert (paired['true_label_jev'] == paired['true_label_mb']).all(), 'label mismatch'
print(f'paired on {len(paired)} tickets\n')

def summarize(frame, name):
    return {'system': name,
            'accuracy': round(accuracy_score(frame['true_label'], frame['predicted_label']), 4),
            'macro_f1': round(f1_score(frame['true_label'], frame['predicted_label'],
                                       average='macro', zero_division=0), 4),
            'weighted_f1': round(f1_score(frame['true_label'], frame['predicted_label'],
                                          average='weighted', zero_division=0), 4),
            'median_latency_ms': round(float(frame['latency_ms'].median()), 1)}

summary = pd.DataFrame([summarize(jev, 'Jev (zero-shot)'),
                        summarize(mb, 'ModernBERT (fine-tuned)')])
print(summary.to_string(index=False))

# McNemar: only the disagreements carry information.
jev_only = int(((paired['correct_jev'] == 1) & (paired['correct_mb'] == 0)).sum())
mb_only = int(((paired['correct_jev'] == 0) & (paired['correct_mb'] == 1)).sum())
print(f'\nJev right / ModernBERT wrong: {jev_only}')
print(f'ModernBERT right / Jev wrong: {mb_only}')
if jev_only + mb_only:
    p = binomtest(jev_only, jev_only + mb_only, 0.5).pvalue
    verdict = 'significant at p<0.05' if p < 0.05 else 'NOT significant - could be noise'
    print(f'McNemar exact p = {p:.4g}  ({verdict})')


def accuracy_at_budget(frame, budgets=(0.1, 0.2, 0.3, 0.5)):
    ranked = frame.sort_values('confidence', ascending=False)
    rows = []
    for budget in budgets:
        keep = int(round(len(ranked) * (1 - budget)))
        if keep <= 0:
            continue
        auto = ranked.head(keep)
        rows.append({'human_review_budget': budget, 'auto_routed': keep,
                     'accuracy_on_auto': round(accuracy_score(
                         auto['true_label'], auto['predicted_label']), 4)})
    return pd.DataFrame(rows)

budget = accuracy_at_budget(jev).merge(
    accuracy_at_budget(mb), on=['human_review_budget', 'auto_routed'],
    suffixes=('_jev', '_mb'))
print('\nAccuracy at fixed human-review budget:')
print(budget.to_string(index=False))


def ece(frame, n_bins=10):
    bins = pd.cut(frame['confidence'].clip(0, 1),
                  bins=[i / n_bins for i in range(n_bins + 1)], include_lowest=True)
    total, score = len(frame), 0.0
    for _, group in frame.groupby(bins, observed=True):
        if not group.empty:
            score += len(group) / total * abs(group['correct'].mean() - group['confidence'].mean())
    return round(score, 4)

print(f"\nCalibration error (lower is better)")
print(f"  Jev ECE:        {ece(jev)}")
print(f"  ModernBERT ECE: {ece(mb)}")

OUT_DIR = Path('/content/drive/MyDrive/opspilot/jev_benchmark')
OUT_DIR.mkdir(parents=True, exist_ok=True)
jev.to_csv(OUT_DIR / 'jev_predictions.csv', index=False)
mb.to_csv(OUT_DIR / 'modernbert_predictions.csv', index=False)
summary.to_csv(OUT_DIR / 'summary.csv', index=False)
budget.to_csv(OUT_DIR / 'accuracy_at_budget.csv', index=False)
print(f'\nSaved 4 files to {OUT_DIR}')
