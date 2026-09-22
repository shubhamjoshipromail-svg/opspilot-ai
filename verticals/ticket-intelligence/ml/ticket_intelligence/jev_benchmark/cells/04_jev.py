"""## 4. Jev (zero-shot)

One Choice question per ticket over the 7 `clean_v1` queues. Criteria are
written to separate the queues that the ModernBERT confusion matrix showed
collapsing into each other.

Set `LIMIT = 50` for a smoke test; `None` runs the full split."""

API_URL = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-latest'
LIMIT = 50
CONCURRENCY = 8

CRITERIA = {
    'technical_product_support': (
        'Anything about the product working, being used, or being accessed: defects '
        'and errors, how-to and configuration questions, integrations, and account '
        'access or environment setup.'),
    'customer_general': (
        'Account and relationship matters with no technical fault, plus generic '
        'questions that fit no other queue: complaints, cancellations, contact '
        'changes, chasing prior tickets, broad information requests.'),
    'billing_and_payments': (
        'Money owed or charged: invoices, payment failures, duplicate or incorrect '
        'charges, refunds, subscription and pricing changes, tax and billing details.'),
    'returns_and_exchanges': (
        'Physical goods moving back or being swapped: returns, exchanges, RMAs, '
        'wrong or damaged items received, shipping a replacement.'),
    'service_outages_and_maintenance': (
        'A shared service is down or degraded for many users, or scheduled '
        'maintenance is the subject. Distinct from one customer isolated fault.'),
    'sales_and_pre_sales': (
        'A prospective or expanding purchase: quotes, pricing for new business, '
        'demos, trials, capability questions asked before buying, upgrades.'),
    'human_resources': (
        'Employment matters: recruitment and applications, onboarding, payroll and '
        'benefits, internal staff or workplace policy questions.'),
}
LABELS = list(CRITERIA)

INSTRUCTIONS = (
    'A customer has written to a support desk. Decide which single operational '
    'queue should own this ticket, based only on the subject and body. Judge what '
    'the customer actually needs done, not the vocabulary they happen to use.')

CLEAN_V1_MAP = {
    'technical_support': 'technical_product_support',
    'it_support': 'technical_product_support',
    'product_support': 'technical_product_support',
    'customer_service': 'customer_general',
    'general_inquiry': 'customer_general',
}
to_clean_v1 = lambda label: CLEAN_V1_MAP.get(label, label)


def ask_jev(state, max_retries=6):
    payload = {'state': state, 'model': MODEL, 'questions': {'queue': {
        'type': 'choice', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}}
    body = json.dumps(payload).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(API_URL, data=body, method='POST', headers={
            'Authorization': f'Bearer {TYPESAFE_API_KEY}',
            'Content-Type': 'application/json'})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                latency = (time.perf_counter() - started) * 1000
                data = json.loads(resp.read().decode())
            answer = data['answers']['queue']
            answer['_usage'] = data.get('usage', {})
            return answer, latency
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 529) or exc.code >= 500:
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue
            raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()[:300]}')
        except (urllib.error.URLError, TimeoutError):
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30) + random.random())


subset = df if LIMIT is None else df.head(LIMIT)
lock = threading.Lock()
jev_rows, stats = [], {'n': 0, 'correct': 0, 'failed': 0}
started_all = time.perf_counter()

def run_one(item):
    _, row = item
    true_label = to_clean_v1(str(row[LABEL_COL]))
    try:
        answer, latency = ask_jev(row['_text'])
    except Exception as exc:
        with lock:
            stats['failed'] += 1
            print(f'  ! {exc}')
        return
    probs = answer.get('probabilities', {}) or {}
    ordered = sorted(probs.values(), reverse=True)
    top1 = float(ordered[0]) if ordered else 0.0
    top2 = float(ordered[1]) if len(ordered) > 1 else 0.0
    predicted = answer.get('choice', '')
    record = {'ticket_id': str(row[ID_COL]), 'true_label': true_label,
              'predicted_label': predicted, 'confidence': answer.get('confidence', 0.0),
              'top_1_probability': top1, 'top_2_probability': top2,
              'top1_top2_margin': top1 - top2, 'correct': int(predicted == true_label),
              'latency_ms': round(latency, 1),
              'input_tokens': answer.get('_usage', {}).get('input_tokens', 0)}
    for label in LABELS:
        record[f'prob_{label}'] = float(probs.get(label, 0.0))
    with lock:
        jev_rows.append(record)
        stats['n'] += 1
        stats['correct'] += record['correct']
        if stats['n'] % 25 == 0:
            print(f"  {stats['n']}/{len(subset)}  acc={stats['correct']/stats['n']:.4f}")

with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    list(pool.map(run_one, subset.iterrows()))

jev = pd.DataFrame(jev_rows)
elapsed = time.perf_counter() - started_all
tokens = jev['input_tokens'].sum()
print(f"\nJev: {len(jev)} tickets, acc={jev['correct'].mean():.4f}, failed={stats['failed']}")
print(f"{elapsed:.1f}s wall, median {jev['latency_ms'].median():.0f}ms/ticket")
print(f"{tokens:,} input tokens = ${tokens/1e6*0.042:.4f}")
