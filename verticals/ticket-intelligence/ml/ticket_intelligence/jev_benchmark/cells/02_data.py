"""## 2. Load the fixed split

Searches Drive for `ticket_splits.zip` (uploaded from the local repo) and
extracts the split CSVs. Using the saved splits matters: they are the exact
rows the published ModernBERT metrics were measured on.

Set `SPLIT` to `val` for the published head-to-head (3,562 rows), or `test`
(3,563) for a final locked evaluation."""

SPLIT = 'val'

def find_split() -> Path:
    root = Path('/content/drive/MyDrive')
    direct = list(root.rglob(f'{SPLIT}.csv'))
    for candidate in direct:
        if 'opspilot' in str(candidate).lower() or 'ticket' in str(candidate).lower():
            return candidate
    for archive in root.rglob('ticket_splits.zip'):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(WORK)
        found = list(WORK.rglob(f'{SPLIT}.csv'))
        if found:
            return found[0]
    if direct:
        return direct[0]
    raise FileNotFoundError(f'No {SPLIT}.csv or ticket_splits.zip found under MyDrive')

TEST_PATH = find_split()
df = pd.read_csv(TEST_PATH)
print(f'{TEST_PATH}  ->  {len(df)} rows')

LABEL_COL = 'category' if 'category' in df.columns else 'true_category'
ID_COL = 'ticket_id' if 'ticket_id' in df.columns else 'external_id'

def build_text(row) -> str:
    subject = str(row.get('subject') or '').strip()
    body = str(row.get('body') or '').strip()
    if not subject and not body:
        return str(row.get('customer_message') or '').strip()
    return f'{subject}\n\n{body}'.strip()

df['_text'] = [build_text(r) for _, r in df.iterrows()]
print(df[LABEL_COL].value_counts())
