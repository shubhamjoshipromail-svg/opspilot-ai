"""## 3. API key

Typed here rather than stored in the notebook, so the key never lands in the
.ipynb file or in Drive."""

from getpass import getpass

TYPESAFE_API_KEY = getpass('TypeSafe API key: ').strip()
assert TYPESAFE_API_KEY and not TYPESAFE_API_KEY.startswith('http'), 'That looks like a URL, not a key'
print(f'key accepted ({len(TYPESAFE_API_KEY)} chars)')
