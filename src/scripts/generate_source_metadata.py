import hashlib
from datetime import datetime

# Percorso del file grezzo
file_path = "data/raw/32024R1689_consolidated_2026-07-27.mhtml"

# 1. Calcolo dell'hash SHA-256
sha256_hash = hashlib.sha256()
with open(file_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
file_hash = sha256_hash.hexdigest()

# 2. Contenuto del file SOURCE.md
source_content = f"""# Source Metadata: AI Act

- Document: Regulation (EU) 2024/1689 (AI Act)
- Official CELEX: `02024R1689-20260727`
- Source URL: https://eur-lex.europa.eu/legal-content/IT/TXT/?uri=CELEX%3A02024R1689-20260727
- Download Date: {datetime.today().strftime("%Y-%m-%d")}
- Raw File Name: `32024R1689_consolidated_2026-07-27.mhtml`
- SHA-256 Hash: `{file_hash}`
"""

# 3. Scrittura del file SOURCE.md in data/raw/
source_path = "data/raw/SOURCE.md"
with open(source_path, "w", encoding="utf-8") as f:
    f.write(source_content)

print(f"File SOURCE.md generato correttamente in {source_path}")
print(f"SHA-256: {file_hash}")
