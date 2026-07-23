#!/bin/bash
set -euo pipefail

TERMINUSDB_URL="http://127.0.0.1:7373"
AUTH="admin:root"
ORG="admin"
DP="buy-abt"
DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"

echo "=== Converting CSV to NDJSON ==="

python3 -c '
import csv, json, sys

for src, dst, typ in [
    ("Abt.csv", "abt.ndjson", "Abt"),
    ("Buy.csv", "buy.ndjson", "Buy"),
]:
    with open(f"{sys.argv[1]}/{src}", newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        with open(f"{sys.argv[1]}/{dst}", "w", encoding="utf-8") as out:
            for row in reader:
                doc = {"@type": typ}
                for k, v in row.items():
                    if v:
                        doc[k] = v
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  {src} -> {dst}")
' "$DATA_DIR"

echo ""
echo "=== Deleting existing data product (if any) ==="
curl -s -X DELETE -u "$AUTH" "$TERMINUSDB_URL/api/db/$ORG/$DP" -w "\nHTTP:%{http_code}\n" || true

echo ""
echo "=== Creating new empty data product ==="
curl -s -X POST -u "$AUTH" "$TERMINUSDB_URL/api/db/$ORG/$DP" \
  -H "Content-Type: application/json" \
  -d '{"label":"buy-abt","comment":"Abt-Buy entity resolution dataset","schema":true}' \
  -w "\nHTTP:%{http_code}\n"

AUTH_PARAMS="author=admin&message=load_abt_buy"

echo ""
echo "=== Loading abt.ndjson ==="
curl -s -X POST -u "$AUTH" "$TERMINUSDB_URL/api/document/$ORG/$DP?raw_json=true&$AUTH_PARAMS" \
  -H "Content-Type: application/json" \
  --data-binary @"$DATA_DIR/abt.ndjson" \
  -w "\nHTTP:%{http_code}\n"

echo ""
echo "=== Loading buy.ndjson ==="
curl -s -X POST -u "$AUTH" "$TERMINUSDB_URL/api/document/$ORG/$DP?raw_json=true&$AUTH_PARAMS" \
  -H "Content-Type: application/json" \
  --data-binary @"$DATA_DIR/buy.ndjson" \
  -w "\nHTTP:%{http_code}\n"

echo ""
echo "=== Checking document count ==="
curl -s -u "$AUTH" "$TERMINUSDB_URL/api/document/$ORG/$DP?count=1" \
  -w "\nHTTP:%{http_code}\n"

echo ""
echo "=== Done ==="
