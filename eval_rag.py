"""Runs real embeddings; evaluates Recall@4 and forbidden document leakage."""
import json
from pathlib import Path
from rag import search_chunks

cases = [
    json.loads(line)
    for line in Path("eval_cases.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
positive, positive_hits, leakage = 0, 0, 0
for c in cases:
    docs = search_chunks(c["question"], c["tenant"], c["groups"], top_k=4)
    ids = {doc["id"] for doc in docs}
    expected = c.get("expected_doc_id")
    if expected:
        positive += 1
        positive_hits += expected in ids
    bad = set(c.get("forbidden_doc_ids", [])) & ids
    leakage += bool(bad)
    print("Question:", c["question"], "Returned:", sorted(ids),
          "Expected:", expected, "Forbidden found:", sorted(bad))
print(f"Positive Recall@4: {positive_hits}/{positive}")
print(f"Permission leakage cases: {leakage}/{len(cases)}")
if positive_hits != positive or leakage:
    raise SystemExit(1)
