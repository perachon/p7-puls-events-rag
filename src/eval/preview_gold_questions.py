from __future__ import annotations

import json
from pathlib import Path

from src.rag.chain import answer_question

GOLD_PATH = Path("data/eval/qa_gold.jsonl")


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path.resolve()}")

    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue  # ignore empty lines
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON on line {i} in {path}.\n"
                f"Line content: {line[:200]}"
            ) from e

    return rows


def main():
    rows = load_jsonl(GOLD_PATH)

    for row in rows:
        print("\n" + "=" * 100)
        print("ID:", row.get("id"))
        print("QUESTION:", row.get("question"))

        result = answer_question(row["question"])

        print("\nANSWER:\n", result.answer)
        print("\nSOURCES COUNT:", len(result.sources))
        print("\nSOURCES (uids):")
        if not result.sources:
            print("  (aucune source)")
        else:
            for s in result.sources:
                md = s.get("metadata") or {}
                print(f"  - {md.get('uid')} | {md.get('first_begin_dt')} | {md.get('location_city')}")


if __name__ == "__main__":
    main()
