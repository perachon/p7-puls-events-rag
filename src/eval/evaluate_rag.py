from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Any

from src.rag.chain import answer_question

GOLD_PATH = Path("data/eval/qa_gold.jsonl")
OUT_DIR = Path("data/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON on line {i}: {line[:200]}") from e
    return rows


def extract_pred_uids(result) -> Set[str]:
    uids = set()
    for s in (result.sources or []):
        md = (s.get("metadata") or {})
        uid = md.get("uid")
        if uid is not None and str(uid).strip():
            uids.add(str(uid).strip())
    return uids


@dataclass
class RowScore:
    exact_match: bool
    precision: float
    recall: float
    f1: float
    verdict: str  # correct / partial / incorrect


def score_uids(expected: Set[str], predicted: Set[str]) -> RowScore:
    # Cas "no result attendu"
    if len(expected) == 0:
        if len(predicted) == 0:
            return RowScore(True, 1.0, 1.0, 1.0, "correct")
        else:
            return RowScore(False, 0.0, 0.0, 0.0, "incorrect")

    inter = expected & predicted
    precision = len(inter) / len(predicted) if predicted else 0.0
    recall = len(inter) / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact = (expected == predicted)

    if exact:
        verdict = "correct"
    elif len(inter) > 0:
        verdict = "partial"
    else:
        verdict = "incorrect"

    return RowScore(exact, precision, recall, f1, verdict)


def main():
    gold = load_jsonl(GOLD_PATH)

    results = []
    totals = {"correct": 0, "partial": 0, "incorrect": 0}

    for row in gold:
        qid = row.get("id")
        question = row.get("question", "")
        expected_uids = {str(x).strip() for x in (row.get("expected_uids") or []) if str(x).strip()}

        rag = answer_question(question)
        pred_uids = extract_pred_uids(rag)

        s = score_uids(expected_uids, pred_uids)
        totals[s.verdict] += 1

        results.append(
            {
                "id": qid,
                "question": question,
                "expected_uids": sorted(expected_uids),
                "predicted_uids": sorted(pred_uids),
                "exact_match": s.exact_match,
                "precision": round(s.precision, 4),
                "recall": round(s.recall, 4),
                "f1": round(s.f1, 4),
                "verdict": s.verdict,
                "answer": rag.answer,
            }
        )

    n = len(results) if results else 1
    summary = {
        "n": len(results),
        "accuracy_correct": round(totals["correct"] / n, 4),
        "rate_partial": round(totals["partial"] / n, 4),
        "rate_incorrect": round(totals["incorrect"] / n, 4),
        "counts": totals,
    }

    # Ecrit un rapport JSON
    out_json = OUT_DIR / "eval_report.json"
    out_json.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Ecrit un CSV simple (lisible rapidement)
    out_csv = OUT_DIR / "eval_report.csv"
    header = ["id", "verdict", "exact_match", "precision", "recall", "f1", "expected_uids", "predicted_uids", "question"]
    lines = [",".join(header)]
    for r in results:
        lines.append(
            ",".join(
                [
                    str(r["id"]),
                    r["verdict"],
                    str(r["exact_match"]),
                    str(r["precision"]),
                    str(r["recall"]),
                    str(r["f1"]),
                    '"' + " ".join(r["expected_uids"]) + '"',
                    '"' + " ".join(r["predicted_uids"]) + '"',
                    '"' + r["question"].replace('"', "'") + '"',
                ]
            )
        )
    out_csv.write_text("\n".join(lines), encoding="utf-8")

    print("=== EVAL SUMMARY ===")
    print(summary)
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
