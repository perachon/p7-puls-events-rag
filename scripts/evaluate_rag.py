from __future__ import annotations

import json
from pathlib import Path

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

from src.rag.chain import answer_question


DATASET_PATH = Path("data/eval/eval_set.jsonl")


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")

    rows = load_jsonl(DATASET_PATH)

    dataset = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for r in rows:
        q = r["question"]
        gt = r["reference_answer"]

        rag_res = answer_question(q)  # RAGResult(answer, sources)

        contexts = []
        for s in rag_res.sources:
            excerpt = (s.get("excerpt") or "").strip()
            if excerpt:
                contexts.append(excerpt)

        dataset["question"].append(q)
        dataset["answer"].append(rag_res.answer)
        dataset["contexts"].append(contexts)
        dataset["ground_truth"].append(gt)

    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
    )

    print("\n=== RAGAS RESULTS ===")
    print(result)

    out_path = Path("data/eval/ragas_report.json")
    out_path.write_text(result.to_pandas().to_json(orient="records", force_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
