from __future__ import annotations

import time
from statistics import mean, median

from src.indexing.faiss_store import load_vectorstore


QUERIES = [
    "conférence climat",
    "exposition campus",
    "association étudiante",
    "santé mentale",
    "atelier intelligence artificielle",
]


def main():
    vs = load_vectorstore()

    # warm-up (important)
    vs.similarity_search("warmup", k=5)

    times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        _ = vs.similarity_search(q, k=5)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print("Search latency (ms):", [round(x, 2) for x in times])
    print("Mean (ms):", round(mean(times), 2))
    print("Median (ms):", round(median(times), 2))


if __name__ == "__main__":
    main()
