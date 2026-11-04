"""Valutazione del retrieval per il corpus dell'AI Act.

Esegue per default ENTRAMBI gli split (`dev` e `test`) su ENTRAMBE le
collezioni (`paragraphs_v1` e `articles_v1`), e produce:

  evals/results/eval_paragraphs_dev.json
  evals/results/eval_articles_dev.json
  evals/results/retrieval_comparison_dev.png
  evals/results/eval_paragraphs_test.json
  evals/results/eval_articles_test.json
  evals/results/retrieval_comparison_test.png

Se passi `--split dev` o `--split test`, esegue solo quello split.
Se passi `--collection paragraphs_v1`, esegue solo quella collezione.

Normalizzazione ID:
- `paragraphs_v1` valuta il recupero del paragrafo esatto (es. `art_6.p1`).
- `articles_v1` valuta il recupero dell'articolo contenente il paragrafo
  rilevante (es. `art_6`), perché il ground truth è a livello di paragrafo
  ma la collezione contiene un punto per articolo.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from copilot.retrieval.embedders import Embedders
from copilot.retrieval.search import METHODS, Method, Retriever
from copilot.retrieval.store import QdrantStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "data" / "eval" / "cases.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

DEFAULT_COLLECTIONS = ("paragraphs_v1", "articles_v1")
DEFAULT_SPLITS = ("dev", "test")
DEFAULT_K = 5
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_SPLIT = "dev"
CHART_NAME = "retrieval_comparison"  # + _<split>.png

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 42

_COLLECTION_LABELS = {
    "paragraphs_v1": "paragraphs",
    "articles_v1": "articles",
}

# Pattern per suffissi:
# - `.sN` = split artificiale del chunker (va sempre rimosso)
# - `.pN` = paragrafo (da rimuovere solo per la collezione articoli)
_SPLIT_SUFFIX_RE = re.compile(r"\.s\d+$")
_PARAGRAPH_SUFFIX_RE = re.compile(r"\.p\d+$")


def _normalize_id(raw_id: str, collection: str) -> str:
    """Normalizza un ID (ground truth o hit) per il matching.

    - Rimuove sempre il suffisso di split `.sN` (artefatto del chunker).
    - Per `articles_v1`, rimuove anche `.pN` per ridurre l'ID a livello
      di articolo (es. `art_6.p1` → `art_6`).
    - Per `paragraphs_v1`, mantiene `.pN` (match a livello di paragrafo).
    """
    base = _SPLIT_SUFFIX_RE.sub("", raw_id)
    if collection == "articles_v1":
        base = _PARAGRAPH_SUFFIX_RE.sub("", base)
    return base


def recall_at_k(hit_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(hit_ids[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(hit_ids: list[str], relevant: set[str], k: int) -> float:
    if not hit_ids:
        return 0.0
    top = set(hit_ids[:k])
    return len(top & relevant) / min(k, len(hit_ids))


def reciprocal_rank(hit_ids: list[str], relevant: set[str]) -> float:
    for rank, hit_id in enumerate(hit_ids, start=1):
        if hit_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hit_ids: list[str], relevant: set[str], k: int) -> float:
    top_k_hits = hit_ids[:k]
    if not top_k_hits or not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, h in enumerate(top_k_hits) if h in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean

    # RNG deterministico per il bootstrap: la riproducibilità è richiesta
    # qui, non un segreto. S311 è soppresso con motivazione.
    rng = random.Random(seed)  # noqa: S311
    idx_pool = list(range(n))
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = [rng.choice(idx_pool) for _ in range(n)]
        samples[i] = sum(values[j] for j in idx) / n

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return mean, lo, hi


def load_cases(path: Path, *, split: str | None = None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"File dei casi non trovato: {path}")
    queries: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            relevant = case.get("relevant_ids") or []
            if not relevant:
                continue
            if split is not None and case.get("split") != split:
                continue
            queries.append(
                {
                    "id": case.get("id"),
                    "query": case["query"],
                    "relevant_ids": list(relevant),
                }
            )
    return queries


def evaluate_method(
    retriever: Retriever,
    *,
    method: Method,
    collection: str,
    k: int,
    queries: list[dict],
) -> dict:
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for q in queries:
        # Ground truth normalizzato per la collezione corrente.
        relevant_parent = {_normalize_id(rid, collection) for rid in q.get("relevant_ids", [])}
        hits = retriever.search(q["query"], collection=collection, method=method, limit=k)
        # Hit normalizzati allo stesso modo del ground truth.
        hit_ids = [_normalize_id(h["id"], collection) for h in hits if h.get("id")]
        recalls.append(recall_at_k(hit_ids, relevant_parent, k))
        precisions.append(precision_at_k(hit_ids, relevant_parent, k))
        reciprocal_ranks.append(reciprocal_rank(hit_ids, relevant_parent))
        ndcgs.append(ndcg_at_k(hit_ids, relevant_parent, k))

    n = len(queries) or 1
    recall_mean, recall_lo, recall_hi = bootstrap_mean_ci(recalls)
    ndcg_mean, ndcg_lo, ndcg_hi = bootstrap_mean_ci(ndcgs)

    return {
        "method": method,
        "collection": collection,
        "k": k,
        "n_queries": len(queries),
        "mean_recall": recall_mean,
        "recall_ci_lo": recall_lo,
        "recall_ci_hi": recall_hi,
        "mean_precision": sum(precisions) / n,
        "mrr": sum(reciprocal_ranks) / n,
        "mean_ndcg": ndcg_mean,
        "ndcg_ci_lo": ndcg_lo,
        "ndcg_ci_hi": ndcg_hi,
    }


def print_summary(results_by_collection: dict[str, list[dict]], split: str) -> None:
    for collection, results in results_by_collection.items():
        print("\n" + "=" * 90)
        print(f"RETRIEVAL EVAL — SUMMARY ({collection}, split={split})")
        print("=" * 90)
        header = (
            f"{'method':<16} {'recall@k [95% CI]':>26} {'prec@k':>10} "
            f"{'MRR':>8} {'nDCG@k [95% CI]':>26}"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            recall_cell = (
                f"{r['mean_recall']:.3f} [{r['recall_ci_lo']:.2f},{r['recall_ci_hi']:.2f}]"
            )
            ndcg_cell = f"{r['mean_ndcg']:.3f} [{r['ndcg_ci_lo']:.2f},{r['ndcg_ci_hi']:.2f}]"
            print(
                f"{r['method']:<16} {recall_cell:>26} "
                f"{r['mean_precision']:>10.3f} {r['mrr']:>8.3f} {ndcg_cell:>26}"
            )
        print("-" * len(header))
        print(
            f"queries: {results[0]['n_queries']} | k: {results[0]['k']} | collection: {collection}"
        )


def plot_comparison(
    results_by_collection: dict[str, list[dict]],
    *,
    out_path: Path,
    split: str,
) -> None:
    """Grafico a barre raggruppate: 4 metriche × 2 collezioni × N metodi.

    Layout: 1 riga × 4 colonne (recall, precision, MRR, nDCG).
    In ogni subplot, barre raggruppate per metodo, colorate per collezione.

    Le metriche `recall` e `nDCG` hanno un CI bootstrap al 95% calcolato
    in `evaluate_method`, quindi vengono plottate con barre d'errore
    asimmetriche (da `*_ci_lo` a `*_ci_hi`). `precision` e `mrr` non
    hanno CI nel codice attuale, quindi restano senza barre d'errore.

    Nota: i valori non sono direttamente confrontabili tra collezioni,
    perché `paragraphs_v1` misura il recupero del paragrafo esatto,
    mentre `articles_v1` misura il recupero dell'articolo contenente
    il paragrafo rilevante (task più facile).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in next(iter(results_by_collection.values()))]
    collections = list(results_by_collection.keys())
    metrics = [
        ("mean_recall", "Recall@k", "recall_ci_lo", "recall_ci_hi"),
        ("mean_precision", "Precision@k", None, None),
        ("mrr", "MRR", None, None),
        ("mean_ndcg", "nDCG@k", "ndcg_ci_lo", "ndcg_ci_hi"),
    ]

    # Palette: una tonalità per collezione (blu = paragraphs, arancio = articles).
    palette = {
        "paragraphs_v1": "#4a9eff",
        "articles_v1": "#ff8c42",
    }

    n_metrics = len(metrics)
    n_methods = len(methods)
    n_collections = len(collections)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 6.0), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_methods)
    width = 0.8 / n_collections

    for ax, (metric_key, metric_label, ci_lo_key, ci_hi_key) in zip(axes, metrics):
        for i, collection in enumerate(collections):
            values = [r[metric_key] for r in results_by_collection[collection]]
            offset = (i - (n_collections - 1) / 2) * width

            # Barre d'errore asimmetriche, solo se il metodo ha il CI.
            yerr = None
            if ci_lo_key is not None and ci_hi_key is not None:
                lo = np.array([r[ci_lo_key] for r in results_by_collection[collection]])
                hi = np.array([r[ci_hi_key] for r in results_by_collection[collection]])
                # matplotlib vuole (distanza dal valore al limite inferiore,
                # distanza dal valore al limite superiore). Il primo è negativo,
                # il secondo positivo; usiamo abs() per evitare sorprese.
                yerr = np.vstack([np.abs(np.array(values) - lo), np.abs(hi - np.array(values))])

            bars = ax.bar(
                x + offset,
                values,
                width,
                label=_COLLECTION_LABELS.get(collection, collection),
                color=palette.get(collection, "#888888"),
                edgecolor="black",
                linewidth=0.6,
                yerr=yerr,
                capsize=3,  # trattini orizzontali in cima e in fondo alla barra d'errore
                error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
            )
            # Etichette numeriche sopra le barre.
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(metric_label)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    # Legenda unica in alto.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=n_collections,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        f"Retrieval eval — paragraphs_v1 & articles_v1 "
        f"(split={split}, k={results_by_collection[collections[0]][0]['k']})",
        y=1.08,
        fontsize=12,
    )
    # Nota sulla non-comparabilità + nota sul CI.
    fig.text(
        0.5,
        -0.02,
        "Note: paragraphs_v1 evaluates retrieval of the exact paragraph; "
        "articles_v1 evaluates retrieval of the article containing the relevant "
        "paragraph (an easier task). The values are not directly comparable. "
        "Error bars on Recall and nDCG show the 95% bootstrap confidence interval "
        "(2000 resamples, seed=42). Precision and MRR have no CI in the current code.",
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color="#555555",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_split(
    *,
    split: str,
    collections: list[str],
    methods: list[Method],
    k: int,
    retriever: Retriever,
    out_dir: Path,
    make_plot: bool,
    trace_id: str,
    timestamp: str,
) -> list[Path]:
    """Esegue un singolo split (dev o test). Ritorna i file scritti."""
    print(f"\n########## SPLIT: {split} ##########")
    queries = load_cases(CASES_PATH, split=split)
    if not queries:
        print(
            f"ATTENZIONE: nessun caso per split={split!r}. Salto.",
            file=sys.stderr,
        )
        return []

    results_by_collection: dict[str, list[dict]] = {}
    for collection in collections:
        print(f"\n>>> Valutazione {collection} (split={split}, {len(queries)} query, k={k})")
        results = [
            evaluate_method(
                retriever,
                method=method,
                collection=collection,
                k=k,
                queries=queries,
            )
            for method in methods
        ]
        results_by_collection[collection] = results

    print_summary(results_by_collection, split)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Un JSON per (collezione, split).
    for collection, results in results_by_collection.items():
        label = _COLLECTION_LABELS.get(collection, collection)
        out = out_dir / f"eval_{label}_{split}.json"
        payload = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "collection": collection,
            "k": k,
            "split": split,
            "n_queries": len(queries),
            "methods": methods,
            "results": results,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out)

    # Un PNG per split (solo se ≥ 2 collezioni).
    if make_plot and len(results_by_collection) >= 2:
        chart_path = out_dir / f"{CHART_NAME}_{split}.png"
        plot_comparison(results_by_collection, out_path=chart_path, split=split)
        print(f"Grafico salvato -> {chart_path}")
        written.append(chart_path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument(
        "--split",
        default="all",
        choices=["dev", "test", "all"],
        help="Split da valutare (default: all, cioè dev + test).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        choices=["articles_v1", "paragraphs_v1"],
        help="Se specificato, valuta solo questa collezione (default: entrambe).",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Non generare i grafici PNG.",
    )
    parser.add_argument(
        "--trace-id",
        default=None,
        help="Trace id per correlazione (default: uuid generato).",
    )
    args = parser.parse_args(argv)

    methods: list[Method] = []
    for m in args.methods.split(","):
        m = m.strip()
        if m not in METHODS:
            parser.error(f"Metodo sconosciuto: {m}. Scegli da {METHODS}.")
        methods.append(m)  # type: ignore[arg-type]

    collections = [args.collection] if args.collection else list(DEFAULT_COLLECTIONS)

    # Quali split eseguire.
    if args.split == "all":
        splits = list(DEFAULT_SPLITS)
    else:
        splits = [args.split]

    # Inizializza retriever e embedders UNA volta sola (riuso tra split).
    embedders = Embedders()
    store = QdrantStore(url=args.qdrant_url)
    retriever = Retriever(embedders=embedders, store=store)

    trace_id = args.trace_id or f"eval:{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(UTC).isoformat()

    all_written: list[Path] = []
    for split in splits:
        written = run_split(
            split=split,
            collections=collections,
            methods=methods,
            k=args.k,
            retriever=retriever,
            out_dir=args.out_dir,
            make_plot=not args.no_plot,
            trace_id=trace_id,
            timestamp=timestamp,
        )
        all_written.extend(written)

    print("\n===== RIEPILOGO FILE SCRITTI =====")
    for p in all_written:
        print(f"  - {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
