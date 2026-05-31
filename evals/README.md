# RAG Evaluation

`rag_cases.jsonl` contains a small starter set for evaluating retrieval and grounded answer quality.

Run against a live service:

```bash
python evals/run_rag_eval.py --base-url http://127.0.0.1:8000 --cases evals/rag_cases.jsonl
```

Generate candidate cases from the current knowledge base:

```bash
python evals/build_rag_cases.py \
  --base-url http://127.0.0.1:8000 \
  --output evals/rag_cases_generated.jsonl \
  --questions-per-chunk 2
```

This also writes a review sheet, for example
`evals/rag_cases_generated.review.csv`. Treat generated cases as candidates;
manually review a sample before using the metrics as formal numbers.

After reviewing the CSV, build the curated set:

```bash
python evals/apply_rag_review.py \
  --cases evals/rag_cases_generated.jsonl \
  --review-csv evals/rag_cases_generated.review.csv \
  --output evals/rag_cases_curated.jsonl
```

Metrics:

- `hit@k`: whether any expected document id appears in the top-k search results.
- `precision@k`: relevant retrieved chunks in top-k divided by k.
- `recall@k`: relevant retrieved chunks in top-k divided by all labeled relevant chunks.
- `mrr`: reciprocal rank of the first expected document.
- `keyword_accuracy`: whether the final chat answer contains all expected keywords.
- `citation_accuracy`: whether `/chat` returns at least one citation matching expected document ids.

Evaluate retrieval only at multiple K values:

```bash
python evals/run_rag_eval.py \
  --base-url http://127.0.0.1:8000 \
  --cases evals/rag_cases_curated.jsonl \
  --top-ks 1,3,5 \
  --skip-chat
```

The eval requires documents with matching ids/titles to exist in the knowledge base. If the service is not running or the KB is empty, the script reports that explicitly instead of producing a misleading score.
