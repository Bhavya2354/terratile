# terratile

An offline land-use classifier service for satellite tiles, built as a take-home for GalaxEye (Backend Engineer, ML Systems). `POST` a tile and the service classifies it on CPU, stores the result, and lets an analyst query or correct it.

- **[DESIGN.md](DESIGN.md):** Part 1, the design note
- **[ANSWERS.md](ANSWERS.md):** Part 3, the problem-solving answers
- The code below is Part 2, the working slice

## What's built

| Endpoint | Purpose |
|---|---|
| `POST /tiles` | Upload a tile (multipart `file`, optional `source`). The service classifies and stores it and returns the row. `201` means new; `200` means the tile was already stored for this model version |
| `GET /tiles` | Query with filters `label`, `status`, `source`, `model_version`, `min_confidence`, `max_confidence`, `since`, `until`, plus `limit`/`offset` |
| `GET /tiles/{id}` | Get one prediction |
| `POST /tiles/{id}/review` | Analyst correction: `{"label": "River"}` |
| `GET /stats` | Counts by label and status, plus how often reviewers agreed with the model |
| `GET /health` | Loaded model version and review threshold |

**Model:** frozen ImageNet ResNet-18 features plus a logistic-regression head trained on `candidate_tiles` (1,050 tiles, 7 classes). The review threshold is picked from cross-validation and stored with the model.

**Stubbed or skipped:** auth, georeferencing, bulk ingestion (a queue and batch worker), a review UI, a model registry, drift monitoring, and the golden-set self-test. See DESIGN.md.

## Run it

Requires Python 3.11. Every step except `pip install` and `fetch_weights` runs offline.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.fetch_weights   # one-time online step: saves ResNet-18 weights to models/

DATASET=path/to/be-mlsys-assignment-dataset
python -m scripts.train --dataset $DATASET      # ~1 min on CPU → models/head.joblib, models/model.json
python -m scripts.evaluate --dataset $DATASET   # scores eval_set → reports/eval.json

uvicorn app.main:app --port 8765
```

Try it:

```bash
curl -F "file=@$DATASET/eval_set/tile_001.png" -F source=demo http://127.0.0.1:8765/tiles
curl "http://127.0.0.1:8765/tiles?status=needs_review"
curl -X POST -H "Content-Type: application/json" -d '{"label":"River"}' http://127.0.0.1:8765/tiles/1/review
curl http://127.0.0.1:8765/stats

python -m scripts.ingest $DATASET/eval_set --url http://127.0.0.1:8765 --source eval_set   # send a whole folder
```

Interactive API docs are served at `http://127.0.0.1:8765/docs`.

To run on an air-gapped machine:
1. On a connected machine, run `pip download -r requirements.txt -d wheels/`.
2. Copy `wheels/`, `models/` and the repo across.
3. On the air-gapped machine, run `pip install --no-index --find-links wheels/ -r requirements.txt`.

Configuration is through environment variables: `TERRATILE_MODEL_DIR`, `TERRATILE_DATA_DIR`, `TERRATILE_AUDIT_RATE` (default `0.02`) and `TERRATILE_MAX_UPLOAD_BYTES`.

## Tests

```bash
pytest -q
```

The API tests swap in a fake classifier, so they need no model weights. They cover storing a result, idempotent re-uploads, both review triggers (low confidence and unexpected size), audit sampling, rejecting bad uploads, query filters and analyst review.

## Results

Eval set: 210 tiles, 30 per class, never used for training or threshold selection. Full report: [reports/eval.json](reports/eval.json).

| Metric | Value |
|---|---|
| Accuracy | **93.3%** |
| Review threshold (from 5-fold CV, target 95%) | 0.51 |
| Auto-accepted | 98.1%, of which 94.2% correct |
| Sent to review | 1.9%, of which 50% correct |
| Latency | ~20 ms/tile on CPU; 210 tiles through the API in ~8 s |

Forest, Residential and SeaLake are near-perfect. Most errors are River, Highway and AnnualCrop mistaken for each other, and **12 of the 14 mistakes are confident**. That's why the service also audits a random sample of accepted tiles instead of trusting the threshold alone (DESIGN.md, ANSWERS.md Q4).

## Layout

```
app/
  main.py         FastAPI app, upload → classify → route → store, query and review endpoints
  classifier.py   ResNet-18 feature extractor + head, loaded from local files only
  db.py           SQLite schema and queries
  config.py       settings from environment
scripts/
  fetch_weights.py  one-time download of backbone weights
  train.py          fit head, pick review threshold from cross-validation
  evaluate.py       score eval_set, write reports/eval.json
  ingest.py         POST a folder of tiles to the running service
tests/test_api.py
```

Dataset: EuroSAT subset (Helber et al.), Sentinel-2 imagery (Copernicus), MIT licensed. It is not included in this repo.
