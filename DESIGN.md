# terratile — design note

An offline service that takes satellite tiles, classifies each one by land use, stores the result and lets an analyst query it. Everything runs on one CPU machine without network access.

## How a tile flows through the system

```
 tile source ──► POST /tiles ──► validate ──► sha256 ──► already seen by this model? ──yes──► return stored row
                                                               │ no
                                                               ▼
                                   classifier (frozen ResNet-18 features → logistic-regression head)
                                                               │  probabilities for all 7 classes
                                                               ▼
                                   routing: accepted | needs_review (low_confidence / unexpected_size / audit_sample)
                                                               │
                              ┌────────────────────────────────┴───────────────────────────────┐
                              ▼                                                                ▼
             raw tile on disk (content-addressed: data/tiles/ab/abcd….png)       one row in SQLite `predictions`
                                                                                               │
                                              analyst ◄── GET /tiles (filters) · GET /stats · POST /tiles/{id}/review
```

1. **Validate.** The upload is size-capped and has to decode as an image. If it doesn't, the service returns 400 and does not guess.
2. **Deduplicate.** `sha256(bytes)` together with `model_version` is a unique key. A client can retry safely, and replaying a batch doesn't add rows. When a new model version ships, the same tile gets a new row, so both results are kept side by side.
3. **Classify.** Inference takes about 20 ms per tile on CPU and runs in a thread pool so it doesn't block the event loop.
4. **Route.** The top-1 label is always stored. `status` records whether the service trusts it (see below).
5. **Persist.** The service writes the raw bytes plus one row holding the full probability vector, confidence, margin, model version, source, size, latency and timestamps.

## What I store, and why

| Stored | Why |
|---|---|
| Full probability vector, not just top-1 | I can re-threshold, compute entropy and watch for drift without re-running the model |
| `model_version` on every row | Mixed-version data can be filtered, compared and reprocessed |
| Raw tile, keyed by hash | Stored rows can be reclassified with a new model, and a human can look at the actual pixels when something looks wrong |
| `reviewed_label` next to `label`, never overwriting it | The model's original answer is what I measure accuracy against. Query results use `COALESCE(reviewed_label, label)` |
| `review_reason` | Shows why a row is in the queue, so the queue itself can be analysed |

**SQLite (WAL mode).** It's a single file with nothing to run or administer, which suits an isolated box. I'd move to Postgres/PostGIS once there are several writers or once tiles carry geo-coordinates and analysts want spatial queries.

## Decisions and trade-offs

**The model.** Options I considered:
- An ImageNet classifier used as-is: its labels don't map to land use.
- CLIP zero-shot: heavy, and weak on 64 px Sentinel-2 tiles.
- Fine-tuning a whole CNN: more accurate, but slower on CPU and easy to overfit with 1,050 images.
- Frozen ImageNet features plus a linear head: what I chose. It trains in under a minute on CPU and scores **93.3% on the 210-tile eval set**. It's also trivial to replace, because the service only depends on `predict()` and a `model.json` file. A backbone pretrained on EuroSAT or Sentinel-2 would be the obvious next step.

**Low-confidence predictions.** Options:
- Always return top-1: silently wrong.
- Reject: loses data.
- Return "unknown": hides what the model thought.
- Store top-1 and flag the row for review: what I chose.

The threshold is **not a magic number**. `train.py` picks it from 5-fold cross-validated predictions so that auto-accepted tiles reach 95% accuracy, and writes it into `model.json` so it is versioned with the model. That gave 0.51 here.

On the eval set, though, **12 of the 14 errors were confident**, mostly among River, Highway and AnnualCrop, which all look like linear strips at 64 px. A confidence gate alone doesn't catch the errors that matter. So I added two more checks:
- **Audit sample.** Roughly 2% of accepted tiles, chosen deterministically from the hash, go to review. That is the only unbiased way to measure accuracy on the tiles the service accepts on its own.
- **Unexpected size.** A tile that isn't 64×64 goes to review, as a cheap guard against inputs unlike the training data.

**Synchronous endpoint vs a queue.** At 20 ms per tile, a synchronous request is the simplest thing that works. If tiles arrive a whole scene at a time (thousands at once), I'd put a queue in front (a folder watcher or a jobs table) and add a batch worker. The endpoint contract and storage wouldn't change.

**What "querying the results" means.** I see four levels:
- (a) Look up and filter records by label, status, confidence, source, time or model version. **Built.**
- (b) Aggregates: counts and mean confidence per class. **Built** (`/stats`).
- (c) A review queue with corrections: **built**, as `status=needs_review` plus `POST /review`.
- (d) Spatial and temporal questions, such as "how much forest in this area was lost since March". **Not built:** the tiles carry no georeference, and this is what analysts ultimately want. It's the first thing I'd ask about.

**Stubbed or skipped:** authentication, georeferencing, bulk scene ingestion and tiling, a review UI, a model registry and rollout, data retention, and exporting metrics and alerts.

## Assumptions

- Tiles are 64×64 RGB Sentinel-2 crops, like EuroSAT. Each tile has one label from a fixed set of 7 classes.
- One CPU box handles modest volume, and a single process is enough.
- Tiles have no geo or time metadata. A free-text `source` field is the only provenance.
- Someone reviews the queue now and then. Without that, the review path is only a label.

## Questions I'd ask GalaxEye

1. What decision does an analyst make from a label, and what does each mistake cost? Missing Industrial may matter far more than confusing River with Highway. The answer sets per-class thresholds.
2. What's the volume and arrival pattern: a steady trickle, or a whole pass arriving at once? Is batch latency fine?
3. Do tiles come with coordinates, capture time and sensor? Is it RGB only, all 13 Sentinel-2 bands, or SAR (multi-sensor)? The model changes completely for SAR.
4. Are tiles really single-label? Mixed tiles, like a river through farmland, are exactly where this model fails.
5. Is there a person to review flagged tiles, and how often? If there isn't, I'd use a different policy, such as an "unknown" class.
6. How do model updates and results move on and off an air-gapped machine? How long should raw tiles be kept?
7. Should there be an "other / cloud / no-data" class? Right now the model has to force every tile into one of 7 classes.
