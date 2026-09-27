# Part 3 — Problem-solving

## 1. The classifier is wrong about 30% of the time. What do you do, and how do you decide whether it's "good enough"?

First I'd work out *where* it's wrong before deciding what to do.
- **Per-class confusion.** Is the 30% spread evenly, or concentrated in two classes that look alike? In my eval, errors cluster in River, Highway and AnnualCrop. Merging or relabelling ambiguous classes can matter more than a better model.
- **Offline vs deployed.** My model scores 93% offline. If it's 70% in the field, that's distribution shift or a pipeline bug, not a modelling problem, and I'd debug it as in Q3.

"Good enough" depends on the decision the label feeds, not on a single accuracy number. I'd compare against:
- **Baselines:** random guessing is 14% for 7 classes, and there's also whatever the analyst does today without the model.
- **Accuracy at a given coverage:** sort tiles by confidence and ask something like "the top 60% are 95% correct". Even a model that is wrong 30% of the time overall saves real work if confidence separates right from wrong: auto-accept the confident part and send the rest to people.
- **The cost of each error:** a missed Industrial site might be expensive, while mixing up River and Highway might not matter.

The model isn't useful if its confidence doesn't separate right from wrong *and* its errors fall on the class the decision depends on.

Fixes, cheapest first:
1. Check preprocessing matches training (channel order, scaling, size).
2. Tune the per-class threshold.
3. Retrain on the corrections from the review queue.
4. Use a backbone pretrained on satellite imagery.
5. Add an "other" class.

## 2. The service runs offline with no one watching. A month later, how do you know it still works correctly?

Correctness needs ground truth, so I'd combine signals that need no labels with a small, steady flow of labels.

- **Liveness (no labels needed):** tiles per day, error rate, latency. If input stops arriving or 400s spike, something upstream broke.
- **Input drift:** tile sizes (`unexpected_size`), plus per-channel brightness and texture statistics compared with the training set. A new sensor, a season change, or haze and cloud all show up here first.
- **Output drift:** the label distribution, mean confidence and share of `needs_review`, compared with the first week after deployment. If "SeaLake" suddenly makes up 40% of tiles, something is off.
- **Golden-set self-test:** ship the 210 labelled eval tiles with the service and re-classify them on a schedule. The accuracy has to match release day (93.3%). This catches corrupted weights, library changes and preprocessing regressions.
- **Audit sample:** about 2% of *accepted* tiles are routed to review regardless of confidence. `reviewed_agreement` in `/stats` then gives an unbiased running estimate of real accuracy, including the confident mistakes that drift checks miss.

Since nobody watches it live, the service should write these checks to a local health report (and expose them in `/stats`) with clear pass/fail lines, so whoever visits next month sees problems immediately.

The audit sample and `/stats` are built. The golden-set self-test and the drift statistics are not implemented yet.

## 3. Tiles come in fine, but the stored results look wrong. How do you find the cause?

1. **Make "wrong" concrete.** Which rows, since when, which `model_version`, which classes? Is it everything or one class? Use `GET /tiles?since=…&model_version=…` and `/stats`. A sudden change points to a deploy or an input change. A slow one points to drift.
2. **Model or pipeline?** Take a handful of bad rows, load the stored raw tile via `stored_path`, and run the classifier on it by hand.
   - A different answer than what's stored means the problem is in the service path: the wrong model loaded, preprocessing differs, or rows were mixed up.
   - The same answer means the model really does see it that way, so continue to step 3.
3. **Check the model itself.** Run `scripts/evaluate.py` against the deployed `models/` directory. About 93% means the artifacts are fine. Much lower means the weights, the class order between `head.joblib` and `model.json`, or a library version changed.
4. **Look at the inputs.** Open the images. Common culprits are BGR vs RGB, 16-bit tiles scaled wrong, a different resolution or sensor, and clouds. Compare channel statistics with the training tiles.
5. **Check the storage and query layer.** Are the labels right in the table but wrong in the API, for example through review overrides, time zones in `since`/`until`, or reading another version's rows?
6. **Fix it.** Reproduce the bug in a test and fix it. Raw tiles are stored, so reprocessing the affected rows under a new model version is cheap and keeps the old rows for comparison.

## 4. What's the weakest part of your design, and what breaks it first?

**The weakest part is trusting confidence.** 12 of the 14 errors on the eval set came with confidence above the review threshold, so most mistakes land in `accepted` without any flag. The 2% audit catches them statistically, but slowly: about 4 reviews per 210 tiles, so it takes a while before an accuracy drop is visible.

**Distribution shift breaks it first.** Tiles from another sensor, region, season or processing chain (SAR, 16-bit, cloud) get forced into one of 7 classes, often confidently. The only guard is the size check, which doesn't look at content. The fixes are:
- an out-of-distribution score on the embedding, such as distance to the nearest class centroid in training data, routing far-away tiles to review;
- an explicit "other" class;
- the golden-set and drift checks from Q2.

**Second: a bulk load.** A whole scene arriving at once (thousands of tiles) hits a single process that does CPU inference synchronously and writes to SQLite. It will slow down long before it breaks, but the fix is a queue and a batch worker in front of the same storage.
