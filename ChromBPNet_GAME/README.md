# `ChromBPNet_GAME/` — Predictor internals

This folder holds the GAME predictor wrapper around ChromBPNet: the Flask service, the inference core, request validation, and the supporting machinery. The files are flat; this README groups them by role.

| File | Role |
|------|------|
| `ChromBPNet_predictor_RestAPI.py` | Flask entrypoint — endpoints, the `/predict` request loop, response assembly, output sanitization |
| `chrombpnet_predict.py` | Inference orchestration — `predict_chrombpnet()`: resolve models, run them, assemble tracks, apply ranges/readout |
| `chrombpnet_utils.py` | Matcher-aware model selection, model loading, fold averaging, padding/slicing, numpy→JSON conversion |
| `schema_validation.py` | Generic GAME schema checks (`validate_request_payload`) + preprocessing (`preprocess_data`) |
| `model_validation.py` | ChromBPNet-specific checks (`model_specific_payload_validation`) + output scaling (`apply_scaling`) |
| `error_checking_functions.py` | `APIError` class hierarchy + field-level check functions |
| `config.py` | Container-aware predictor versioning + supported wire formats |
| `predictor_content_handler.py` | JSON / MessagePack request decode + response encode |
| `predictor_help_message.json` | Metadata served at `/help` |

The **inference core** and **request validation** — are documented in full below. Model weights are covered in [`models/`](models/README.md); the internal ChromBPNet library in [`chrombpnet/`](chrombpnet/README.md).

---

## Request lifecycle

The `/predict` handler runs these in order; each raises on failure and is caught by a single central error handler:

```
decode_request                     # JSON / MessagePack → dict    (predictor_content_handler.py)
        │
validate_request_payload           # generic GAME schema          (schema_validation.py)
        │
model_specific_payload_validation  # ChromBPNet rules             (model_validation.py)
        │
preprocess_data                    # flanking + spec + range bounds (schema_validation.py)
        │
predict_chrombpnet                 # inference core               (chrombpnet_predict.py)
        │
apply_scaling (per task)           # linear / log output          (model_validation.py)
        │
response assembly + sanitization   # ChromBPNet_predictor_RestAPI.py
```

---

# Inference core

## Model architecture recap

Each fold model takes a **2114 bp** one-hot sequence `(N, 2114, 4)` and emits two heads:

| Head | Shape | Meaning |
|------|-------|---------|
| `wo_bias_bpnet_logits_profile_predictions` | `(N, 1000)` | Per-base profile **logits** over the central 1000 bp |
| `wo_bias_bpnet_logcount_predictions` | `(N, 1)` | Single **log** of total counts in the window |

The final base-pair track is the count-weighted profile:

```
track = softmax(profile_logits) * exp(logcounts)      # shape (N, 1000)
```

`softmax` distributes the predicted total signal across the 1000 bp window, and `exp(logcounts)` sets its magnitude. The model reads 2114 bp but predicts the central 1000 bp; the 557 bp on each side are receptive-field flank.

## Design decisions

### 1. Always predict the full track first; apply readout/range afterward

`predict_chrombpnet` always calls the fold predictor with `is_point_readout=False`, producing full 1000 bp tracks regardless of what the request asked for. Point readout (a single mean) and prediction-range cropping are then applied as **pure post-processing** once the full track is assembled.

This keeps a single, well-tested prediction path and avoids branching the model call on readout type. The point branch inside `predict_across_folds_for_selected_matched_models` from the Log counts head of the model is never used in this Predictor. 

### 2. Fold averaging respects the requested scale

Five folds are averaged per model, and *how* they are averaged depends on the requested scale:

- **`linear`** → arithmetic mean across folds in linear space (mean of the final count-weighted predictions).
- **`log`** → geometric mean across folds: average in log space, then `exp` back to linear, i.e. `exp(mean(log(pred_per_fold)))`.

> `predict_across_folds_for_selected_matched_models` **always returns linear-scale predictions.** `scale_actual` only controls the *averaging* method, not the output scale. The final log transform (if requested) is applied separately and later, via `apply_scaling()`. This separation is deliberate: averaging happens once per fold set, output scaling happens once per task at serialization time.

### 3. Cross-assay pooling

`choose_model` returns *all* mapping rows for a cell line. For every cell line that has both an ATAC and a DNASE model (all except H1), both fold sets — 10 fold-models — are loaded and averaged together into a single mean track. The task result then carries `type_actual: ["DNASE", "ATAC"]`, and the server adds `aggregation: {"models": "mean"}` whenever `type_actual` has more than one entry. H1 (DNASE only) yields a 5-fold mean and no aggregation field.

## Matcher-aware model selection (`choose_model`)

```
requested cell_type
   │
   ├─ exact match in model_mappings.txt?  ──► use those rows   (matcher_version = "N/A")
   │
   └─ no ─► POST {cell_type_requested, cell_type_list} to  http://<matcher_ip>:<matcher_port>/match
              │
              ├─ cell_type_actual returned ─► look up rows for it ─► use them
              ├─ NULL / empty               ─► request error (no usable match)
              └─ connection failure         ─► structured error, request does not crash
```

The return signature is a 3-tuple `(models_or_error_msg, cell_type_actual_or_None, matcher_version)`. A `None` in the second slot signals the caller to record an error for that task and skip it, while still returning a valid response for the other tasks.

## Prediction ranges

Ranges are applied **after** the full track is assembled, per sequence:

```python
seq_range = prediction_ranges.get(seq_id, [])
if seq_range:
    start, end = seq_range
    cropped = raw_pred[start:end + 1]      # end is INCLUSIVE, per the API spec
    result  = mean(cropped) if point else cropped
else:
    result  = mean(raw_pred) if point else raw_pred   # full sequence
```

- `end` is **inclusive** (`[start, end]`), so the slice is `start:end+1`.
- An absent or empty range means "use the whole sequence."
- Point readout takes the mean of whatever range survived (cropped window or full track).
- Range *validation* (bounds, ordering, types) happens earlier in validation — the inference core only *applies* already-validated ranges.

## Long-sequence handling

ChromBPNet's window predicts 1000 bp from a 2114 bp input, with 557 bp of flank on each side (`(2114 − 1000) / 2 = 557`). Sequences are routed by length.

### Short path — all sequences ≤ 1000 bp

Pad each sequence to 2114 bp (`pad_sequences`, centered N-padding), one-hot encode, predict once, then slice the prediction back to the original length using the centered offset (`slice_predictions`).

### Long path — any sequence > 1000 bp

Short and long sequences are split; short ones use the path above. Each long sequence is chunked so that **every base receives exactly one prediction**:

1. Prepend a 557 bp upstream N-flank so the first real base lands at the start of the first prediction window.
2. Slide a 2114 bp window in **1000 bp steps**, tracking how far prediction has reached.
3. For a **full 2114 bp** chunk: keep all 1000 predicted bases.
4. For a **trailing partial** chunk (< 2114 bp): pad downstream with N to 2114 bp, then keep
   - `1000` bases if `len(chunk) − 557 ≥ 1000`, otherwise
   - `len(chunk) − 557` bases (only the genuinely new, non-flank bases).
5. Reassemble by concatenating each sequence's kept slices in order (`slice_predictions_longSeqs`).

Downstream padding (rather than centered padding) is used on trailing chunks so the first base of the chunk stays aligned to the start of the prediction window, keeping the per-base bookkeeping exact. The reassembled track has exactly one value per input base, identical in resolution to the short path.

## Output sanitization

Before serialization the server clamps non-finite floats (`NaN`, `±Inf`) to `±1e5` (`_sanitize_for_json`), and `apply_scaling`'s log branch clips to a small positive epsilon (`1e-10`) before `log` to avoid `log(0) = -inf`. These guard the log-scale path, where zero-signal bases would otherwise serialize as infinities.

---

# Request validation & preprocessing

## Error model

All custom errors subclass `APIError`, so a single Flask error handler catches them and emits a standardized JSON body — `{"error": [{<error_key>: <message>}, ...]}` — with the right HTTP status:

| Class | Status | `error_key` | Use |
|-------|:------:|-------------|-----|
| `BadRequestError` | 400 | `bad_prediction_request` | Malformed or schema-invalid request (missing keys, bad types, bad ranges) |
| `PredictionFailedError` | 422 | `prediction_request_failed` | Request well-formed but unfulfillable (unsupported species/type/readout, invalid bases, no model match) |
| `ServerError` | 500 | `server_error` | Unexpected backend failure |

Errors are **always serialized as JSON**, even when the client requested MessagePack.

## Generic schema validation (`validate_request_payload`)

Checks are ordered so later checks can assume earlier structure exists. Missing-key checks short-circuit (raise immediately) before any value checks run.

1. **Mandatory top-level keys:** `readout`, `prediction_tasks`, `sequences`.
2. **Mandatory per-task keys:** `name`, `type`, `cell_type`, `species` (reported with the task name, or its index as fallback).
3. **Field value/type checks:**
   - `readout` ∈ `{point, track, interaction_matrix}`, single string.
   - `name`, `cell_type`, `species` — single strings.
   - `type` — single string; accepted if `accessibility`/`expression` or starts with `binding_`, `expression_`, `conformation_`.
   - `scale` (optional) ∈ `{linear, log}` if present.
4. **Prediction ranges (if present):** sequence IDs must exactly match those in `sequences`; each value is a 2-element list of integers, non-negative, `start ≤ end`.
5. **Flanks (if present):** `upstream_seq` / `downstream_seq` must be single strings.

> These are the *generic* GAME checks. They intentionally allow request types (e.g. `expression_*`) this predictor does not serve — narrowing to what ChromBPNet supports happens in the model-specific layer below.

## ChromBPNet-specific validation (`model_specific_payload_validation`)

Runs after the generic schema passes and rejects requests this predictor cannot serve:

- **Readout:** `interaction_matrix` is rejected (only `point` and `track` are supported).
- **Species:** every task must be `homo_sapiens`.
- **Type:** every task must be `accessibility`.

Violations are collected and raised together as a `PredictionFailedError`, so the caller sees all offending tasks at once.

## Preprocessing (`preprocess_data`)

Runs after validation and produces the final sequence dict handed to the model:

1. **Flanking.** If `upstream_seq` / `downstream_seq` are present, they are concatenated onto every sequence (`upstream + sequence + downstream`) *before* length-based routing, so flanks count toward the short-vs-long decision.
2. **Sequence spec check** (`check_seqs_specifications`): no empty sequences; only `A/T/C/G/N` bases (case-insensitive). Offending IDs are reported.
3. **Range bounds check:** for each non-empty range, both `start` and `end` must be `< len(sequence)` (0-based, `end` inclusive). Out-of-bounds ranges raise a `PredictionFailedError` naming the maximum valid index.

> Range *format* validation (types, ordering, non-negativity) lives in `validate_request_payload`; range *bounds* validation lives here because bounds can only be checked once flanks have been applied.

## Output scaling (`apply_scaling`)

Called per task after predictions return from `predict_chrombpnet` (which always yields linear values):

- **`linear` (or unset):** returned unchanged; effective scale `"linear"`.
- **`log`:** `np.log(np.clip(arr, 1e-10, None))` — the epsilon clip prevents `log(0) = -inf`; residual non-finite values are caught downstream by `_sanitize_for_json`.

Returns `(transformed_predictions, effective_scale)`; the effective scale is echoed back as `scale_prediction_actual`.