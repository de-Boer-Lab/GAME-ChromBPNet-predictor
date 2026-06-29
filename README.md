# GAME-ChromBPNet-Predictor

A RESTful ChromBPNet Predictor for the **Genomic API for Model Evaluation (GAME)** framework. It serves base-pair–resolution chromatin accessibility predictions from DNA sequence over a Flask API, negotiates JSON / MessagePack wire formats, and resolves requested cell types against its bundled models via the GAME **Matcher** service.

The underlying model is ChromBPNet (Pampari et al., 2025, Kundaje Lab). This container ships the **no-bias** model heads only — the Tn5/DNase bias model has already been factored out, so predictions reflect sequence-driven accessibility directly. The ChromBPNet Predictor encapsulates 9 individual chromatin accessibility models (5 folds for each model, some trained with ATAC-seq data and others with DNase) across five cell types.

---

## Repository layout

The predictor package lives under `ChromBPNet_GAME/`. This top-level README is the entry point; each subfolder carries its own README with the detailed design rationale.

```
GAME-ChromBPNet-predictor/
├── README.md                             # ← this file: overview, how to run, models, Matcher
└── ChromBPNet_GAME/                      # the predictor package
    ├── ChromBPNet_predictor_RestAPI.py   # Flask entrypoint: endpoints, request loop, response assembly
    ├── chrombpnet_predict.py             # inference core: orchestration, ranges, long-seq chunking
    ├── chrombpnet_utils.py               # model selection (Matcher), loading, fold averaging, slicing
    ├── schema_validation.py              # generic GAME schema checks + preprocessing
    ├── model_validation.py               # ChromBPNet-specific checks + output scaling
    ├── error_checking_functions.py       # error classes + field-level checks
    ├── config.py                         # container-aware predictor versioning + wire-format config
    ├── predictor_content_handler.py      # JSON / MessagePack decode + encode (content negotiation)
    ├── predictor_help_message.json       # metadata served at /help
    ├── chrombpnet_predictor.def          # Apptainer build definition
    ├── dev_run.sh                        # bind-mount dev runner (no rebuild needed)
    ├── setup.py / requirements.txt       # editable install of the vendored chrombpnet package
    │
    ├── README.md                         # predictor internals (inference core + validation)
    ├── chrombpnet/                        
    │   └── README.md                     # vendored upstream Kundaje Lab library
    └── models/
        └── README.md                     # no-bias fold weights + cell-type→model mapping
```

> Start with [`ChromBPNet_GAME/`](ChromBPNet_GAME/README.md) for the core inference and validation logic, [`ChromBPNet_GAME/models/`](ChromBPNet_GAME/models/README.md) for the model set, and [`ChromBPNet_GAME/chrombpnet/`](ChromBPNet_GAME/chrombpnet/README.md) for the model interal library.

---

## Models

Predictions are averaged across **5 cross-validation folds** per model. Models are keyed by ENCODE experiment accession (ENCID) and assay:

| Cell line | ATAC | DNASE |
|-----------|:----:|:-----:|
| K562      | ✅   | ✅    |
| GM12878   | ✅   | ✅    |
| HEPG2     | ✅   | ✅    |
| IMR90     | ✅   | ✅    |
| H1        | —    | ✅    |

- **Supported feature:** `accessibility`
- **Supported species:** `homo_sapiens`
- **Input window:** 2114 bp · **Prediction window:** 1000 bp · **Resolution:** 1 bp (`bin_size: 1`)

**Cross-assay pooling.** Model selection returns *all* rows matching a requested cell line. For every cell line except H1, that means both the DNASE and ATAC fold sets are pulled and averaged together into a single mean (10 fold-models total). The response then reports `type_actual: ["DNASE", "ATAC"]` and includes an `aggregation: {"models": "mean"}` field. Requesting H1 pulls only its 5 DNASE folds, so no aggregation field is emitted. See [`ChromBPNet_GAME/models/`](ChromBPNet_GAME/models/README.md) and [`ChromBPNet_GAME/`](ChromBPNet_GAME/README.md) for the mechanics.

---

## How to run

All commands run from inside `ChromBPNet_GAME/`:

```bash
cd ChromBPNet_GAME
```

### Build the container

```bash
apptainer build chrombpnet_predictor.sif chrombpnet_predictor.def
```

The predictor name is versioned automatically from the Apptainer build timestamp (e.g. `ChromBPNet_20251128-180629_PST`); outside a container it falls back to `ChromBPNet_dev`.

### Start the predictor

```bash
apptainer run --nv --containall chrombpnet_predictor.sif <HOST> <PORT> <MATCHER_IP> <MATCHER_PORT>
```

| Arg | Meaning |
|-----|---------|
| `HOST` | IP / hostname the predictor binds to |
| `PORT` | Port the predictor listens on |
| `MATCHER_IP` | IP of the running Matcher service |
| `MATCHER_PORT` | Port of the running Matcher service |

All four arguments are required — ChromBPNet always needs a reachable Matcher to resolve cell types it doesn't host directly.

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict`  | Submit sequences + prediction tasks, receive predictions |
| `GET`  | `/formats`  | Supported request/response MIME types |
| `GET`  | `/help`     | Predictor metadata (`predictor_help_message.json`) |

Request and response bodies may be `application/json` or `application/msgpack`. The response format follows the client `Accept` header when supported; **errors are always returned as JSON**. See [`ChromBPNet_GAME/`](ChromBPNet_GAME/README.md) for the request schema and error contract.

---

## How Matcher is used

When a `/predict` task names a `cell_type`:

1. **Exact match first.** The requested cell type is matched (case-insensitively) against the `Cell Line` column of `models/model_mappings.txt`. On a hit, those models are used directly and `matcher_version` is reported as `"N/A"`.
2. **Matcher fallback.** With no exact match, the predictor POSTs to `http://<MATCHER_IP>:<MATCHER_PORT>/match` with the requested cell type and the list of cell lines it hosts. Matcher returns a `cell_type_actual` (the closest available cell line) and a `matcher_version`.
3. **Resolution.** The returned `cell_type_actual` is looked up in the mapping table and its models are used. The response surfaces both the requested and actual cell type (`cell_type_requested` vs `cell_type_actual`) plus the `matcher_version`.
4. **No match.** If Matcher returns `NULL` / no usable cell type, the task fails with a request error rather than guessing.

Matcher connectivity failures are caught and surfaced as a structured error instead of crashing the request. The exact selection logic lives in `choose_model()` — see [`ChromBPNet_GAME/`](ChromBPNet_GAME/README.md).