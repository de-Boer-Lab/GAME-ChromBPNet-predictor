# `models/` — Model weights & cell-type mapping

This folder holds the trained ChromBPNet weights and the table that maps requested cell types onto them.

```
models/
├── model_mappings.txt          # Cell Line → ENCID → Assay
└── models_nobias/
    └── model.chrombpnet_nobias.fold_{0-4}.{ENCID}.h5
```

---

## Weights

Each `.h5` is a no-bias ChromBPNet fold model: a Keras model taking `(N, 2114, 4)` one-hot input and producing two heads — profile logits `(N, 1000)` and a single log-count scalar `(N, 1)`. Loading requires the custom `multinomial_nll` loss object (handled by `load_model_wrapper` in `../chrombpnet_utils.py`); models are loaded with `compile=False`.

**No-bias** means the Tn5/DNase enzymatic bias model has already been factored out, so these heads predict sequence-driven accessibility directly — no bias correction is applied at inference time.

### Filename convention

```
model.chrombpnet_nobias.fold_{i}.{ENCID}.h5      # i ∈ 0..4
```

There are **5 folds per ENCID**. The loader builds these paths from the ENCIDs selected for a task and expects them under `models/models_nobias/`; a missing file raises `FileNotFoundError` naming the offending path.

---

## Mapping table (`model_mappings.txt`)

```
Cell Line,ENCID,Assay
K562,ENCSR000EOT,DNASE
GM12878,ENCSR000EMT,DNASE
HEPG2,ENCSR149XIL,DNASE
IMR90,ENCSR477RTP,DNASE
H1,ENCSR000EMU,DNASE
K562,ENCSR868FGK,ATAC
GM12878,ENCSR637XSC,ATAC
HEPG2,ENCSR291GJU,ATAC
IMR90,ENCSR200OML,ATAC
```

Nine models across five cell lines and two assays. **H1 is DNASE-only**; the other four have both ATAC and DNASE.

| Cell line | DNASE ENCID | ATAC ENCID |
|-----------|-------------|------------|
| K562      | ENCSR000EOT | ENCSR868FGK |
| GM12878   | ENCSR000EMT | ENCSR637XSC |
| HEPG2     | ENCSR149XIL | ENCSR291GJU |
| IMR90     | ENCSR477RTP | ENCSR200OML |
| H1        | ENCSR000EMU | — |

---

## How the table is used

`choose_model` (in `../chrombpnet_utils.py`) matches a requested cell type against the `Cell Line` column case-insensitively and returns **all** matching rows:

- **Exact match** → those rows are used directly.
- **No exact match** → the Matcher service is queried with the requested cell type and the list of available cell lines; the returned `cell_type_actual` is then looked up here.

Because *all* rows for a cell line are returned, requesting a cell line with both assays selects both model sets. Their folds are pooled — **5 folds × 2 assays = 10 fold-models averaged into one mean track** — and the task reports `type_actual: ["DNASE", "ATAC"]` with `aggregation: {"models": "mean"}`. Requesting H1 selects only its 5 DNASE folds, giving a 5-fold mean and no aggregation field.

---

## Note

ENCIDs are ENCODE experiment accessions; the models are the Kundaje Lab ChromBPNet models (Pampari et al., 2025, preprint). The cell types exposed by this container are **HEPG2, K562, H1, GM12878, IMR90** (see `../predictor_help_message.json`).