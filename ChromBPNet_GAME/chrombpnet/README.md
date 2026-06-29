# `chrombpnet/` — Vendored upstream library

This is the upstream **ChromBPNet** package from the Kundaje Lab ([github.com/kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet)). 

## What the predictor actually uses

At inference time the GAME wrapper imports only **two** utilities from this package:

| Import | Used by | Purpose |
|--------|---------|---------|
| `chrombpnet.training.utils.one_hot.dna_to_one_hot` | `chrombpnet_predict.py` | One-hot encode padded sequences to `(N, 2114, 4)` before prediction |
| `chrombpnet.training.utils.losses.multinomial_nll` | `chrombpnet_utils.py` | Registered as a Keras custom object so the `.h5` fold models load (`load_model_wrapper`) |

Everything else in this folder is upstream machinery for *training and analysing* ChromBPNet models and is **not used by the Predictor**. It is retained so the package installs cleanly and so the two utilities above resolve, and for reproducibility of how the bundled models were produced.

## Layout (upstream)

```
chrombpnet/
├── CHROMBPNET.py            # upstream CLI entrypoint (train/predict/qc) — unused by the API
├── parsers.py, pipelines.py # CLI argument parsing + pipeline glue — unused by the API
├── data/                    # reference motifs, PWMs, MEME files (ATAC / DNASE / TF)
├── training/
│   ├── models/              # model architectures (bpnet_model.py, chrombpnet_with_bias_model.py)
│   ├── data_generators/     # batch generators + initializers
│   ├── utils/               # one_hot.py ✅, losses.py ✅, augment, callbacks, metrics, data_utils
│   ├── train.py, predict.py # training / batch-prediction drivers — unused by the API
│   └── metrics.py
├── helpers/                 # hyperparameter search, preprocessing, GC-matched negatives,
│                            #   chrom splits, report generation — unused by the API
└── evaluation/              # interpretation, TF-MoDISco, marginal/in-vivo footprints,
                             #   bigwig generation, figure notebooks — unused by the API
```

✅ marks the two modules the Predictor imports.

