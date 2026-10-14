# SearchJev

Training code for **SearchJev: A Calibrated System-1 Model for Search Agents**.

SearchJev turns the short, structured decisions of a search agent (is this document relevant? is the
evidence sufficient? where should the next query go? which link next?) into schema-conditioned typed
prediction. Given a search state and a decision schema, it returns a calibrated probability
distribution over the legal values of every field in a single forward pass of a Qwen3.5 backbone,
without autoregressive decoding.

<p align="center">
  <img src="assets/decision.png" alt="Search decision performance" width="60%">
</p>

*Search decision performance. On SearchDecision-Bench, SearchJev achieves 5.2-5.3x faster decisions than
same-size Qwen3.5. The x-axis uses Qwen3.5-4B as the 1x reference.*

<p align="center">
  <img src="assets/architecture.png" alt="Architecture of the Dual-System Search Agent" width="100%">
</p>

*Architecture of the Dual-System Search Agent. System 2 handles planning, query generation, and final
answer synthesis, while SearchJev (System 1) makes search decisions specified by the schema at each step.
Confident decisions are accepted; uncertain ones are delegated to System 2. The search controller uses the
resolved decisions to select actions and update the search state with retrieved evidence.*

Everything is open: code, data, model weights, and the complete training logs
(links are anonymized for review).

This repository contains the complete training pipeline used for the paper models
(SearchJev-0.8B and SearchJev-4B; the same recipe also trains the 9B backbone):

1. **data conversion**: the SearchDecision-Bench release -> per-dataset JEV files (`tools/convert_unified_v2.py`);
2. **data preparation**: schema instantiation, option permutation, label-reliability weights,
   deduplication and removal of any train/val state that also occurs in a test split (`training/data.py`);
3. **training**: LoRA on Qwen3.5, restricted first-token option-label readout, soft-label
   cross-entropy + 0.5 x Brier score (`training/loss.py`, `training/trainer.py`), single GPU or DDP;
4. **calibration**: one temperature per question type fitted on half of the validation data,
   and a held-out evaluation on the other half (`training/calibration.py`).

## Repository layout

```
start_training.py          one-click entry point: prepare -> train -> calibrate -> held-out eval -> summary
config.yaml                the paper recipe (all hyper-parameters; overridable with --set key=value)
training/
  config.py                config loading, presets, validation
  schema.py, labels.py     JEV rows -> typed decision examples; single-token option labels
  prompting.py             prompt rendering (state + question + options)
  data.py, collator.py     data preparation, caching, length-grouped batching
  modeling.py              Qwen3.5 + LoRA, restricted option-label readout
  loss.py, trainer.py      soft-label CE + Brier, transformers.Trainer subclass
  calibration.py           temperature scaling
  inference.py             JEVPredictor (batched inference with a trained run)
  metrics.py, gpus.py, bootstrap.py
tools/convert_unified_v2.py
```

## Installation

Python 3.12 on Linux with CUDA GPUs (the paper runs used 6x NVIDIA H200). The Qwen3.5 linear-attention
kernels (`flash-linear-attention`) require a GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or: uv sync  (exact versions from pyproject.toml)
```

`start_training.py` re-executes itself with `./.venv/bin/python` (or `$JEV_PYTHON`) when it is started
with an interpreter that lacks torch.

## Data preparation

SearchDecision-Bench is trained on; the SearchJev-V1 families (`classification`, `ranking`, `search`)
are used only for validation and calibration.

```bash
# 1. SearchDecision-Bench (train / val / test)
hf download <anonymized>/search_jev_unified_v2_release_20260928 --repo-type dataset \
    --local-dir data/search_jev_unified_v2_release_20260928
python tools/convert_unified_v2.py          # -> data/SearchJev-V2/unified_v2/<dataset>.jsonl

# 2. SearchJev-V1 (validation-only families)
hf download <anonymized>/SearchJev-V1 --repo-type dataset --local-dir data/SearchJev-V1
for f in classification ranking search; do ln -s ../SearchJev-V1/$f data/SearchJev-V2/$f; done
```

Without SearchJev-V1, add `--set data.val_only_families=[]` (validation and calibration then use the
SearchDecision-Bench validation split only).

Data preparation runs automatically at the start of training and is cached under `data/processed/`;
`--prepare-only` stops after it. Rows whose split is `test` are never used for training or validation.

## Training

```bash
# one model (all visible GPUs; DDP via torchrun when more than one GPU is selected)
python start_training.py --model 4B --gpus 0,1,2,3,4,5

# the paper models, one after another (output: outputs/searchjev-{0.8b,4b})
python start_training.py --models 0.8B,4B --gpus 0,1,2,3,4,5

# log to Weights & Biases
python start_training.py --model 4B --wandb SearchJev

# inspect the resolved configuration and data sources without doing anything
python start_training.py --model 4B --dry-run

# end-to-end smoke test on one GPU (0.8B, a few hundred examples, 20 steps)
python start_training.py --debug

# resume from the latest checkpoint
python start_training.py --model 4B --resume auto
```

Any config key can be overridden, e.g. `--set training.learning_rate=1e-4`.

### Paper recipe

| | SearchJev-0.8B / 4B |
|---|---|
| Backbone | Qwen3.5-0.8B / 4B (instruct) |
| Tuning | LoRA, rank 16, alpha 32, dropout 0.05 (attention, Gated DeltaNet and MLP projections) |
| Optimizer | AdamW (fused), lr 5e-5, weight decay 0.01, cosine schedule, 5% warm-up, grad clip 1.0, bf16 |
| Batch | 22 per GPU x 6 GPUs = 132; 1 epoch = 5,176 steps over 683,130 training decisions |
| Max input length | 2,048 tokens (only the state is truncated) |
| Loss | soft-label cross-entropy + 0.5 x Brier score |
| Label weight | 0.5 for LLM-assigned labels, demonstrations and constructed rewrites |
| Calibration | one temperature per question type on 5,000 validation decisions |

With a different number of GPUs, keep the effective batch at 132
(`per_device_train_batch_size x gradient_accumulation_steps x #GPUs`).

## Outputs

```
outputs/searchjev-4b/
  adapter/                 LoRA adapter (+ tokenizer)
  calibration.json         per-question-type temperatures
  label_space.json         option-label alphabet
  config.resolved.yaml     the exact configuration of the run
  prepare_stats.json       data-preparation statistics
  train_metrics.json, train.log
  eval_val/metrics.json    held-out validation metrics before / after temperature scaling
  summary.json
```

## Inference

```python
from training.inference import JEVPredictor

predictor = JEVPredictor("outputs/searchjev-4b", device="cuda", merge_lora=True)
result = predictor.decide(
    state={"question": "When was the Eiffel Tower completed?",
           "documents": ["The Eiffel Tower was completed in March 1889 for the World's Fair."]},
    question="Is the evidence sufficient to answer the question now?",
    question_type="noul",                      # noul (yes/no) | choice | score
)
print(result.choice, result.confidence, result.probabilities)
```

`predictor.answer_jev(row)` answers every question of a row in the JEV format used by the data files.


## License

The code is released under the [Apache License 2.0](LICENSE). The datasets that SearchDecision-Bench is
built from keep their own licenses (listed in the paper's appendix).
