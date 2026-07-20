# MAYA — Phase 3 Sprint 2

## AI Training Engine

**Status:** Complete  
**Scope:** Modular training only — **no Grad-CAM, Flask, frontend, SHA-256, or PDF reports**

---

## Training workflow

```
CLI (--profile) → TrainingConfig / ModelConfig
        ↓
ModelFactory.create()          Phase 2 DataLoaders (train/val)
        ↓                              ↓
   AdamW + CE loss  ←—— Trainer.fit() ——→ validate.py metrics
        ↓
 CosineAnnealingLR + EarlyStopping + ModelCheckpoint
        ↓
 history.csv · checkpoints · experiment_XXX.json · sprint2 artefacts
```

1. Load a named **profile** (epochs / batch size).
2. Build the model **only** via `ModelFactory` (Sprint 3.1).
3. Stream batches from Phase 2 `create_dataloader` (never load the full corpus).
4. For each epoch: train → validate → callbacks → history row.
5. Persist `best.pt` / `last.pt`, append `experiment_XXX.json`, write Sprint 2 artefacts.

## Epochs & batches

| Profile | Epochs | Batch size |
|---------|--------|------------|
| `debug` | 2 | 4 |
| `development` | 10 | 8 |
| `production` | 25 | 8 |

An **epoch** = one full pass over the training DataLoader (optionally capped with `--max-batches` for smoke runs). A **batch** is one mini-batch of images/labels moved to the selected device, forward → loss → backward → optimizer step, then tensors are deleted to free RAM.

## Optimizer, loss, scheduler

| Component | Choice | Why |
|-----------|--------|-----|
| Optimizer | **AdamW** | Stable fine-tuning of the classifier head with weight decay |
| Loss | **CrossEntropyLoss** | Standard multi-class objective for REAL/FAKE logits |
| Scheduler | **CosineAnnealingLR** | Smooth LR decay over `T_max=epochs` without manual step schedules |
| Clipping | **grad_clip_norm** (default 1.0) | Prevents exploding gradients on CPU fine-tunes |

## Early stopping

`EarlyStopping` watches `val_loss` (mode=`min`) with configurable patience / minΔ. When patience is exceeded, `CallbackContext.stop_training` ends the loop without requiring trainer special-cases.

## Checkpoints

`ai/engine/checkpoint.py` saves:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`, `metrics`, `config`

`last.pt` is always written; `best.pt` is updated by `ModelCheckpoint` when the monitored metric improves. Resume via `--resume path/to.pt`.

## Experiment tracking

`ExperimentTracker` writes sequential files:

`artifacts/experiments/experiment_001.json`, `experiment_002.json`, …

Each record stores timestamp, dataset version (`dataset/versions/CURRENT`), model name, hyperparameters, metrics, duration, and notes.

## Module interactions

| Module | Role |
|--------|------|
| `ai/training/training_config.py` | Profiles + `TrainingConfig` |
| `ai/training/train.py` | CLI entry |
| `ai/training/validate.py` | Validation metrics only |
| `ai/training/callbacks.py` | EarlyStopping, ModelCheckpoint, LR hook |
| `ai/engine/trainer.py` | Epoch loop coordinator |
| `ai/engine/checkpoint.py` | Save / load / resume |
| `ai/engine/history.py` | Epoch metrics + CSV |
| `ai/engine/experiment.py` | `experiment_XXX.json` |
| `ai/training/artifacts.py` | Sprint 2 output pack |
| `ai/engine/device.py` | CPU / CUDA / MPS (Sprint 3.1) |
| `ai/models/model_factory.py` | Model creation (Sprint 3.1) |

## How to run

```bash
python -m ai.training.train --profile debug
python -m ai.training.train --profile development
python -m ai.training.train --profile production

# Equivalent:
python scripts/train.py --profile debug

# Optional smoke cap (memory / speed)
python scripts/train.py --profile debug --max-batches 5
```

Logs: `logs/training.log`  
Artefacts: `artifacts/phase3/sprint2/`

## Tests

```bash
python -m pytest tests/test_training.py -q
```

## Future extensibility

- Register new profiles with `register_profile("overnight", epochs=50, batch_size=8)`.
- Swap architectures by changing `ModelConfig.model_name` once Sprint 3.1 builders exist.
- Add callbacks (CSV loggers, W&B) without editing `Trainer.fit`.

## Explicit non-goals

Evaluation dashboards, Grad-CAM, Flask routes, frontend pages.
