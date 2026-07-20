# MAYA — Phase 2.5 Engineering Review

**Phase type:** Code review / optimization / architecture validation (no AI, no Flask, no UI)  
**Date:** 2026-07-15  
**Scope:** `ai/datasets/*`, `tests/test_dataset.py`, dataset docs & versioning

---

## Architecture validation

| Layer | Role | Current | Verdict |
|-------|------|---------|---------|
| Configuration | Paths, budgets, seeds, loader defaults | `dataset_config.py` | Good; needs versioning + future-dataset registry |
| Utility | Shared FS / image helpers | Missing (`utils/`) | **Gap** — duplicated count/extension logic |
| Processing | Inventory → integrity → sample → preprocess | Present, SRP-split | Strong |
| Visualization / Reports | Plots + markdown/JSON/CSV | Present | Strong |
| Versioning / Metadata | Sealed corpus identity | Missing | **Gap** |
| Testing | Pipeline + Dataset/DataLoader | Present, thin | Expand |
| Logging | Operational logs | Module loggers OK; CLI imports Flask logging | Decouple from `backend` |

**Layered architecture:** Dataset code correctly lives under `ai/` and does not import Flask (except the CLI script). That CLI coupling should move to a dataset-local logging setup.

**Suggested improvements (applied in this phase):**

1. Add `ai/datasets/utils/` for shared path/image/checksum helpers  
2. Add versioning + `dataset_metadata.json` without duplicating all image bytes by default  
3. Centralize dataset logging (no Flask dependency)  
4. Registry for future FaceForensics++ / Celeb-DF adapters  
5. Expand tests for config, corruption, DataLoader, folders  

---

## Per-module review

### `dataset_config.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Central tunables for the data pipeline |
| Responsibilities | Paths, budgets, seed, formats, loader defaults |
| Strengths | Dataclass + Enums; env override for raw dir; frozen budgets |
| Weaknesses | No version id; no `dataset_root`; no future-dataset registry; no project version |
| Complexity | Low |
| Memory | Negligible |
| Maintainability | High |
| Scalability | Medium until multi-corpus roots exist |
| Improvements | Version paths, registry, explicit split sizes, env overrides |

### `dataset.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Lazy PyTorch `Dataset` over processed splits |
| Responsibilities | Index folders; open RGB on demand |
| Strengths | True lazy load; pathlib; skip unknown folders |
| Weaknesses | Extension check duplicated vs validation |
| Complexity | Low |
| Memory | Index of paths only (~few MB for 5.2k) — good |
| Maintainability | High |
| Scalability | High for additional class folders via aliases |
| Improvements | Use shared `is_image_file` helper |

### `dataloader.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Configurable DataLoader + transform presets |
| Strengths | Norm only in transforms; workers=0 default for 8 GB |
| Weaknesses | ImageNet mean/std are inline magic tuples |
| Complexity | Low |
| Memory | Controlled by batch_size |
| Improvements | Move mean/std to config constants |

### `sampling.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Seeded balanced disjoint splits |
| Strengths | Reproducible; clear error on insufficient data |
| Weaknesses | Holds path lists per class (necessary, light) |
| Complexity | Low |
| Memory | Fine for ~5–10k paths |
| Improvements | None required beyond config-driven budgets |

### `validation.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Folder existence / count balance checks |
| Strengths | Structured report dataclass |
| Weaknesses | Local `_count_images`; redundant `+ (".jpg", ".jpeg")` |
| Improvements | Shared counter helper; trust config extensions |

### `statistics.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Class / format / geometry summaries |
| Strengths | Cap on dimension scans; pandas summary |
| Weaknesses | Keeps width/height lists (capped ≤5k — acceptable) |
| Memory | Acceptable on 8 GB |
| Improvements | Prefer inventory samples before re-open (already done) |

### `visualization.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | QA plots |
| Strengths | Agg backend; seeded sample; closes figures |
| Weaknesses | Loads thumbnails for grid only (OK) |
| Improvements | None critical |

### `reports.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Markdown / JSON / CSV writers |
| Strengths | Clear outputs matching Phase 2 contract |
| Improvements | Add metadata writer integration |

### `preprocessing.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Aspect-preserving resize + JPEG write |
| Strengths | Incremental; skip corrupt; no permanent normalize |
| Weaknesses | `_unique_destination` is generically reusable |
| Improvements | Move uniqueness helper to utils |

### `pipeline.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Orchestrate steps 1–6 |
| Strengths | Clear logging of steps; resets processed for reproducibility |
| Weaknesses | No version seal / metadata emit |
| Improvements | Emit metadata + version record after validation |

### `test_dataset.py`

| Aspect | Assessment |
|--------|------------|
| Purpose | Automated pipeline / Dataset checks |
| Strengths | Tiny e2e smoke test; corrupt zero-byte skipped |
| Weaknesses | Missing dedicated config, folder, checksum, version tests |
| Improvements | Expand coverage per Phase 2.5 checklist |

### Related: `inventory.py` / `integrity.py` / `csv_ingest.py`

| Strengths | Generators, skip+continue, no full-RAM loads |
| Weaknesses | `csv_ingest` hardcodes absolute Windows CSV path |
| Improvements | Resolve CSV via config / project root |

---

## Planned improvements (implementation order)

1. **Utils layer** — `ai/datasets/utils/` (fs, images, checksums)  
2. **Dataset logging** — `ai/datasets/logging_setup.py` (no Flask)  
3. **Config expansion** — versioning paths, future dataset registry, constants  
4. **Metadata + versioning** — `metadata.py`, `versioning.py`, `dataset_metadata.json`  
5. **Light refactors** — validation / preprocessing / dataset / dataloader / csv_ingest use utils  
6. **Pipeline** — seal version + write metadata after validation  
7. **Docs** — `DATASET.md`, versioning doc, README + ROADMAP update  
8. **Tests** — expand `test_dataset.py`  

**Non-goals:** Retrain nothing; do not re-download; do not duplicate 5200 images into `versions/v1/` by default (manifest + checksum seal instead) to protect disk on an 8 GB workstation.
