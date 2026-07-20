# MAYA — Architecture (Canonical)

**Authoritative content:** [`03_ARCHITECTURE.md`](03_ARCHITECTURE.md)

## Binding rules for Phase 2

1. Layered architecture: Presentation → Application → Business → **AI Analysis** → Storage  
2. Dataset / PyTorch code lives under `ai/` and **must not** import Flask  
3. Original raw dataset bytes are immutable; derived assets go to `dataset/processed/`  
4. Prefer generators / incremental I/O on 8 GB RAM hosts  
5. One module = one responsibility under `ai/datasets/`
