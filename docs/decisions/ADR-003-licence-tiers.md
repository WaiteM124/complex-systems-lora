# ADR-003: Licence tiers for corpus ingest and publication

**Status:** Accepted
**Date:** 2026-11-26

---

## Context

The complex-systems corpus spans CC BY, CC BY-NC-SA, and more restrictive licences. Publishing or training on the wrong tier would create legal and portfolio risk.

## Decision

- **Tier A** (CC BY): may be trained on and published (with attribution).
- **Tier B** (CC BY-NC-SA): may be trained on but never committed or published; stored under `data/private/` only.
- **Tier C**: never ingested.
- Miles's notes are optional private input (tier N), never committed. "Worked examples" sections and study-record files are dropped.

CI (`test_sources_licence`, `test_repo_clean`) enforces these rules.

## Consequences

- Public eval datasets contain tier-A items only.
- Private training data is uploaded separately (for example a private Kaggle dataset) and is gitignored.
