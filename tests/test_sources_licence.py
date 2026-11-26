from __future__ import annotations

import pytest

from app.data.sources import TierCError, ingestable_ids, load_sources, require_ingestable


def test_forty_six_sources_present() -> None:
    data = load_sources()
    assert len(data.sources) == 46
    ids = {source.id for source in data.sources}
    assert len(ids) == 46


def test_only_six_ingestable() -> None:
    ids = ingestable_ids()
    assert len(ids) == 6
    for source in load_sources().sources:
        if source.ingest:
            assert source.licence_class in {"A", "B"}
        if source.licence_class == "C":
            assert source.ingest is False


def test_tier_c_raises() -> None:
    tier_c = next(s.id for s in load_sources().sources if s.licence_class == "C")
    with pytest.raises(TierCError):
        require_ingestable(tier_c)
