from __future__ import annotations

import pytest

from app.data.leakage import LeakageError, assert_no_leakage, find_leaks


def test_planted_13gram_caught() -> None:
    shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    train = [f"intro text {shared} trailing train words"]
    test = [f"other intro {shared} trailing test words"]
    leaks = find_leaks(train, test, n=13)
    assert leaks
    with pytest.raises(LeakageError):
        assert_no_leakage(train, test, n=13)


def test_no_leak_when_disjoint() -> None:
    train = ["completely different train passage about networks and hubs"]
    test = ["unrelated test passage discussing entropy and information"]
    assert find_leaks(train, test, n=13) == []
    assert_no_leakage(train, test, n=13)
