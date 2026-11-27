from __future__ import annotations

import pytest

from app.scoring import (
    ScoredExample,
    answerability_f1,
    concept_micro_f1,
    json_validity,
    parse_output,
    percentile,
    quote_faithfulness,
    quote_is_faithful,
    score_examples,
    tokens_per_second,
)

VALID = '{"answerable": true, "answer": "A.", "quote": "Q one.", "concepts": ["chaos"]}'
ABSTAIN = '{"answerable": false, "answer": "", "quote": "", "concepts": []}'


def test_parse_strict_and_lenient() -> None:
    strict = parse_output(VALID)
    assert strict.strict_valid and strict.parsed and strict.answerable is True
    fenced = parse_output("```json\n" + VALID + "\n```")
    assert not fenced.strict_valid and fenced.parsed and fenced.concepts == ("chaos",)
    prose = parse_output("Sure! " + ABSTAIN + " Hope that helps.")
    assert not prose.strict_valid and prose.parsed and prose.answerable is False
    thinking = parse_output("<think>hmm {not json}</think>" + VALID)
    assert thinking.parsed


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        "[1, 2]",
        '{"answerable": "yes", "answer": "", "quote": "", "concepts": []}',
        '{"answerable": true, "answer": "", "quote": ""}',
        '{"answerable": true, "answer": "", "quote": "", "concepts": [1]}',
    ],
)
def test_parse_rejects_bad_shapes(text: str) -> None:
    parsed = parse_output(text)
    assert not parsed.parsed and not parsed.strict_valid


def test_json_validity() -> None:
    assert json_validity([VALID, ABSTAIN, "```json\n" + VALID + "\n```", "x"]) == 0.5
    assert json_validity([]) is None


def test_answerability_f1_macro_and_none_counts_wrong() -> None:
    gold = [True, True, False, False]
    perfect = answerability_f1(gold, [True, True, False, False])
    assert perfect["macro"] == 1.0
    always_yes = answerability_f1(gold, [True, True, True, True])
    # answerable: tp=2 fp=2 fn=0 -> 2/3; unanswerable: tp=0 -> 0; macro 1/3
    assert always_yes["answerable"] == pytest.approx(2 / 3)
    assert always_yes["unanswerable"] == 0.0
    assert always_yes["macro"] == pytest.approx(1 / 3)
    unparsed = answerability_f1(gold, [None, True, False, None])
    # None flips to the wrong label: [False, True, False, True]
    assert unparsed["macro"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        answerability_f1([True], [])
    assert answerability_f1([], [])["macro"] is None


def test_quote_faithfulness() -> None:
    passage = "First sentence here. Second one. Third one."
    assert quote_is_faithful("Second one.", passage)
    assert not quote_is_faithful("second one.", passage)
    assert not quote_is_faithful("", passage)
    assert not quote_is_faithful("First sentence here. Second one. Third one.", passage)
    good = parse_output(VALID.replace("Q one.", "Second one."))
    bad = parse_output(VALID.replace("Q one.", "Not in passage."))
    abstain = parse_output(ABSTAIN)
    rate, n = quote_faithfulness([(good, passage), (bad, passage), (abstain, passage)])
    assert (rate, n) == (0.5, 2)
    assert quote_faithfulness([(abstain, passage)]) == (None, 0)


def test_concept_micro_f1() -> None:
    assert concept_micro_f1([["a", "b"]], [["a", "b"]]) == 1.0
    # tp=1 fp=1 fn=1 -> 2/4
    assert concept_micro_f1([["a", "b"]], [["a", "c"]]) == 0.5
    assert concept_micro_f1([[]], [[]]) is None


def test_percentile_and_throughput() -> None:
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert percentile(values, 50) == 300.0
    assert percentile(values, 95) == pytest.approx(480.0)
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile(values, 101)
    assert tokens_per_second([1000.0, 1000.0], [10, 30]) == 20.0
    assert tokens_per_second([], []) is None


def test_score_examples_numbers_only() -> None:
    passage = "Q one. Other text."
    examples = [
        ScoredExample(True, ("chaos",), passage, VALID, 100.0, 10),
        ScoredExample(False, (), passage, ABSTAIN, 300.0, 10),
        ScoredExample(True, ("chaos",), passage, "garbage", 200.0, 5),
    ]
    metrics = score_examples(examples)
    assert metrics["n_items"] == 3
    assert metrics["json_validity"] == pytest.approx(0.6667)
    assert metrics["quote_faithfulness"] == 1.0
    assert metrics["latency_p50_ms"] == 200.0
    assert metrics["tokens_per_sec"] == pytest.approx(41.67)
    assert all(not isinstance(v, str) for v in metrics.values())
