import json

from rag.observability import bounded_json


def test_bounded_json_redacts_vectors_and_bounds_nested_user_metadata():
    value = {
        "notes": "x" * 100,
        "labels": list(range(20)),
        "vector": [0.1] * 1_000,
        "nested": {str(index): index for index in range(20)},
    }

    safe, counters = bounded_json(
        value,
        max_items=3,
        max_string_characters=12,
        preserve_root_mapping=True,
    )

    assert safe["notes"] == "x" * 12
    assert safe["labels"] == [0, 1, 2]
    assert safe["vector"] == {
        "redacted": True,
        "reason": "vector payload omitted from diagnostics",
    }
    assert len(safe["nested"]) == 3
    assert counters["strings_truncated"] == 1
    assert counters["items_truncated"] == 34
    assert counters["values_redacted"] == 1
    json.dumps(safe, allow_nan=False)


def test_bounded_json_handles_cycles_and_non_finite_values():
    cyclic = []
    cyclic.append(cyclic)
    safe, counters = bounded_json(
        {"cycle": cyclic, "bad": float("nan")},
        max_items=5,
        max_string_characters=20,
        preserve_root_mapping=True,
    )
    assert safe["cycle"][0]["redacted"] is True
    assert safe["bad"] is None
    assert counters["cycles_redacted"] == 1
    assert counters["non_finite_replaced"] == 1
