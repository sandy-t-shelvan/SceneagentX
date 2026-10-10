from tools.partfield_parts import choose_k
from tools.review_step import ReviewError, review_block


def test_choose_k_only_when_a_split_beats_one_body():
    assert choose_k(None) is None
    assert choose_k({}) is None
    assert choose_k({2: 0.4, 3: 0.8}) is None
    assert choose_k({1: 0.5, 2: 0.4}) is None
    assert choose_k({1: 0.2, 3: 0.9, 2: 0.4}) == 3


def test_review_fails_closed_without_calling_an_agent(tmp_path):
    try:
        review_block("scan", "B5", tmp_path)
    except ReviewError as exc:
        text = str(exc)
    else:
        raise AssertionError("review should fail when the mask file is missing")
    assert "SCENEAGENT-RUN-STEP" in text
    assert "masks/object.json" in text
