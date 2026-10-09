import random

import pytest

from tools.audit_bean_counts import count_block, dup_pairs, iou
from tools.make_count_test import assign_split


def test_iou_and_duplicate_pairs():
    a, b = [0, 0, 10, 10], [0, 1, 10, 9]  # IoU 0.9
    assert iou(a, a) == 1.0
    assert iou(a, [20, 20, 5, 5]) == 0.0
    assert 0.7 < iou(a, b) < 1.0
    assert dup_pairs([a, b, [50, 50, 10, 10]]) == 1


def test_count_block_scores_null_as_zero():
    rows = [{"n_gt": "4", "n_beans": "4"}, {"n_gt": "2", "n_beans": ""}]
    s = count_block(rows)
    assert s["MAE_null_as_0"] == 1.0 and s["null_rate"] == 0.5 and s["under_rate"] == 0.5


def test_assign_split_is_group_level_and_exact():
    groups = [("a", 2), ("b", 2), ("c", 3), ("d", 4), ("e", 4)]
    s1 = assign_split(groups, 7, random.Random(1))
    s2 = assign_split(groups, 7, random.Random(1))
    assert s1 == s2  # deterministic with seed
    assert sum(n for g, n in groups if s1[g] == "dev") == 7
    with pytest.raises(RuntimeError):
        assign_split([("a", 4), ("b", 4)], 3, random.Random(1))
