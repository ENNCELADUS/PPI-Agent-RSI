from src.data.builders.build_pring_memory import final_pairs, split


def test_split_is_disjoint_seeded_and_forty_percent():
    disc, final = split({f"P{i}" for i in range(10)})
    assert len(final) == 4 and not disc & final and disc | final == {f"P{i}" for i in range(10)}
    assert split({f"P{i}" for i in range(10)}) == (disc, final)


def test_final_pairs_keep_inside_non_self_and_balance():
    final = {"A", "B", "C"}
    rows = [("A", "B", 1), ("A", "C", 1), ("B", "C", 0), ("A", "A", 1), ("A", "Z", 0), ("B", "B", 0)]
    got = final_pairs(rows, final)
    assert sum(r[2] for r in got) == 1 and len(got) == 2 and ("B", "C", 0) in got
