import random

from src.data.builders.build_mus22t_agent_split import transductive_split


def test_transductive_split_is_8_1_1_by_positives_with_1_to_1_unique_negatives():
    rng = random.Random(1)
    nodes = [f"p{i:03d}" for i in range(200)]
    positives = sorted({tuple(sorted(rng.sample(nodes, 2))) for _ in range(400)})  # sparse: 1:1 negatives exist
    labelled = transductive_split(positives)
    n = len(positives)
    pos = {s: [p for p, y in labelled[s] if y] for s in labelled}
    assert [len(pos[s]) for s in ("train", "val", "test")] == [
        int(0.8 * n),
        int(0.1 * n),
        n - int(0.8 * n) - int(0.1 * n),
    ]
    assert sorted(p for s in pos.values() for p in s) == positives  # every positive used once, none dropped
    negatives = [p for s in labelled.values() for p, y in s if not y]
    assert all(sum(y for _, y in labelled[s]) * 2 == len(labelled[s]) for s in labelled)
    assert len(set(negatives)) == len(negatives) and not set(negatives) & set(positives)
    assert all(a < b for a, b in negatives)
