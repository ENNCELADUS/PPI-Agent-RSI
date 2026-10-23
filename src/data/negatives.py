"""PRING negative sampling (Zheng et al. 2025, data_process/graph_gen/negative_sample.py,
after Bernett et al. 2024), used by the agent split.

Both proteins are drawn from the multiset of the positives' endpoints, so negatives keep the
positives' degree distribution; the second is redrawn until the pair is not a self or same-gene
pair, not known and not already drawn.
"""


class IdentityKeys:
    """`keys` for ids that are already one node per protein (e.g. PRING's)."""

    def of(self, a, b):
        return [(a, b)]

    def same_gene(self, a, b):
        return False


def pring_negatives(positives, n, is_known, rng, keys):
    """n unordered pairs; `is_known` gets keys.of(pair) and rejects the pair if true."""
    candidates = [acc for pair in positives for acc in pair]
    drawn = {}
    while len(drawn) < n:
        a = rng.choice(candidates)
        while True:
            b = rng.choice(candidates)
            pair = (a, b) if a < b else (b, a)
            if a != b and not keys.same_gene(a, b) and pair not in drawn and not is_known(keys.of(*pair)):
                break
        drawn[pair] = None
    return list(drawn)
