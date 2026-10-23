import random

from src.data.negatives import IdentityKeys, pring_negatives


def test_negatives_are_new_non_self_pairs_over_positive_endpoints():
    positives = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "e")]
    known = set(positives)
    negs = pring_negatives(positives, 4, lambda ks: any(k in known for k in ks), random.Random(0), IdentityKeys())
    ends = {x for p in positives for x in p}
    assert len(set(negs)) == 4
    assert all(a < b and a in ends and b in ends and (a, b) not in known for a, b in negs)
