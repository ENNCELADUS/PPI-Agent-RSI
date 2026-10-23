import random

from src.data.builders.build_pring_sandbox import balance, go_descendants, scrub


def test_balance_keeps_tested_negatives_and_fills_to_one_to_one():
    pairs = {("a", "b"): 1, ("a", "c"): 1, ("b", "c"): 0, ("b", "d"): 0, ("c", "d"): 0, ("a", "d"): 0}
    out = balance(pairs, {("c", "d"): "negatome"}, random.Random(0))
    assert sum(out.values()) == 2 and len(out) == 4
    assert out[("c", "d")] == 0 and out[("a", "b")] == out[("a", "c")] == 1


def test_scrub_drops_sentences_naming_another_gene():
    text = (
        "FUNCTION: Plays a role in autophagy {ECO:0000269|PubMed:1}. Recruits BAX to mitochondria. BECN1 is cytosolic."
    )
    assert scrub(text, {"BAX", "BECN1"}, {"BECN1"}) == ("Plays a role in autophagy. BECN1 is cytosolic.", 1)
    assert scrub(text, {"BAX", "BECN1"}, set()) == ("Plays a role in autophagy.", 2)


def test_go_descendants_follows_is_a_only(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text(
        "[Term]\nid: GO:0000001\n\n[Term]\nid: GO:0000002\nis_a: GO:0000001 ! root\n\n"
        "[Term]\nid: GO:0000003\nis_a: GO:0000002 ! child\n\n[Term]\nid: GO:0000004\n"
        "relationship: part_of GO:0000001\n\n[Typedef]\nid: part_of\nis_a: GO:0000001\n"
    )
    assert go_descendants(obo, ["GO:0000001"]) == {"GO:0000001", "GO:0000002", "GO:0000003"}
