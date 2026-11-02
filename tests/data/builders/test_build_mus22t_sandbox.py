from src.data.builders.build_mus22t_sandbox import scrub


def test_scrub_drops_sentences_naming_another_gene_in_any_case():
    symbols, own = {"CYB5B", "EMC10", "ARC"}, {"EMC10"}
    text = "Part of the EMC complex. Interacts with CYB5B. Binds Cyb5b in the ER. Located in the arc of the cell."
    kept, dropped = scrub(text, symbols, own)
    assert kept == "Part of the EMC complex. Located in the arc of the cell."  # lower-case 'arc' is a word
    assert dropped == 2
