"""Data locations, all under the git-ignored data/ directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Private TMP data and the agent split built from it.
TMP = ROOT / "data" / "TMP"
RAW = TMP / "raw"
PUBLIC = TMP / "public_db"
PUBLIC_NEG = PUBLIC / "negative"  # experimentally tested non-interactions
OUT = TMP / "agent_split"
IDMAP = PUBLIC / "uniprot_idmap.map.gz"
STRING_ALIAS = PUBLIC / "string_alias.map.gz"

RAW_FILES = {
    "pretrain_pos": (RAW / "homo_positive_train.txt", "\t"),
    "pretrain_neg": (RAW / "homo_negative_train.csv", ","),
    "finetune_pos": (RAW / "homo_positive_finetune.txt", "\t"),
    "finetune_neg": (RAW / "homo_negative_pair_fintune.csv", ","),
    "test": (RAW / "depedent_test.csv", ","),
}

# Public PRING dev sandbox.
PRING = ROOT / "data" / "PRING"
PRING_RAW = PRING / "raw"
SANDBOX = PRING / "sandbox"  # what the agent may read
HIDDEN = PRING / "hidden"  # labels and the anon-id map; never given to the agent
