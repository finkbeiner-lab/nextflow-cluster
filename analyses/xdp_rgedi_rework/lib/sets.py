"""Which threshold set is primary in this run, and how sets are named in figures and documents (decision D28)."""

from lib import cfg

_A = cfg.params().get("analysis", {})
PRIMARY = _A.get("primary_set", "D_sd")
COUNT_SETS = _A.get("count_sets", ["D_sd", "B"])
LABELS = _A.get("set_labels", {})


def label(name: str) -> str:
    return LABELS.get(name, name)
