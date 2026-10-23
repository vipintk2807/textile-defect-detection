"""Group-aware, stratified train/validation/test splitting and split-integrity checks.

Samples that share a ``group_id`` (same specimen/session) or are byte-identical duplicates are
kept together so that near-identical images cannot leak across splits. When no group metadata
exists each image is its own group, which cannot protect against near-duplicates of the same
physical sample; that limitation is reported by the dataset tools.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from .dataset import VALID_SPLITS, Issue, Sample
from .errors import DatasetError


@dataclass
class SplitResult:
    samples: list[Sample]
    warnings: list[str] = field(default_factory=list)
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 42


def _allocate(n_groups: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Number of groups per (train, val, test) for a class with ``n_groups`` groups."""
    if n_groups == 1:
        return 1, 0, 0
    if n_groups == 2:
        return 1, 0, 1
    _, r_val, r_test = ratios
    n_val = max(1, round(n_groups * r_val))
    n_test = max(1, round(n_groups * r_test))
    n_train = n_groups - n_val - n_test
    while n_train < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
        n_train = n_groups - n_val - n_test
    return n_train, n_val, n_test


def merge_duplicate_groups(samples: list[Sample], hashes: dict[str, str] | None) -> dict[str, str]:
    """Effective group per ``sample_id``: declared groups merged with exact-duplicate sets."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for s in samples:
        find(f"g:{s.group_key}")
    if hashes:
        first_for_hash: dict[str, str] = {}
        for s in samples:
            digest = hashes.get(s.sample_id)
            if not digest:
                continue
            if digest in first_for_hash:
                union(f"g:{s.group_key}", first_for_hash[digest])
            else:
                first_for_hash[digest] = f"g:{s.group_key}"
    return {s.sample_id: find(f"g:{s.group_key}") for s in samples}


def assign_splits(samples: list[Sample], *, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                  seed: int = 42, hashes: dict[str, str] | None = None) -> SplitResult:
    """Assign ``train``/``val``/``test`` per group, stratified by class, deterministically."""
    if not samples:
        raise DatasetError("There are no samples to split.")
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise DatasetError("Split ratios must be three positive numbers that sum to 1.")

    effective = merge_duplicate_groups(samples, hashes)
    members: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        members[effective[s.sample_id]].append(s)

    warnings: list[str] = []
    group_class: dict[str, str] = {}
    for gid, group in members.items():
        labels = Counter(s.class_label for s in group)
        if len(labels) > 1:
            warnings.append(
                f"Group '{gid[2:]}' contains multiple class labels {sorted(labels)}; it is kept "
                "together and assigned using its most common label.")
        group_class[gid] = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    by_class: dict[str, list[str]] = defaultdict(list)
    for gid, label in group_class.items():
        by_class[label].append(gid)

    rng = random.Random(seed)
    group_split: dict[str, str] = {}
    for label in sorted(by_class):
        gids = sorted(by_class[label])
        rng.shuffle(gids)
        n_train, n_val, n_test = _allocate(len(gids), ratios)
        if len(gids) < 3:
            warnings.append(
                f"Class '{label}' has only {len(gids)} independent group(s); it cannot appear in "
                "every split, so its validation/test performance cannot be estimated.")
        for i, gid in enumerate(gids):
            group_split[gid] = ("train" if i < n_train
                                else "val" if i < n_train + n_val else "test")

    result = [replace(s, split=group_split[effective[s.sample_id]]) for s in samples]
    return SplitResult(result, warnings, tuple(ratios), seed)  # type: ignore[arg-type]


def check_split_integrity(samples: list[Sample],
                          hashes: dict[str, str] | None = None) -> list[Issue]:
    """Return errors (confirmed leakage/structure problems) and warnings (review items)."""
    issues: list[Issue] = []
    seen_ids: dict[str, str] = {}
    for s in samples:
        if s.split not in VALID_SPLITS:
            issues.append(Issue("error", "invalid_split",
                                f"Sample has invalid split '{s.split}'.", s.sample_id))
        if s.sample_id in seen_ids:
            code = "sample_in_multiple_splits" if seen_ids[s.sample_id] != s.split \
                else "duplicate_sample_id"
            issues.append(Issue("error", code, "sample_id appears more than once.", s.sample_id))
        seen_ids[s.sample_id] = s.split

    group_splits: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        if s.group_id:
            group_splits[s.group_id].add(s.split)
    for gid, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            issues.append(Issue("error", "group_spans_splits",
                                f"Group '{gid}' appears in multiple splits {sorted(splits)} "
                                "(possible leakage).", ""))

    if hashes:
        digest_splits: dict[str, set[str]] = defaultdict(set)
        for s in samples:
            digest = hashes.get(s.sample_id)
            if digest:
                digest_splits[digest].add(s.split)
        for digest, splits in digest_splits.items():
            if len(splits) > 1:
                issues.append(Issue("error", "duplicate_across_splits",
                                    f"Identical image content (sha256 {digest[:12]}...) appears "
                                    f"in multiple splits {sorted(splits)} (leakage).", ""))

    classes = {s.class_label for s in samples}
    for split in VALID_SPLITS:
        in_split = {s.class_label for s in samples if s.split == split}
        if not in_split:
            issues.append(Issue("warning", "empty_split", f"Split '{split}' has no samples."))
            continue
        for missing in sorted(classes - in_split):
            issues.append(Issue("warning", "class_missing_from_split",
                                f"Class '{missing}' has no samples in split '{split}'."))
    return issues
