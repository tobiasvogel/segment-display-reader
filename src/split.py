import random
from collections import defaultdict

from .config import (
    VALIDATION_DISPLAY_IDS,
    VALIDATION_FRACTION,
    RANDOM_SEED,
)


def split_samples(samples):
    """
    Split samples without leaking offline augmentations of the same original
    image across training and validation.

    If VALIDATION_DISPLAY_IDS is set, complete display IDs are held out.
    Otherwise, the split is performed by source_id, where an original and all
    files named *_augNNN share the same source_id.
    """
    samples = list(samples)

    if VALIDATION_DISPLAY_IDS:
        wanted = set(VALIDATION_DISPLAY_IDS)

        train = [s for s in samples if s.display_id not in wanted]
        val = [s for s in samples if s.display_id in wanted]

        if not train:
            raise RuntimeError("Display split produced an empty training set.")
        if not val:
            raise RuntimeError(
                "No samples found for VALIDATION_DISPLAY_IDS."
            )

        return train, val

    groups = defaultdict(list)
    for sample in samples:
        groups[sample.source_id].append(sample)

    group_ids = list(groups)
    if len(group_ids) < 2:
        raise RuntimeError(
            "At least two independent source images are required for a "
            "train/validation split."
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(group_ids)

    n_val_groups = max(
        1,
        int(round(len(group_ids) * VALIDATION_FRACTION)),
    )
    n_val_groups = min(n_val_groups, len(group_ids) - 1)

    val_ids = set(group_ids[:n_val_groups])

    train = [
        sample
        for group_id, group_samples in groups.items()
        if group_id not in val_ids
        for sample in group_samples
    ]
    val = [
        sample
        for group_id, group_samples in groups.items()
        if group_id in val_ids
        for sample in group_samples
    ]

    return train, val
