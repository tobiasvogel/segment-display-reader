import random
from .config import (
    VALIDATION_DISPLAY_IDS,
    VALIDATION_FRACTION,
    RANDOM_SEED,
)


def split_samples(samples):
    if VALIDATION_DISPLAY_IDS:
        wanted = set(VALIDATION_DISPLAY_IDS)

        train = [s for s in samples if s.display_id not in wanted]
        val = [s for s in samples if s.display_id in wanted]

        if not train:
            raise RuntimeError("Display-Split erzeugt leeres Trainingsset.")
        if not val:
            raise RuntimeError(
                "Keine Samples für VALIDATION_DISPLAY_IDS gefunden."
            )

        return train, val

    samples = list(samples)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(samples)

    n_val = max(1, int(round(len(samples) * VALIDATION_FRACTION)))
    n_val = min(n_val, len(samples) - 1)

    return samples[n_val:], samples[:n_val]
