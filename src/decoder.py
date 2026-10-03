import math

from .labels import CHAR_TO_SEGMENTS


def rank_characters(probabilities, eps=1e-7):
    """
    Rank valid characters by the Bernoulli likelihood of their expected
    seven-segment pattern.

    The returned confidence is normalized only across the known character set;
    it should therefore be treated as a relative decoder confidence rather than
    a calibrated probability of correctness.
    """
    probs = [min(1.0 - eps, max(eps, float(p))) for p in probabilities]

    scored = []
    for char, pattern in CHAR_TO_SEGMENTS.items():
        log_likelihood = 0.0
        for p, expected_on in zip(probs, pattern):
            log_likelihood += (
                math.log(p) if expected_on else math.log(1.0 - p)
            )

        scored.append({
            "char": char,
            "pattern": pattern,
            "log_likelihood": log_likelihood,
        })

    scored.sort(key=lambda item: item["log_likelihood"], reverse=True)

    max_log = scored[0]["log_likelihood"]
    weights = [
        math.exp(item["log_likelihood"] - max_log)
        for item in scored
    ]
    weight_sum = sum(weights)

    for item, weight in zip(scored, weights):
        item["confidence"] = weight / weight_sum

    return scored


def decode_probabilities(probabilities):
    ranked = rank_characters(probabilities)

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    second_confidence = (
        second["confidence"] if second is not None else 0.0
    )

    return {
        "char": best["char"],
        "confidence": best["confidence"],
        "second_char": second["char"] if second is not None else None,
        "second_confidence": second_confidence,
        "margin": best["confidence"] - second_confidence,
        "ranking": ranked,
    }
