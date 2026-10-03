import math

from .labels import get_charset


def rank_characters(segment_type, probabilities, eps=1e-7):
    """
    Rank valid characters for one segment family by Bernoulli likelihood.

    Confidence is normalized over the configured charset for that family and is
    therefore a relative decoder confidence, not a calibrated correctness
    probability.
    """
    charset = get_charset(segment_type)
    if not charset:
        return []

    active = int(segment_type)
    probs = [
        min(1.0 - eps, max(eps, float(p)))
        for p in probabilities[:active]
    ]

    scored = []
    for char, pattern in charset.items():
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


def decode_probabilities(segment_type, probabilities):
    ranked = rank_characters(segment_type, probabilities)

    if not ranked:
        return {
            "char": "?",
            "confidence": 0.0,
            "second_char": None,
            "second_confidence": 0.0,
            "margin": 0.0,
            "ranking": [],
        }

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
