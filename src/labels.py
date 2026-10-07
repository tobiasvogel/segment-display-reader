from .config import (
    SEGMENT_TYPES,
    MAX_SEGMENTS,
    CHARACTER_WHITELISTS,
)


TYPE_TO_INDEX = {segment_type: i for i, segment_type in enumerate(SEGMENT_TYPES)}
INDEX_TO_TYPE = {i: segment_type for segment_type, i in TYPE_TO_INDEX.items()}

SEGMENT_NAMES = tuple(
    ["A", "B", "C", "D", "E", "F", "G"]
    + [f"S{i}" for i in range(7, MAX_SEGMENTS)]
)

CHARSETS = {
    7: {
        "0": (1, 1, 1, 1, 1, 1, 0),
        "1": (0, 1, 1, 0, 0, 0, 0),
        "2": (1, 1, 0, 1, 1, 0, 1),
        "3": (1, 1, 1, 1, 0, 0, 1),
        "4": (0, 1, 1, 0, 0, 1, 1),
        "5": (1, 0, 1, 1, 0, 1, 1),
        "6": (1, 0, 1, 1, 1, 1, 1),
        "7": (1, 1, 1, 0, 0, 0, 0),
        "8": (1, 1, 1, 1, 1, 1, 1),
        "9": (1, 1, 1, 1, 0, 1, 1),
        "-": (0, 0, 0, 0, 0, 0, 1),

        # Conventional lowercase 7-segment "h":
        # C + E + F + G.
        # If a particular display uses another glyph, adjust this tuple.
        "h": (0, 0, 1, 0, 1, 1, 1),
	"H": (0, 1, 1, 0, 1, 1, 1),
    },

    # Segment naming/layout conventions differ between vendors for 13/14/16
    # segment displays. Populate these maps explicitly when adding such data.
    13: {},
    14: {},
    16: {},
}


def get_whitelist(segment_type):
    return tuple(CHARACTER_WHITELISTS.get(int(segment_type), ()))


def get_charset(segment_type):
    segment_type = int(segment_type)
    whitelist = set(get_whitelist(segment_type))
    charset = CHARSETS.get(segment_type, {})

    return {
        char: pattern
        for char, pattern in charset.items()
        if char in whitelist
    }


def get_pattern(segment_type, label):
    return get_charset(segment_type).get(label)


def validate_label(segment_type, label):
    segment_type = int(segment_type)
    whitelist = get_whitelist(segment_type)

    if label not in whitelist:
        raise ValueError(
            f"Character {label!r} is not whitelisted for "
            f"{segment_type}-segment displays. "
            f"Allowed: {''.join(whitelist) or '<none>'}"
        )

    if label not in CHARSETS.get(segment_type, {}):
        raise ValueError(
            f"Character {label!r} is whitelisted for {segment_type}-segment "
            "displays but has no segment pattern configured in src/labels.py."
        )


def padded_target(segment_type, label):
    validate_label(segment_type, label)
    pattern = CHARSETS[int(segment_type)][label]

    target = [0.0] * MAX_SEGMENTS
    mask = [0.0] * MAX_SEGMENTS

    for i, value in enumerate(pattern):
        target[i] = float(value)
        mask[i] = 1.0

    return target, mask


def hard_decode(segment_type, bits):
    charset = get_charset(segment_type)
    active = int(segment_type)
    key = tuple(int(x) for x in bits[:active])

    for char, pattern in charset.items():
        if tuple(pattern) == key:
            return char

    return "?"
