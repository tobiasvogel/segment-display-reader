from .config import SEGMENT_TYPES, MAX_SEGMENTS


TYPE_TO_INDEX = {segment_type: i for i, segment_type in enumerate(SEGMENT_TYPES)}
INDEX_TO_TYPE = {i: segment_type for segment_type, i in TYPE_TO_INDEX.items()}

# Segment ordering is model-internal and intentionally generic beyond 7-segment.
# For 7-segment, positions 0..6 correspond to A..G.
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
    },

    # 13/14/16-segment conventions are not universal enough to invent here.
    # Add device/project-specific character maps when training such samples.
    13: {},
    14: {},
    16: {},
}


def get_charset(segment_type):
    return CHARSETS.get(int(segment_type), {})


def get_pattern(segment_type, label):
    return get_charset(segment_type).get(label)


def padded_target(segment_type, label):
    pattern = get_pattern(segment_type, label)
    target = [0.0] * MAX_SEGMENTS
    mask = [0.0] * MAX_SEGMENTS

    if pattern is None:
        return target, mask

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
