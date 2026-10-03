SEGMENT_ORDER = ("A", "B", "C", "D", "E", "F", "G")

CHAR_TO_SEGMENTS = {
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
}

SEGMENTS_TO_CHAR = {
    bits: char for char, bits in CHAR_TO_SEGMENTS.items()
}

def decode_segments(bits):
    return SEGMENTS_TO_CHAR.get(tuple(int(x) for x in bits), "?")
