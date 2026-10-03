IMAGE_WIDTH = 96
IMAGE_HEIGHT = 144

RAW_DIR = "data/raw"
AUGMENTED_DIR = "data/augmented"
MODEL_DIR = "models"
OUTPUT_DIR = "output"
BENCHMARK_RESULTS_DIR = "benchmark-results"
BENCHMARK_DIR = "benchmarks"
MODEL_PREFIX = "segment"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# One shared model supports these display families.
SEGMENT_TYPES = (7, 13, 14, 16)
MAX_SEGMENTS = max(SEGMENT_TYPES)

# Legacy filenames such as d0_4_001.png default to this type.
# New data can encode the type explicitly, e.g. s14_d0_A_001.png.
DEFAULT_SEGMENT_TYPE = 7

# Optional overrides for legacy filenames.
# Example:
# DISPLAY_TYPE_BY_ID = {"d20": 14, "d21": 16}
DISPLAY_TYPE_BY_ID = {}

BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# Hold out complete display IDs when set, e.g. ["d3"].
VALIDATION_DISPLAY_IDS = []

# Used only when VALIDATION_DISPLAY_IDS is empty.
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

SEGMENT_THRESHOLD = 0.5

# Multi-task loss weighting.
TYPE_LOSS_WEIGHT = 0.5
SEGMENT_LOSS_WEIGHT = 1.0

# Stop after this many epochs without validation character-accuracy improvement.
EARLY_STOPPING_PATIENCE = 15
