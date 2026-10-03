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

SEGMENT_TYPES = (7, 13, 14, 16)
MAX_SEGMENTS = max(SEGMENT_TYPES)

DEFAULT_SEGMENT_TYPE = 7

DISPLAY_TYPE_BY_ID = {}

# Characters the decoder is allowed to consider for each display family.
# Keep this intentionally application-specific: a smaller whitelist reduces
# ambiguity between visually similar segment patterns.
#
# 7-segment includes lowercase "h" for hour indicators.
CHARACTER_WHITELISTS = {
    7: tuple("0123456789-h"),
    13: tuple(),
    14: tuple(),
    16: tuple(),
}

BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

VALIDATION_DISPLAY_IDS = []

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

SEGMENT_THRESHOLD = 0.5

TYPE_LOSS_WEIGHT = 0.5
SEGMENT_LOSS_WEIGHT = 1.0

EARLY_STOPPING_PATIENCE = 15
