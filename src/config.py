IMAGE_WIDTH = 96
IMAGE_HEIGHT = 144

RAW_DIR = "data/raw"
AUGMENTED_DIR = "data/augmented"
MODEL_DIR = "models"
OUTPUT_DIR = "output"
BENCHMARK_RESULTS_DIR = "benchmark-results"
BENCHMARK_DIR = "benchmarks"
MODEL_PREFIX = "segment7"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

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

# Stop after this many epochs without validation character-accuracy improvement.
EARLY_STOPPING_PATIENCE = 15
