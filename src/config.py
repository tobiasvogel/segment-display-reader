IMAGE_WIDTH = 96
IMAGE_HEIGHT = 144

RAW_DIR = "data/raw"
AUGMENTED_DIR = "data/augmented"
MODEL_DIR = "models"
OUTPUT_DIR = "output"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Bei sehr kleinen Datensätzen absichtlich klein halten.
NUM_WORKERS = 0

# Wenn angegeben, werden komplette Display-IDs als Validation zurückgehalten.
# Beispiel: ["d3"]
VALIDATION_DISPLAY_IDS = []

# Falls leer, erfolgt ein reproduzierbarer zufälliger Split.
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

SEGMENT_THRESHOLD = 0.5

EARLY_STOPPING_PATIENCE = 15
