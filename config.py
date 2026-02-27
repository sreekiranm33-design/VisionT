from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Paths
DATA_DIR = PROJECT_ROOT / "NEU-CLS-64"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "runs"

# Dataset
NUM_CLASSES = 9
IMAGE_SIZE = 64
IN_CHANNELS = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Model
EMBED_DIM = 192
NUM_HEADS = 3
DEPTH = 6
MLP_RATIO = 4.0
PATCH_SIZE = 2
DROP_PATH_RATE = 0.1
LAYER_SCALE_INIT = 1e-4

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 200
NUM_WORKERS = 4

# Optimizer (AdamW)
LR = 1e-3
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.999)

# Scheduler
WARMUP_EPOCHS = 10

# Regularization & Loss
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
USE_CLASS_WEIGHTED_LOSS = True
USE_WEIGHTED_SAMPLER = False
CLASS_WEIGHT_POWER = 0.5

# Augmentation (RandAugment)
RANDAUG_N = 2
RANDAUG_M = 9

# Normalization (grayscale)
NORM_MEAN = [0.5]
NORM_STD = [0.5]

# Misc
DEVICE = "cuda"
AMP = True
SAVE_EVERY = 10
EVAL_USE_TTA = True
TTA_FLIPS = ("none", "hflip", "vflip", "hvflip")
