import os

# ============================================================
# Project Settings
# ============================================================

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- TGCN WLASL-100 model ----
MODEL_DIR = os.path.join(BASE_DIR, "models", "tgcn")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "asl100.ini")
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "ckpt.pth")

# ---- WLASL Dataset Path (for reference / gloss list) ----
WLASL_JSON_PATH = os.path.join(BASE_DIR, "data", "WLASL_v0.3.json")

# ---- ASL100 Gloss List (sorted alphabetically to match the TGCN LabelEncoder) ----
ASL100_GLOSSES = [
    'accident', 'africa', 'all', 'apple', 'basketball',
    'bed', 'before', 'bird', 'birthday', 'black',
    'blue', 'book', 'bowling', 'brown', 'but',
    'can', 'candy', 'chair', 'change', 'cheat',
    'city', 'clothes', 'color', 'computer', 'cook',
    'cool', 'corn', 'cousin', 'cow', 'dance',
    'dark', 'deaf', 'decide', 'doctor', 'dog',
    'drink', 'eat', 'enjoy', 'family', 'fine',
    'finish', 'fish', 'forget', 'full', 'give',
    'go', 'graduate', 'hat', 'hearing', 'help',
    'hot', 'how', 'jacket', 'kiss', 'language',
    'last', 'later', 'letter', 'like', 'man',
    'many', 'medicine', 'meet', 'mother', 'need',
    'no', 'now', 'orange', 'paint', 'paper',
    'pink', 'pizza', 'play', 'pull', 'purple',
    'right', 'same', 'school', 'secretary', 'shirt',
    'short', 'son', 'study', 'table', 'tall',
    'tell', 'thanksgiving', 'thin', 'thursday', 'time',
    'walk', 'want', 'what', 'white', 'who',
    'woman', 'work', 'wrong', 'year', 'yes'
]

# ---- TGCN sequence and architecture settings (from models/tgcn/asl100.ini) ----
NUM_SAMPLES = 50          # Number of frames sampled per sign gesture
NUM_KEYPOINTS = 55        # 13 upper body + 21 left hand + 21 right hand
HIDDEN_SIZE = 64
NUM_STAGES = 20
DROP_P = 0.3
NUM_CLASSES = 100         # WLASL100 vocabulary size

# ---- Inference Settings ----
DEVICE = "cpu"            # Fast CPU inference (~15ms on modern CPUs)
CONFIDENCE_THRESHOLD = 0.3   # Confidence threshold for reliable recognition
BUFFER_STRIDE = 5         # Frame stride for motion tracking
