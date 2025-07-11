import os
import subprocess
from argparse import Namespace

# assume this file lives alongside the gradio app file, so all functions are importable:
from app import download, create_dataset, gen_toml, gen_sh

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
BASE_MODEL       = "flux-dev"      # must match a key in models.yaml
OUTPUT_NAME      = "myanimation2025061102"
IMAGE_DIR        = f"./raw_data/{OUTPUT_NAME}/images"       # only .png/.jpg
CAPTION_DIR      = f"./raw_data/{OUTPUT_NAME}/captions"     # matching .txt files
RESOLUTION       = 512                        # resize shorter edge → this
CLASS_TOKENS     = OUTPUT_NAME              # your instance prompt
NUM_REPEATS      = 10
SEED             = 42
VRAM             = "20G"                      # one of ["20G","16G","12G"]
WORKERS          = 2
LEARNING_RATE    = "8e-4"
NETWORK_DIM      = 4
MAX_EPOCHS       = 16
SAVE_EVERY       = 4
TIMESTEP_SAMPLING= "shift"
GUIDANCE_SCALE   = 1.0
SAMPLE_PROMPTS   = [OUTPUT_NAME]  # optional list
SAMPLE_EVERY     = 0                           # how many steps between sample generations
# ────────────────────────────────────────────────────────────────────────────────

# 1) make sure models are downloaded
# download(BASE_MODEL)

# 2) stage your dataset under outputs/{OUTPUT_NAME}/dataset
dataset_folder = f"outputs/{OUTPUT_NAME}/dataset"
os.makedirs(dataset_folder, exist_ok=True)

# collect in-order lists of image paths and captions
image_files   = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)])
caption_files = []
for img in image_files:
    txt = os.path.splitext(os.path.basename(img))[0] + ".txt"
    caption_files.append(os.path.join(CAPTION_DIR, txt))

# copy+resize images and write captions
create_dataset(
    dataset_folder,
    RESOLUTION,
    image_files,
    *caption_files
)

# 3) generate dataset TOML
train_config = gen_toml(
    dataset_folder=dataset_folder,
    resolution=RESOLUTION,
    class_tokens=CLASS_TOKENS,
    num_repeats=NUM_REPEATS
)
with open(f"outputs/{OUTPUT_NAME}/dataset.toml", "w") as f:
    f.write(train_config)

# 4) generate the “accelerate launch …” shell snippet
train_script = gen_sh(
    BASE_MODEL,
    OUTPUT_NAME,
    RESOLUTION,
    SEED,
    WORKERS,
    LEARNING_RATE,
    NETWORK_DIM,
    MAX_EPOCHS,
    SAVE_EVERY,
    TIMESTEP_SAMPLING,
    GUIDANCE_SCALE,
    VRAM,
    SAMPLE_PROMPTS,
    SAMPLE_EVERY,
    # if you have extra advanced flags, pass them here as booleans or strings
)
sh_path = f"outputs/{OUTPUT_NAME}/train.sh"
with open(sh_path, "w") as f:
    f.write(train_script)
os.chmod(sh_path, 0o755)

print(f"✔️  Prepared training script at {sh_path}")
print("⏳  Launching training…")

# 5) kick off training (this will mirror exactly what Gradio would do)
subprocess.run(["bash", sh_path], check=True)
print(f"✅  Training complete!  Check outputs/{OUTPUT_NAME} for your LoRA weights.")