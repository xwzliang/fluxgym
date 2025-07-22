import os
import re
import uuid
import threading
import subprocess
from subprocess import PIPE, STDOUT
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import logging
from logging.handlers import RotatingFileHandler
import uvicorn

# ─── Workdir ─────────────────────────────────────────────────────────────────────
WORK_DIR = Path(".")

# ─── your existing helpers ─────────────────────────────────────────────────────
from app import download, create_dataset, gen_toml, gen_sh

# ─── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = WORK_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("training_server")
logger.setLevel(logging.INFO)

rotator = RotatingFileHandler(
    LOG_DIR / "server.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=1,
)
rotator.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(rotator)

# ─── Job registry ───────────────────────────────────────────────────────────────
jobs = {}
jobs_lock = threading.Lock()


# ─── Request model ──────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    trigger_word: str


app = FastAPI()


# ─── Background training worker ─────────────────────────────────────────────────
def run_training(job_id: str, trigger_word: str):
    BASE_MODEL = "flux-dev"
    OUTPUT_NAME = trigger_word
    IMAGE_DIR = WORK_DIR / "raw_data" / OUTPUT_NAME / "images"
    CAPTION_DIR = WORK_DIR / "raw_data" / OUTPUT_NAME / "captions"
    RESOLUTION = 512
    CLASS_TOKENS = OUTPUT_NAME
    NUM_REPEATS = 10
    SEED = 42
    VRAM = "20G"
    WORKERS = 2
    LEARNING_RATE = "8e-4"
    NETWORK_DIM = 4
    MAX_EPOCHS = 16
    SAVE_EVERY = 4
    TIMESTEP_SAMPLING = "shift"
    GUIDANCE_SCALE = 1.0
    SAMPLE_PROMPTS = [OUTPUT_NAME]
    SAMPLE_EVERY = 0

    logger.info(f"[{job_id}] Starting training for '{trigger_word}'")
    with jobs_lock:
        jobs[job_id]["status"] = "running"

    try:
        # ─── 1) download base model ───────────────────────────────────────────────
        # download(BASE_MODEL)

        # ─── 2) prepare dataset folder ────────────────────────────────────────────
        ds_folder = WORK_DIR / "outputs" / OUTPUT_NAME / "dataset"
        ds_folder.mkdir(parents=True, exist_ok=True)

        # ─── 3) collect and validate image/caption files ─────────────────────────
        if not IMAGE_DIR.is_dir():
            raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
        if not CAPTION_DIR.is_dir():
            raise FileNotFoundError(f"Caption directory not found: {CAPTION_DIR}")

        # only real files with proper extensions
        image_paths = sorted(
            str(p)
            for p in IMAGE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

        # ─── Match captions ──────────────────────────────────────────────────────
        caption_paths = []
        for img_path in image_paths:
            cap = CAPTION_DIR / (Path(img_path).stem + ".txt")
            if not cap.is_file():
                raise FileNotFoundError(f"Missing caption for {img_path}: {cap}")
            caption_paths.append(str(cap))

        # ─── CORRECTED CALL: first argument is the *list* of images  ────────────
        create_dataset(
            str(ds_folder),
            RESOLUTION,
            image_paths,  # <-- a list, not a string
            *caption_paths,  # <-- each caption afterwards
        )

        # ─── 5) write dataset.toml ────────────────────────────────────────────────
        toml_cfg = gen_toml(
            dataset_folder=str(ds_folder),
            resolution=RESOLUTION,
            class_tokens=CLASS_TOKENS,
            num_repeats=NUM_REPEATS,
        )
        out_base = WORK_DIR / "outputs" / OUTPUT_NAME
        out_base.mkdir(parents=True, exist_ok=True)
        (out_base / "dataset.toml").write_text(toml_cfg)

        # ─── 6) write and launch train.sh ────────────────────────────────────────
        sh = gen_sh(
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
        )
        sh_path = out_base / "train.sh"
        sh_path.write_text(sh)
        sh_path.chmod(0o755)

        logger.info(f"[{job_id}] Launching training script: {sh_path}")
        # ─── POPEN + real-time logging ────────────────────────────────────────
        proc = subprocess.Popen(
            ["bash", str(sh_path)],
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            bufsize=1,
        )
        # stream each line as it comes
        last_pct = None
        pct_re = re.compile(r"\b(\d{1,3})%")

        for raw in proc.stdout:
            line = raw.rstrip()
            # try to pull out the percentage
            m = pct_re.search(line)
            if m:
                pct = int(m.group(1))
                # skip logging if it's the same as last time
                if pct == last_pct:
                    continue
                last_pct = pct

            # if no percentage was found, or it changed, log it
            logger.info(f"[{job_id}] {line}")

        retcode = proc.wait()
        if retcode:
            raise subprocess.CalledProcessError(retcode, str(sh_path))

        # ─── after process exits ─────────────────────────────────────────────
        safetensors = out_base / f"{OUTPUT_NAME}.safetensors"
        if not safetensors.exists():
            raise FileNotFoundError(f"Expected {safetensors} but not found")

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output"] = str(safetensors)

        logger.info(f"[{job_id}] Training completed successfully")

    except Exception as e:
        logger.exception(f"[{job_id}] Training failed")
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)


# ─── API endpoints ──────────────────────────────────────────────────────────────
@app.post("/start_training")
def start_training(req: StartRequest):
    trigger = req.trigger_word
    safetensors = WORK_DIR / "outputs" / trigger / f"{trigger}.safetensors"

    # If already trained, skip and return immediately
    if safetensors.exists():
        logger.info(
            f"Output for '{trigger}' already exists at {safetensors}, skipping training."
        )
        return {"status": "completed", "output": str(safetensors)}

    # Otherwise queue a new job
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "pending", "output": None, "error": None}

    threading.Thread(target=run_training, args=(job_id, trigger), daemon=True).start()

    logger.info(f"[{job_id}] Queued job for trigger_word='{trigger}'")
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": job["status"], "error": job["error"]}


@app.get("/output/{job_id}")
def download_output(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, "Job not completed")
    path = job["output"]
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Output file missing")
    return FileResponse(path, filename=Path(path).name)


# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FluxGym training server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port)
