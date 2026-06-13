# ============================================================================
# V-JEPA 2.1 domain-adaptive pretraining on endoscopic video (single server)
# ----------------------------------------------------------------------------
# What this script does, in order:
#   1. Starts ClearML (same pattern as the LeJEPA run) and pulls the dataset
#      archive from MinIO.
#   2. Clones facebookresearch/vjepa2 if it is not already present and puts it
#      on sys.path. We run Meta's own app/vjepa_2_1 trainer, not a rewrite.
#   3. Scans the data root for video files AND frame-sequence folders.
#      Frame sequences are re-encoded to mp4 with ffmpeg, because the repo's
#      VideoDataset decodes through decord and cannot read folders of frames.
#   4. Writes the space-delimited CSV the repo expects ("path label").
#   5. Downloads/repacks the released checkpoint
#      vjepa2_1_vitl_dist_vitG_384.pt into the schema that
#      app/vjepa_2_1/train.py::load_checkpoint expects
#      (encoder / target_encoder / predictor / opt / scaler / epoch with
#      "module.backbone." key prefixes). The released file stores the encoder
#      under "ema_encoder" and ships a ViT-G distillation predictor that does
#      not match the SSL predictor in the pretrain config, so the predictor is
#      re-initialized fresh. This was verified against the repo's loader.
#   6. Generates the single-GPU YAML config (anneal mode, no image co-training)
#      and launches training in-process so ClearML auto-uploads checkpoints.
#
# Edit the CONFIGURATION block, nothing else should need changes.
# ============================================================================

import warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

import copy
import csv as csv_mod
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

import torch
import yaml

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# --- ClearML ---------------------------------------------------------------
CLEARML_ENABLED = True
CLEARML_PROJECT = "lukmanov-team"
CLEARML_TASK_NAME = "VJEPA21_VitL_Endo_DomainPretrain"
CLEARML_OUTPUT_URI = "s3://api.blackhole2.ai.innopolis.university:443/lukmanov-team/checkpoints"

# --- Data (MinIO S3 Archives) -----------------------------------------------
DATA_S3_LONG = "s3://api.blackhole2.ai.innopolis.university:443/lukmanov-team/Hyper-Kvasir Long.zip"
DATA_S3_SHORT = "s3://api.blackhole2.ai.innopolis.university:443/lukmanov-team/Hyper-Kvasir Short.tar"
LOCAL_DATA_DIR = None  

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v", ".wmv")
FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MIN_SEQ_FRAMES = 16              # folders with fewer ordered frames are skipped

# The temporal rate (frames per second of REAL procedure time) that your frame
# sequences represent. If frames were dumped at the native camera rate use 25;
# if they were extracted at 1 fps, 1 real second sits between consecutive
# frames and you should think about whether such sequences belong in a 4 fps
# motion objective at all. Values below VIDEO_FPS are clamped up to VIDEO_FPS
# because the repo computes frame_step = native_fps // VIDEO_FPS and asserts
# it is > 0.
FRAME_SEQ_FPS = 25

# --- Checkpoint --------------------------------------------------------------
# Local .pt/.pth path, s3:// path, or https URL of the released 2.1 ViT-L file.
CKPT_SOURCE = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"  # EDIT ME if you have it locally

# --- Repo / run folders -------------------------------------------------------
VJEPA2_REPO_DIR = os.path.abspath("./vjepa2")
RUN_DIR = os.path.abspath("./vjepa21_endo_run")   # checkpoints + logs + generated config land here

# --- Training (single 40GB GPU defaults) -------------------------------------
DEVICES = ["cuda:0"]   # add "cuda:1", ... to launch the repo's multi-process mode
BATCH_SIZE = 24        # per GPU; bf16 + activation checkpointing, ~fits 40GB. Try 16-20 if memory allows.
NUM_WORKERS = 24
EPOCHS = 10
IPE = 300              # iterations per "epoch"; loader is refreshed when exhausted,
                       # so this can exceed len(dataset)/batch and decouples
                       # checkpoint cadence from dataset size
LR = 2.0e-4            # peak LR. Official cooldown used 6e-4 at effective batch 1536;
                       # at batch ~12 stay in the 5e-5 .. 1.5e-4 range
FINAL_LR = 2.0e-6      # anneal mode decays linearly LR -> FINAL_LR over EPOCHS*IPE
WEIGHT_DECAY = 0.04
EMA = [0.99925, 0.99925]
CROP_SIZE = 256        # checkpoint is 384px-trained but uses RoPE with interpolation,
                       # 256px/16f is the official pretrain operating point and what a
                       # single GPU can afford. Bump to 384 only with much more memory.
FRAMES_PER_CLIP = 16
VIDEO_FPS = 4          # used only as the stride floor for frame-sequence conversion

# --- Temporal window (how much REAL TIME the 16 frames span) -----------------
# This is the lever for "how much the model sees" in seconds. Stride is derived
# per file from native fps as round(window_s * native_fps / FRAMES_PER_CLIP),
# so a value means the same real duration on 25fps and 30fps sources.
# Provide a LIST to randomize per sample: each clip independently picks one,
# so a batch mixes dense short-context and sparse long-context windows.
# Set to a single float for a fixed (non-fps) window instead.
WINDOW_SECONDS = [1.0, 4.0, 8.0, 12.0, 16.0]
MIN_STRIDE = 1         # floor so dense windows never collapse to stride 0
SAVE_EVERY_EPOCHS = 1
SEED = 239

DRY_RUN = False        # True: do all preparation, skip the actual training launch


# ============================================================================
# 2. CLEARML
# ============================================================================
task = None
if CLEARML_ENABLED:
    from clearml import Task, StorageManager

    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=CLEARML_TASK_NAME,
        output_uri=CLEARML_OUTPUT_URI,
    )
    task.connect(
        {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "ipe": IPE,
            "lr": LR,
            "final_lr": FINAL_LR,
            "crop_size": CROP_SIZE,
            "frames_per_clip": FRAMES_PER_CLIP,
            "fps": VIDEO_FPS,
            "frame_seq_fps": FRAME_SEQ_FPS,
            "ckpt_source": CKPT_SOURCE,
        }
    )

os.makedirs(RUN_DIR, exist_ok=True)


# ============================================================================
# 3. GET THE VJEPA2 REPO
# ============================================================================
def ensure_repo(repo_dir):
    if not os.path.isdir(os.path.join(repo_dir, "app", "vjepa_2_1")):
        print(f"[repo] cloning facebookresearch/vjepa2 into {repo_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/vjepa2.git", repo_dir],
            check=True,
        )
    else:
        print(f"[repo] using existing repo at {repo_dir}")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


# ============================================================================
# 4. DATA PREPARATION
# ============================================================================
def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(
            "ffmpeg not found. Install it (apt install ffmpeg) or pip install imageio-ffmpeg."
        )


def convert_frame_dir_to_mp4(frame_dir, out_path, fps, ffmpeg_exe):
    """Re-encode an ordered folder of frames into an mp4 decord can read.
    Frames are natural-sorted and symlinked to a numbered pattern first, so
    non-zero-padded names (frame_2.jpg before frame_10.jpg) stay in order."""
    frames = [
        f
        for f in os.listdir(frame_dir)
        if os.path.splitext(f)[1].lower() in FRAME_EXTS
    ]
    if len(frames) < MIN_SEQ_FRAMES:
        return False
    frames.sort(key=natural_key)
    ext = os.path.splitext(frames[0])[1].lower()
    frames = [f for f in frames if f.lower().endswith(ext)]

    tmp = out_path + ".frames_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        os.symlink(os.path.abspath(os.path.join(frame_dir, f)), os.path.join(tmp, f"f_{i:08d}{ext}"))

    enc_fps = max(int(fps), VIDEO_FPS)  # native_fps // VIDEO_FPS must stay > 0
    if enc_fps != fps:
        print(f"[convert] WARNING {frame_dir}: requested fps {fps} < {VIDEO_FPS}, encoding at {enc_fps}")
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-framerate", str(enc_fps),
        "-i", os.path.join(tmp, f"f_%08d{ext}"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[convert] FAILED for {frame_dir}: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def prepare_data(data_root, run_dir):
    """Return path to the training CSV covering native videos + converted sequences."""
    converted_dir = os.path.join(run_dir, "converted_sequences")
    os.makedirs(converted_dir, exist_ok=True)

    videos, frame_dirs = [], []
    for dirpath, dirnames, filenames in os.walk(data_root):
        if os.path.abspath(dirpath).startswith(os.path.abspath(converted_dir)):
            continue
        vids_here = [f for f in filenames if f.lower().endswith(VIDEO_EXTS)]
        imgs_here = [f for f in filenames if os.path.splitext(f)[1].lower() in FRAME_EXTS]
        videos += [os.path.join(dirpath, f) for f in vids_here]
        if not vids_here and len(imgs_here) >= MIN_SEQ_FRAMES:
            frame_dirs.append(dirpath)

    print(f"[data] found {len(videos)} native video files and {len(frame_dirs)} frame-sequence folders")

    if frame_dirs:
        ffmpeg_exe = find_ffmpeg()
        for fd in sorted(frame_dirs):
            safe = re.sub(r"[^0-9A-Za-z_-]+", "_", os.path.relpath(fd, data_root)).strip("_")
            out_path = os.path.join(converted_dir, f"{safe}.mp4")
            if os.path.exists(out_path):
                videos.append(out_path)
                continue
            if convert_frame_dir_to_mp4(fd, out_path, FRAME_SEQ_FPS, ffmpeg_exe):
                videos.append(out_path)
                print(f"[convert] {fd} -> {out_path}")

    if not videos:
        raise RuntimeError(f"No usable videos found under {data_root}")

    # the repo parses the CSV with a single-space delimiter, so paths must not
    # contain spaces; offending files get a sanitized symlink
    safe_dir = os.path.join(run_dir, "safe_paths")
    final_paths = []
    for v in sorted(set(videos)):
        if " " in v:
            os.makedirs(safe_dir, exist_ok=True)
            link = os.path.join(safe_dir, re.sub(r"\s+", "_", os.path.basename(v)))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(v), link)
            v = link
        final_paths.append(os.path.abspath(v))

    csv_path = os.path.join(run_dir, "endoscopy_train.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.writer(f, delimiter=" ", lineterminator="\n")
        for p in final_paths:
            w.writerow([p, 0])  # label column is unused in pretraining
    print(f"[data] wrote {len(final_paths)} entries to {csv_path}")
    return csv_path


# ============================================================================
# 5. CHECKPOINT FETCH + REPACK
# ============================================================================
def fetch_checkpoint(source, run_dir):
    if source.startswith("s3://"):
        from clearml import StorageManager

        local = StorageManager.get_local_copy(source)
    elif source.startswith(("http://", "https://")):
        local = os.path.join(run_dir, os.path.basename(source))
        if not os.path.exists(local):
            print(f"[ckpt] downloading {source}")
            urllib.request.urlretrieve(source, local)
    else:
        local = source
    if not os.path.exists(local):
        raise FileNotFoundError(f"checkpoint not found: {local}")
    return local


def repack_vjepa21_checkpoint(src_path, dst_path):
    """Convert the released V-JEPA 2.1 checkpoint into the schema expected by
    app/vjepa_2_1/train.py::load_checkpoint.

    The released file keeps the encoder under 'ema_encoder' and bundles a
    ViT-G distillation predictor (depth 12, teacher dim 1664) that is
    architecturally different from the SSL predictor in the pretrain config
    (depth 24, embed 384, own 4-level targets), so the predictor starts fresh.
    Encoder weights are re-prefixed to 'module.backbone.' to match
    DDP(MultiSeqWrapper(vit)) state dicts; verified to load with
    '<All keys matched successfully>' through the repo's own loader.
    """
    if os.path.exists(dst_path):
        print(f"[repack] reusing {dst_path}")
        return dst_path
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    enc_sd = None
    if isinstance(ckpt, dict):
        for key in ("ema_encoder", "target_encoder", "encoder"):
            if key in ckpt:
                enc_sd = ckpt[key]
                print(f"[repack] using '{key}' weights from {os.path.basename(src_path)}")
                break
    if enc_sd is None:
        enc_sd = ckpt
        print("[repack] checkpoint looks like a raw state dict")

    clean = {}
    for k, v in enc_sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        clean["module.backbone." + k] = v

    try:
        scaler_state = torch.amp.GradScaler("cuda").state_dict()
    except Exception:
        scaler_state = {
            "scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
            "growth_interval": 2000, "_growth_tracker": 0,
        }

    torch.save(
        {
            "encoder": clean,
            "target_encoder": copy.deepcopy(clean),
            "predictor": {},                              # fresh SSL predictor
            "opt": {"state": {}, "param_groups": []},     # mismatch is caught, optimizer reinits
            "scaler": scaler_state,
            "epoch": 0,
        },
        dst_path,
    )
    print(f"[repack] wrote {dst_path} ({len(clean)} encoder tensors)")
    return dst_path


# ============================================================================
# 6. CONFIG GENERATION (single GPU, anneal mode, no image co-training)
# ============================================================================
def build_config(csv_path, anneal_ckpt_path, run_dir):
    cfg = {
        "app": "vjepa_2_1",
        "folder": run_dir,
        "data": {
            "dataset_type": "VideoDataset",
            "datasets": [csv_path],
            # no datasets_weights: a single dataset uses the plain
            # DistributedSampler and the loader is refreshed when exhausted
            "batch_size": BATCH_SIZE,
            "crop_size": CROP_SIZE,
            "patch_size": 16,
            "dataset_fpcs": [FRAMES_PER_CLIP],
            "tubelet_size": 2,
            "fps": VIDEO_FPS,
            "num_workers": NUM_WORKERS,
            "pin_mem": True,
        },
        "data_aug": {
            "auto_augment": False,
            "motion_shift": False,
            "random_resize_aspect_ratio": [0.75, 1.35],
            "random_resize_scale": [0.3, 1.0],
            "reprob": 0.0,
        },
        # NOTE: img_data / img_mask are intentionally absent. The image
        # co-training branch computes img_world_size = int(world_size * 0.5),
        # which is 0 on a single GPU and crashes on a modulo-by-zero. It also
        # adds nothing for endoscopic domain adaptation.
        "loss": {
            "loss_exp": 1.0,
            "predict_all": True,        # dense loss on context tokens too (the 2.1 objective)
            "shift_by_n": 0,
            "weight_distance_loss": False,
        },
        "mask": [
            {
                "aspect_ratio": [0.75, 1.5],
                "full_complement": False,
                "max_keep": None,
                "max_temporal_keep": 1.0,
                "num_blocks": 8,
                "spatial_scale": [0.15, 0.15],
                "temporal_scale": [1.0, 1.0],
            },
            {
                "aspect_ratio": [0.75, 1.5],
                "full_complement": False,
                "max_keep": None,
                "max_temporal_keep": 1.0,
                "num_blocks": 2,
                "spatial_scale": [0.7, 0.7],
                "temporal_scale": [1.0, 1.0],
            },
        ],
        "meta": {
            "dtype": "bfloat16",
            "eval_freq": 100,
            "load_checkpoint": True,
            "read_checkpoint": None,
            "save_every_freq": SAVE_EVERY_EPOCHS,
            "seed": SEED,
            "use_sdpa": True,
        },
        "model": {
            "has_cls_first": False,
            "img_temporal_dim_size": 1,
            "interpolate_rope": True,
            "is_causal": False,
            "lambda_value_img": 0.5,
            "lambda_value_vid": 0.5,     # weight of the dense context loss
            "lambda_progressive": False,
            # the released vjepa2_1_vitl_dist_vitG_384.pt encoder has NO
            # modality embedding parameters (the hub loads it strict=True
            # without them), so this must stay False to match the checkpoint
            "modality_embedding": False,
            "model_name": "vit_large",
            "n_registers": 0,
            "n_registers_predictor": 0,
            "normalize_predictor": False,
            "pred_depth": 24,            # must be one of {4,8,12,20,24,40}
            "pred_embed_dim": 384,
            "pred_num_heads": 12,
            "pred_is_causal": False,
            "uniform_power": True,
            "use_activation_checkpointing": True,
            "use_mask_tokens": True,
            "use_rope": True,
            "zero_init_mask_tokens": True,
        },
        "optimization": {
            "is_anneal": True,           # load released weights, epoch resets to 0,
            "anneal_ckpt": anneal_ckpt_path,  # LR decays linearly LR -> FINAL_LR
            "resume_anneal": True,       # restarts pick up RUN_DIR/latest.pth.tar
            "ema": EMA,
            "epochs": EPOCHS,
            "ipe": IPE,
            "ipe_scale": 1.0,
            "lr": LR,
            "final_lr": FINAL_LR,
            "start_lr": LR,
            "warmup": 0,
            "weight_decay": WEIGHT_DECAY,
            "final_weight_decay": WEIGHT_DECAY,
        },
    }
    cfg_path = os.path.join(run_dir, "endo_vitl16_anneal_256px_16f.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[config] wrote {cfg_path}")
    return cfg_path


# ============================================================================
# 7. INSTALL THE RANDOMIZED-WINDOW DATASET (monkeypatch, repo untouched)
# ============================================================================
def install_windowed_dataset():
    """Replace the repo's make_videodataset so init_data builds the
    WindowedVideoDataset. The repo imports make_videodataset lazily inside
    init_data, so patching the module attribute before launch is enough.
    fps/frame_step args coming from init_data are dropped; WINDOW_SECONDS
    drives stride instead."""
    import src.datasets.video_dataset as vds
    from windowed_video_dataset import make_windowed_videodataset

    def _patched(*args, **kwargs):
        kwargs.pop("fps", None)
        kwargs.pop("frame_step", None)
        kwargs.pop("duration", None)
        kwargs["window_seconds"] = WINDOW_SECONDS
        kwargs["min_stride"] = MIN_STRIDE
        return make_windowed_videodataset(*args, **kwargs)

    vds.make_videodataset = _patched
    print(f"[window] randomized temporal window installed: {WINDOW_SECONDS} s "
          f"(stride per file = round(window * native_fps / {FRAMES_PER_CLIP}))")


# ============================================================================
# 8. OPTIONAL: tail the repo's CSV log into ClearML scalars
# ============================================================================
def start_log_tailer(run_dir):
    if task is None:
        return

    def _tail():
        log_file = os.path.join(run_dir, "log_r0.csv")
        seen = 0
        logger = task.get_logger()
        while True:
            time.sleep(20)
            try:
                if not os.path.exists(log_file):
                    continue
                with open(log_file) as f:
                    rows = f.readlines()
                for line in rows[max(1, seen):]:
                    parts = line.strip().split(",")
                    if len(parts) < 3:
                        continue
                    epoch, itr, loss = int(parts[0]), int(parts[1]), float(parts[2])
                    step = (epoch - 1) * IPE + itr
                    logger.report_scalar("Training", "loss", iteration=step, value=loss)
                seen = len(rows)
            except Exception:
                pass

    threading.Thread(target=_tail, daemon=True).start()


# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    ensure_repo(VJEPA2_REPO_DIR)

    # -- data
    if LOCAL_DATA_DIR:
        data_root = LOCAL_DATA_DIR
    else:
        from clearml import StorageManager

        print(f"[data] downloading and extracting {DATA_S3_ARCHIVE}")
        data_root = StorageManager.get_local_copy(DATA_S3_ARCHIVE)
    print(f"[data] data root: {data_root}")
    csv_path = prepare_data(data_root, RUN_DIR)

    # -- checkpoint
    raw_ckpt = fetch_checkpoint(CKPT_SOURCE, RUN_DIR)
    anneal_ckpt = repack_vjepa21_checkpoint(raw_ckpt, os.path.join(RUN_DIR, "anneal_init.pth.tar"))

    # -- config
    cfg_path = build_config(csv_path, anneal_ckpt, RUN_DIR)

    if DRY_RUN:
        print("[dry-run] preparation finished, skipping training launch")
        return

    start_log_tailer(RUN_DIR)

    # the windowed-dataset module lives next to this script; keep it importable
    # after we chdir into the repo
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    os.chdir(VJEPA2_REPO_DIR)
    install_windowed_dataset()

    if len(DEVICES) == 1:
        # in-process launch: ClearML hooks stay active, so torch.save inside
        # the repo's trainer auto-uploads latest.pth.tar / e*.pth.tar to MinIO
        from app.main import process_main

        process_main(rank=0, fname=cfg_path, world_size=1, devices=DEVICES)
    else:
        # multi-process: spawned workers re-import, so the patch must be applied
        # inside each. We launch a tiny shim instead of app.main directly.
        shim = os.path.join(RUN_DIR, "_launch_shim.py")
        with open(shim, "w") as f:
            f.write(
                "import sys\n"
                f"sys.path.insert(0, {_script_dir!r})\n"
                f"sys.path.insert(0, {VJEPA2_REPO_DIR!r})\n"
                "import pretrain\n"
                "pretrain.install_windowed_dataset()\n"
                "import runpy\n"
                "sys.argv = ['app.main', '--fname', "
                f"{cfg_path!r}, '--devices'] + {DEVICES!r}\n"
                "runpy.run_module('app.main', run_name='__main__')\n"
            )
        subprocess.run([sys.executable, shim], check=True, cwd=VJEPA2_REPO_DIR)


if __name__ == "__main__":
    main()
