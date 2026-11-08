#!/usr/bin/env python3
"""Download VieNeu-TTS v3 Nano (+ MOSS tokenizer) into ./models for offline use.

Usage (from repo root, with network):
  uv run python scripts/download_models.py
  # or: .venv/bin/python scripts/download_models.py

Result:
  models/hub/...                         Hugging Face cache layout
  models/VieNeu-TTS-v3-Nano/...          flat ONNX dir for VIENEU_ONNX_DIR
  models/MOSS-Audio-Tokenizer-Nano-ONNX  flat copy of tokenizer ONNX assets
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "models"
NANO_REPO = "pnnbao-ump/VieNeu-TTS-v3-Nano"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
NANO_GRAPH = [
    "text_encoder.onnx",
    "duration_predictor.onnx",
    "vector_estimator.onnx",
    "codec_decoder.onnx",
    "config.json",
    "constants.npz",
]
NANO_CLONE = [
    "speaker_encoder.onnx",
    "codec_encoder.onnx",
    "reference_encoder.onnx",
    "denoiser.onnx",
]


def _hub_model_dir(hub: Path, repo_id: str) -> Path:
    return hub / f"models--{repo_id.replace('/', '--')}"


def _latest_snapshot(model_dir: Path) -> Path:
    snaps = sorted((model_dir / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(f"no snapshots under {model_dir}")
    return snaps[-1]


def _materialize(snapshot: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in snapshot.iterdir():
        if not src.is_file() and not src.is_symlink():
            continue
        real = src.resolve()
        target = dest / src.name
        if target.exists():
            copied.append(f"exists {src.name}")
            continue
        shutil.copy2(real, target)
        copied.append(f"copied {src.name} ({target.stat().st_size} bytes)")
    return copied


def download_repo(repo_id: str, hub: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        cache_dir=str(hub.parent),  # expects .../hub parent? HF uses cache_dir/hub
        local_dir=None,
    )
    # snapshot_download with cache_dir returns the snapshot path
    return Path(local)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    ap.add_argument("--skip-moss", action="store_true", help="skip MOSS tokenizer (preset TTS still works)")
    ap.add_argument("--offline-check", action="store_true", help="only verify local files, do not download")
    args = ap.parse_args()

    models_dir: Path = args.models_dir.resolve()
    hub = models_dir / "hub"
    hub.mkdir(parents=True, exist_ok=True)

    # huggingface_hub uses cache_dir/hub when cache_dir points at models_dir
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))

    if args.offline_check:
        nano_flat = models_dir / "VieNeu-TTS-v3-Nano"
        missing = [n for n in NANO_GRAPH if not (nano_flat / n).is_file()]
        if missing:
            print("MISSING in", nano_flat, ":", ", ".join(missing), file=sys.stderr)
            return 2
        print("OK offline Nano graph:", nano_flat)
        for n in NANO_GRAPH:
            p = nano_flat / n
            print(f"  {n}: {p.stat().st_size}")
        return 0

    print(f"HF_HOME={models_dir}")
    print(f"Downloading {NANO_REPO} ...")
    # Prefer explicit cache_dir=models_dir so files land in models/hub
    from huggingface_hub import snapshot_download

    nano_snap = Path(
        snapshot_download(
            repo_id=NANO_REPO,
            repo_type="model",
            token=args.token,
            cache_dir=str(models_dir),
        )
    )
    print("Nano snapshot:", nano_snap)
    nano_flat = models_dir / "VieNeu-TTS-v3-Nano"
    for line in _materialize(nano_snap, nano_flat):
        print(" ", line)

    # Best-effort clone assets (may 404 on some revisions)
    try:
        from huggingface_hub import hf_hub_download

        for fn in NANO_CLONE:
            try:
                path = hf_hub_download(
                    NANO_REPO, fn, repo_type="model", token=args.token, cache_dir=str(models_dir)
                )
                dest = nano_flat / fn
                if not dest.exists():
                    shutil.copy2(path, dest)
                print(f"  clone asset {fn}: ok")
            except Exception as e:  # noqa: BLE001
                print(f"  clone asset {fn}: skip ({e})")
    except Exception as e:  # noqa: BLE001
        print("clone asset pass skipped:", e)

    if not args.skip_moss:
        print(f"Downloading {MOSS_REPO} ...")
        moss_snap = Path(
            snapshot_download(
                repo_id=MOSS_REPO,
                repo_type="model",
                token=args.token,
                cache_dir=str(models_dir),
            )
        )
        print("MOSS snapshot:", moss_snap)
        moss_flat = models_dir / "MOSS-Audio-Tokenizer-Nano-ONNX"
        for line in _materialize(moss_snap, moss_flat):
            print(" ", line)

    missing = [n for n in NANO_GRAPH if not (nano_flat / n).is_file()]
    if missing:
        print("ERROR: Nano graph still missing:", ", ".join(missing), file=sys.stderr)
        return 1

    print("\nDone. Offline run example:")
    print(f"  export HF_HOME={models_dir}")
    print(f"  export VIENEU_ONNX_DIR={nano_flat}")
    print("  export HF_HUB_OFFLINE=1")
    print("  ./scripts/run_lan_api.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
