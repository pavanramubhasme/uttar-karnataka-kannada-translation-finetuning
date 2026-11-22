#!/usr/bin/env python3
"""
Download North Karnataka (Vijayapura/Bijapur + Bagalkot) Kannada speech
from Project Vaani (IISc / ARTPARK, CC BY 4.0) on Hugging Face.

ONE-TIME SETUP
  1. pip install -U datasets huggingface_hub
  2. Log in to Hugging Face and click "Agree and access repository" on BOTH:
       https://huggingface.co/datasets/ARTPARK-IISc/Vaani
       https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part
  3. Create a READ token at https://huggingface.co/settings/tokens, then run:
       hf auth login          (older versions: huggingface-cli login)

USAGE
  python download_vaani_bijapur.py --list           # show matching subsets only
  python download_vaani_bijapur.py                  # 500 clips per subset (quick sample)
  python download_vaani_bijapur.py --max 0          # everything (can be tens of GB)
  python download_vaani_bijapur.py --transcribed    # only clips that have transcripts
  python download_vaani_bijapur.py --no-zip         # skip creating the .zip

OUTPUT
  vaani_uttara_karnataka/<subset>/audio/*.wav (or original format)
  vaani_uttara_karnataka/<subset>/metadata.csv   (transcript, speaker info, etc.)
  vaani_uttara_karnataka.zip
"""
import argparse
import csv
import json
import os
import shutil
import sys

try:
    from datasets import Audio, Image, get_dataset_config_names, load_dataset
except ImportError:
    sys.exit("Missing library. Run:  pip install -U datasets huggingface_hub")

MAIN_REPO = "ARTPARK-IISc/Vaani"
TRANSCRIBED_REPO = "ARTPARK-IISc/Vaani-transcription-part"
DISTRICT_KEYWORDS = ["bijapur", "vijayapura", "vijaypur", "bagalkot", "bagalkote"]


def auth_hint(err):
    msg = str(err)
    if any(s in msg for s in ("401", "403", "gated", "Gated", "authenticated", "Unauthorized")):
        print(
            "\nAccess denied. Make sure you (1) clicked 'Agree and access repository' on the\n"
            "dataset page while logged in, and (2) ran 'hf auth login' with a read token.\n"
        )


def find_configs(repo, keywords):
    try:
        names = get_dataset_config_names(repo)
    except Exception as err:  # noqa: BLE001
        print(f"Could not list subsets of {repo}: {err}")
        auth_hint(err)
        return [], []
    hits = [n for n in names if any(k in n.lower() for k in keywords)]
    return names, hits


def make_row_filter(keywords):
    """Keep a row if any text field mentions the district (used when subsets are per-language)."""
    def keep(example):
        for value in example.values():
            if isinstance(value, str) and any(k in value.lower() for k in keywords):
                return True
        return False
    return keep


def to_cell(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def save_split(ds, out_dir, max_n, row_filter=None):
    """Write audio files + metadata.csv for one (streaming) split. Returns clip count."""
    feats = ds.features
    if feats is None and hasattr(ds, "_resolve_features"):
        ds = ds._resolve_features()
        feats = ds.features
    feats = feats or {}

    audio_cols = [c for c, f in feats.items() if isinstance(f, Audio)] or (
        ["audio"] if not feats or "audio" in feats else []
    )
    image_cols = [c for c, f in feats.items() if isinstance(f, Image)]

    # Don't decode audio/images: write the original bytes straight to disk.
    for col in audio_cols:
        if col in feats:
            ds = ds.cast_column(col, Audio(decode=False))
    for col in image_cols:
        ds = ds.cast_column(col, Image(decode=False))

    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metadata.csv")

    n = 0
    fieldnames = None
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Kannada shows in Excel
        writer = None
        for ex in ds:
            if row_filter and not row_filter(ex):
                continue
            rec = {}
            for col in audio_cols:
                a = ex.get(col)
                if not a:
                    continue
                data = a.get("bytes")
                src = a.get("path") or ""
                if data is None and src and os.path.exists(src):
                    with open(src, "rb") as f:
                        data = f.read()
                if data is None:
                    continue
                ext = os.path.splitext(src)[1] or ".wav"
                prefix = "" if len(audio_cols) == 1 else f"{col}_"
                fname = f"{prefix}{n:06d}{ext}"
                with open(os.path.join(audio_dir, fname), "wb") as f:
                    f.write(data)
                rec[f"{col}_file"] = f"audio/{fname}"
                rec[f"{col}_original_name"] = os.path.basename(src)
            for k, v in ex.items():
                if k in audio_cols or k in image_cols:
                    continue
                rec[k] = to_cell(v)
            if writer is None:
                fieldnames = list(rec.keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
            writer.writerow(rec)
            n += 1
            if n % 50 == 0:
                print(f"    {n} clips saved...", flush=True)
            if max_n and n >= max_n:
                break
    if n == 0:
        os.remove(csv_path)
    return n


def download(repo, configs, out_root, max_n, row_filter=None):
    total = 0
    for cfg in configs:
        print(f"\n== {repo} :: {cfg}")
        try:
            dsd = load_dataset(repo, cfg, streaming=True)
        except Exception as err:  # noqa: BLE001
            print(f"  Failed to open: {err}")
            auth_hint(err)
            continue
        for split, ds in dsd.items():
            out_dir = os.path.join(out_root, f"{repo.split('/')[-1]}__{cfg}__{split}")
            print(f"  split '{split}' -> {out_dir}")
            try:
                n = save_split(ds, out_dir, max_n, row_filter)
            except Exception as err:  # noqa: BLE001
                print(f"  Error while downloading: {err}")
                auth_hint(err)
                continue
            print(f"  done: {n} clips")
            total += n
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max", type=int, default=500, help="clips per subset/split (0 = all). Default 500")
    p.add_argument("--transcribed", action="store_true", help="only the transcribed subset")
    p.add_argument("--list", action="store_true", help="only list matching subsets, download nothing")
    p.add_argument("--out", default="vaani_uttara_karnataka", help="output folder")
    p.add_argument("--no-zip", action="store_true", help="do not create a .zip at the end")
    args = p.parse_args()

    plan = []  # (repo, configs, row_filter)

    if not args.transcribed:
        names, hits = find_configs(MAIN_REPO, DISTRICT_KEYWORDS)
        print(f"{MAIN_REPO}: {len(names)} subsets, matching: {hits or 'none'}")
        if names and not hits:
            print("  Bagalkot/Bijapur not found by name. Karnataka subsets available:")
            print("  ", [n for n in names if "karnataka" in n.lower()])
        if hits:
            plan.append((MAIN_REPO, hits, None))

    names, hits = find_configs(TRANSCRIBED_REPO, DISTRICT_KEYWORDS)
    print(f"{TRANSCRIBED_REPO}: {len(names)} subsets, matching: {hits or 'none'}")
    if hits:
        plan.append((TRANSCRIBED_REPO, hits, None))
    elif names:
        kn = [n for n in names if "kannada" in n.lower()]
        if kn:
            print(f"  No district subset; will filter Kannada subset(s) {kn} for Bijapur/Bagalkot rows.")
            plan.append((TRANSCRIBED_REPO, kn, make_row_filter(DISTRICT_KEYWORDS)))

    if args.list or not plan:
        if not plan:
            print("\nNothing to download (see messages above).")
        return

    total = 0
    for repo, cfgs, filt in plan:
        total += download(repo, cfgs, args.out, args.max, filt)

    print(f"\nTotal clips saved: {total}  ->  {os.path.abspath(args.out)}")
    if total and not args.no_zip:
        zip_path = shutil.make_archive(args.out, "zip", args.out)
        print(f"Zipped: {zip_path}")


if __name__ == "__main__":
    main()