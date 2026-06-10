#!/usr/bin/env python3
"""
build_landcover.py

Mirrors ESA WorldCover 10 m land-cover tiles for the Caribbean
region to the Hugging Face dataset cariflux/landcover-caribbean.

Unlike the SoilGrids mirror, no reprocessing is done. WorldCover
tiles are already clean cloud-optimized GeoTIFFs (3x3 degree, byte-
valued class codes, proper NA handling) served from the AWS Open
Data bucket. We download each land-containing tile and re-upload it
to HF unchanged, preserving the original tile filenames so the
cariflux R package's existing per-tile fetch logic works against
the mirror with only a base-URL change.

Why mirror a reliable AWS source at all?
    Consistency and reproducibility. With CHIRPS, CHIRTS, and
    SoilGrids all on HF, putting land cover there too means the
    CARIFLUX data layer is a single, self-contained, version-pinned
    snapshot. AWS Open Data is reliable today, but bucket paths and
    program lifecycles change over years; a mirror insulates against
    that.

Tile coverage
    WorldCover only publishes tiles that contain land. We iterate
    over every 3x3 degree grid cell in the Caribbean bounding box
    and attempt to fetch each; ocean-only cells return 404 and are
    skipped automatically, so the final mirror contains exactly the
    land-containing tiles with no manual tile list to maintain.

SoilGrids 2.0 and WorldCover v200 are both static products, so this
script -- like build_soilgrids.py -- has no scheduled-refresh mode.

Inputs (env vars):
    HF_TOKEN       Hugging Face access token. Required.
    VERSION        WorldCover version. Default "v200" (2021 release).
    YEAR           WorldCover year. Default "2021" (matches v200).
    SKIP_EXISTING  "true" to skip tiles already on HF (default
                   "true"; lets you rerun without re-uploading).
    KEEP_LOCAL     "true" to keep downloaded tiles in ./input_landcover/
                   after upload (default "false"; deletes each tile
                   after a successful push to save disk).

Output:
    Pushes ESA_WorldCover_10m_<year>_<version>_<TILE>_Map.tif files
    to cariflux/landcover-caribbean.

Caribbean bbox (grid of 3x3 deg tiles scanned):
    lon: -90 to -58
    lat:   9 to  28
"""

import math
import os
import sys
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi

# ---- Config -----------------------------------------------------
HF_REPO_ID = "cariflux/landcover-caribbean"

# Caribbean bounding box. Tiles are anchored at multiples of 3 deg.
BBOX = dict(lon_min=-90, lon_max=-58, lat_min=9, lat_max=28)

# AWS Open Data bucket for ESA WorldCover.
AWS_URL_TEMPLATE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    "{version}/{year}/map/"
    "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
)


def lower3(v: float) -> int:
    """Round down to the nearest multiple of 3 (tile SW corner)."""
    return 3 * math.floor(v / 3)


def tile_grid(bbox: dict) -> list:
    """All 3x3 deg tile IDs whose SW corner is in the bbox grid."""
    lat_starts = range(lower3(bbox["lat_min"]),
                       lower3(bbox["lat_max"]) + 1, 3)
    lon_starts = range(lower3(bbox["lon_min"]),
                       lower3(bbox["lon_max"]) + 1, 3)
    tiles = []
    for lat in lat_starts:
        for lon in lon_starts:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return tiles


def tile_filename(tile: str, version: str, year: str) -> str:
    return f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"


def list_existing_files(token: str) -> set:
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"  warning: could not list HF repo files: {e}")
        return set()
    return {f for f in files if f.endswith(".tif")}


def ensure_repo(token: str) -> None:
    api = HfApi(token=token)
    try:
        api.repo_info(HF_REPO_ID, repo_type="dataset")
        print(f"HF dataset {HF_REPO_ID} exists, proceeding.")
    except Exception:
        print(f"HF dataset {HF_REPO_ID} not found, creating it.")
        api.create_repo(HF_REPO_ID, repo_type="dataset",
                        private=False, exist_ok=True)


def tile_exists_on_aws(url: str) -> bool:
    """HEAD request; True if AWS serves this tile (200), else False."""
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False


def download_tile(url: str, dest: Path) -> bool:
    """Stream-download a tile to dest. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            if r.status_code != 200:
                print(f"  AWS returned HTTP {r.status_code}")
                return False
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        # sanity: a real tile is many MB, a 404/error page is tiny
        if dest.stat().st_size < 1e5:
            print(f"  downloaded file too small "
                  f"({dest.stat().st_size} bytes); treating as failure")
            dest.unlink(missing_ok=True)
            return False
        return True
    except requests.RequestException as e:
        print(f"  download error: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def push_to_hf(local_file: Path, remote_path: str, token: str) -> None:
    size_mb = local_file.stat().st_size / (1024 ** 2)
    print(f"  uploading -> hf://datasets/{HF_REPO_ID}/{remote_path} "
          f"({size_mb:.1f} MB)")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=remote_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {remote_path}",
    )


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is required.", file=sys.stderr)
        return 1

    version = os.environ.get("VERSION", "v200")
    year    = os.environ.get("YEAR", "2021")
    skip_existing = (
        os.environ.get("SKIP_EXISTING", "true").lower() == "true"
    )
    keep_local = (
        os.environ.get("KEEP_LOCAL", "false").lower() == "true"
    )

    staging = Path("./input_landcover")
    staging.mkdir(exist_ok=True)

    tiles = tile_grid(BBOX)

    print("=== Mirroring ESA WorldCover Caribbean tiles ===")
    print(f"Version      : {version}  Year: {year}")
    print(f"HF repo      : {HF_REPO_ID}")
    print(f"Grid tiles   : {len(tiles)} (ocean-only tiles will 404 "
          f"and be skipped)")
    print(f"Skip existing: {skip_existing}")
    print(f"Keep local   : {keep_local}")
    print()

    ensure_repo(token)
    existing = list_existing_files(token) if skip_existing else set()
    if existing:
        print(f"Already on HF: {len(existing)} tiles\n")

    uploaded = []
    skipped  = []
    no_land  = []
    failed   = []

    for i, tile in enumerate(tiles, 1):
        fname = tile_filename(tile, version, year)
        url   = AWS_URL_TEMPLATE.format(version=version, year=year,
                                        tile=tile)
        prefix = f"[{i:2d}/{len(tiles)}] {tile}"

        if skip_existing and fname in existing:
            print(f"{prefix}: already on HF, skip")
            skipped.append(tile)
            continue

        if not tile_exists_on_aws(url):
            print(f"{prefix}: no tile on AWS (ocean-only), skip")
            no_land.append(tile)
            continue

        print(f"{prefix}: downloading...")
        dest = staging / fname
        if not download_tile(url, dest):
            print(f"{prefix}: DOWNLOAD FAILED")
            failed.append(tile)
            continue

        try:
            push_to_hf(dest, fname, token)
            uploaded.append(tile)
        except Exception as e:
            print(f"{prefix}: UPLOAD FAILED: {e}")
            failed.append(tile)
            # leave the local file so a rerun can re-push it
            continue

        if not keep_local:
            dest.unlink(missing_ok=True)

    print("\n=== Final summary ===")
    print(f"  uploaded   : {len(uploaded)}")
    print(f"  skipped    : {len(skipped)} (already on HF)")
    print(f"  no-land    : {len(no_land)} (ocean-only, not on AWS)")
    print(f"  failed     : {len(failed)}  {failed if failed else ''}")

    if failed:
        print("\nRerun to retry failures (cached uploads are skipped).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
