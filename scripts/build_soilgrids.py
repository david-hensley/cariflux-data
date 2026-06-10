#!/usr/bin/env python3
"""
build_soilgrids.py

Uploads the 36 pre-mosaicked SoilGrids Caribbean GeoTIFFs (6
properties x 6 depths) to the Hugging Face dataset
cariflux/soilgrids-caribbean.

Acquisition note
----------------
Unlike build_chirps_year.py / build_chirts_year.py, this script
does NOT fetch from upstream. SoilGrids' native ISRIC endpoints
(files.isric.org VRT tree, vsicurl access via GDAL) have been
unreliable for programmatic access -- individual tile files and
even whole VRT indices are intermittently broken. We obtain the
data manually via the ISRIC WCS web interface
(https://maps.isric.org/) by downloading the Caribbean region as
W and E halves per property/depth, then mosaicking them locally
in R using:

    cariflux/data-raw/mosaic_soilgrids_for_mirror.R

That R script produces 36 GeoTIFFs ({property}_{depth}.tif)
covering the Caribbean bbox at SoilGrids' native 250 m. This
Python script just uploads those 36 files to HF.

Because SoilGrids 2.0 is a static dataset (no time dimension,
not republished on a schedule), this script has no scheduled-
refresh mode. It is invoked manually after a rebuild.

Inputs (env vars):
    HF_TOKEN       Hugging Face access token. Required.
    SOURCE_DIR     Directory containing the 36 mosaicked .tif
                   files. Default: ./input_soilgrids/
    SKIP_EXISTING  "true" to skip files already on HF (default
                   "true"; this lets you rerun without re-
                   uploading work that's already done).

Output:
    Pushes 36 {property}_{depth}.tif files to
    cariflux/soilgrids-caribbean.

Caribbean bbox (covered by the mosaics):
    lon: -90 to -58
    lat:   9 to  28
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

# ---- Config -----------------------------------------------------
HF_REPO_ID = "cariflux/soilgrids-caribbean"

PROPERTIES = ["bdod", "clay", "silt", "sand", "soc", "cfvo",
              "cec", "phh2o"]
DEPTHS     = ["0-5cm", "5-15cm", "15-30cm",
              "30-60cm", "60-100cm", "100-200cm"]


def list_existing_files(token: str) -> set:
    """Return set of .tif filenames currently in the HF dataset."""
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"  warning: could not list HF repo files: {e}")
        return set()
    return {f for f in files if f.endswith(".tif")}


def ensure_repo(token: str) -> None:
    """Create the HF dataset if it does not already exist."""
    api = HfApi(token=token)
    try:
        api.repo_info(HF_REPO_ID, repo_type="dataset")
        print(f"HF dataset {HF_REPO_ID} exists, proceeding.")
    except Exception:
        print(f"HF dataset {HF_REPO_ID} not found, creating it.")
        api.create_repo(HF_REPO_ID, repo_type="dataset",
                        private=False, exist_ok=True)


def push_to_hf(local_file: Path, remote_path: str, token: str) -> None:
    print(f"Uploading {local_file.name} "
          f"-> hf://datasets/{HF_REPO_ID}/{remote_path}")
    size_mb = local_file.stat().st_size / (1024 ** 2)
    print(f"  size: {size_mb:.1f} MB")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=remote_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {remote_path}",
    )
    print("  done.")


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is required.", file=sys.stderr)
        return 1

    source_dir = Path(os.environ.get("SOURCE_DIR", "./input_soilgrids"))
    skip_existing = (
        os.environ.get("SKIP_EXISTING", "true").lower() == "true"
    )

    if not source_dir.exists():
        print(f"ERROR: source dir does not exist: {source_dir}",
              file=sys.stderr)
        return 1

    print(f"=== Uploading SoilGrids Caribbean mirror ===")
    print(f"Source dir   : {source_dir.resolve()}")
    print(f"HF repo      : {HF_REPO_ID}")
    print(f"Skip existing: {skip_existing}")
    print()

    ensure_repo(token)
    existing = list_existing_files(token) if skip_existing else set()
    if existing:
        print(f"Already on HF: {sorted(existing)}\n")

    succeeded = []
    skipped   = []
    failed    = []
    missing   = []

    for prop in PROPERTIES:
        for depth in DEPTHS:
            fname = f"{prop}_{depth}.tif"
            local_path = source_dir / fname

            if not local_path.exists():
                print(f"  MISSING from source: {fname}")
                missing.append(fname)
                continue

            if skip_existing and fname in existing:
                print(f"  skip (on HF already): {fname}")
                skipped.append(fname)
                continue

            try:
                push_to_hf(local_path, fname, token)
                succeeded.append(fname)
            except Exception as e:
                print(f"  ERROR uploading {fname}: {e}")
                failed.append(fname)

    print("\n=== Final summary ===")
    print(f"  uploaded : {len(succeeded)}")
    print(f"  skipped  : {len(skipped)}")
    print(f"  missing  : {len(missing)}  {missing if missing else ''}")
    print(f"  failed   : {len(failed)}   {failed if failed else ''}")

    return 0 if not failed and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
