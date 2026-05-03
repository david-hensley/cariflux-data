#!/usr/bin/env python3
"""
build_chirps_year.py

Bootstrap fetch for ONE year of CHIRPS v2 daily precipitation,
cropped to the Caribbean region. Output is a compressed NetCDF
file pushed to the Hugging Face dataset cariflux/chirps-caribbean.

This script is intentionally minimal — it's the small-taste
pipeline to verify everything works end-to-end. Once verified,
we'll generalize to all years and add ocean masking.

Inputs (env vars):
    HF_TOKEN   Hugging Face access token with write access to
               the cariflux org. Required.
    YEAR       Year to fetch (e.g. "2015"). Required.

Output:
    Pushes <YEAR>.nc to cariflux/chirps-caribbean on Hugging Face.

Caribbean bbox used:
    lon: -87 to -55    (Belize/Yucatan to eastern Lesser Antilles)
    lat:   7 to  27    (Trinidad/N. Venezuela to southern Florida)
"""

import os
import sys
import tempfile
from pathlib import Path

import requests
import xarray as xr
from huggingface_hub import HfApi

# -- Config --------------------------------------------------------
CARIBBEAN_BBOX = dict(lon_min=-87.0, lon_max=-55.0,
                      lat_min=7.0,   lat_max=27.0)

CHIRPS_URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_daily/netcdf/p05/chirps-v2.0.{year}.days_p05.nc"
)

HF_REPO_ID = "cariflux/chirps-caribbean"


def fetch_global_year(year: str, dest: Path) -> None:
    """Download the global CHIRPS year file (~1 GB) to disk.

    Runs from GitHub Actions where the IP is not on UCSB's
    CrowdSec blocklist, so the download just works.
    """
    url = CHIRPS_URL_TEMPLATE.format(year=year)
    print(f"Fetching {url}")
    print(f"  -> {dest}")

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    pct = 100 * downloaded / total
                    print(f"  {downloaded // (1024**2)} MB ({pct:.0f}%)")

    size_mb = dest.stat().st_size / (1024 ** 2)
    print(f"Downloaded {size_mb:.1f} MB")


def crop_to_caribbean(in_file: Path, out_file: Path) -> dict:
    """Read the global CHIRPS year file, crop to Caribbean bbox,
    write a compressed NetCDF.

    Returns a small summary dict.
    """
    print(f"Opening {in_file} ...")
    ds = xr.open_dataset(in_file)
    print(f"Global dataset shape: {dict(ds.sizes)}")

    # Crop. CHIRPS uses 'longitude' / 'latitude' as the dim names
    # but historically that's varied; handle both.
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude"  if "latitude"  in ds.coords else "lat"

    cropped = ds.sel({
        lon_name: slice(CARIBBEAN_BBOX["lon_min"], CARIBBEAN_BBOX["lon_max"]),
        lat_name: slice(CARIBBEAN_BBOX["lat_min"], CARIBBEAN_BBOX["lat_max"]),
    })
    print(f"Cropped shape:        {dict(cropped.sizes)}")

    # Compress on write. zlib level 5 is a sensible default.
    encoding = {
        var: {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32",
        }
        for var in cropped.data_vars
    }

    print(f"Writing {out_file} ...")
    cropped.to_netcdf(out_file, encoding=encoding)
    out_size_mb = out_file.stat().st_size / (1024 ** 2)
    print(f"Compressed output: {out_size_mb:.1f} MB")

    return {
        "global_size_mb": in_file.stat().st_size / (1024 ** 2),
        "cropped_size_mb": out_size_mb,
        "cropped_dims":   dict(cropped.sizes),
    }


def push_to_hf(local_file: Path, remote_path: str, token: str) -> None:
    """Upload a file to the chirps-caribbean dataset on HF."""
    print(f"Uploading {local_file.name} -> hf://datasets/{HF_REPO_ID}/{remote_path}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=remote_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {remote_path} (small-taste bootstrap)",
    )
    print("Upload complete.")


def main() -> int:
    year = os.environ.get("YEAR")
    if not year:
        print("ERROR: YEAR env var is required.", file=sys.stderr)
        return 1
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is required.", file=sys.stderr)
        return 1

    print(f"=== Building CHIRPS year {year} for Caribbean ===")
    print(f"Bbox: {CARIBBEAN_BBOX}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        global_file  = tmpdir / f"chirps-v2.0.{year}.days_p05.nc"
        cropped_file = tmpdir / f"{year}.nc"

        fetch_global_year(year, global_file)
        summary = crop_to_caribbean(global_file, cropped_file)

        print("\n=== Crop summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        push_to_hf(cropped_file, f"{year}.nc", token)

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
