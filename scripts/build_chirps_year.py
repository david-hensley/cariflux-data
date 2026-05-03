#!/usr/bin/env python3
"""
build_chirps_year.py

Fetches CHIRPS v2 daily precipitation for a year or range of
years, crops each year to the Caribbean bbox, and pushes the
resulting NetCDFs to the Hugging Face dataset
cariflux/chirps-caribbean.

Note on ocean masking
---------------------
We do NOT apply a separate land mask. CHIRPS is a land-only
product and already returns NaN over the deep ocean.

Inputs (env vars):
    HF_TOKEN           Hugging Face access token. Required.

    For manual runs:
      MODE             "single" or "range" (matches the workflow
                       UI input). Default "single".
      YEAR             Single year to fetch (e.g. "2015").
      YEAR_START       Range mode: first year.
      YEAR_END         Range mode: last year (inclusive).
      SKIP_EXISTING    "true" to skip years already on HF.

    For scheduled runs (cron):
      SCHEDULED_REFRESH "true" forces a single-year run for the
                       current calendar year, with skip_existing
                       disabled (overwrite). All other inputs
                       are ignored.

Output:
    Pushes one <YEAR>.nc per year to cariflux/chirps-caribbean.

Caribbean bbox:
    lon: -89 to -58
    lat:   9.5 to 28
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from huggingface_hub import HfApi

# ---- Config -----------------------------------------------------
CARIBBEAN_BBOX = dict(lon_min=-89.0, lon_max=-58.0,
                      lat_min=9.5,   lat_max=28.0)

CHIRPS_URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_daily/netcdf/p05/chirps-v2.0.{year}.days_p05.nc"
)

HF_REPO_ID = "cariflux/chirps-caribbean"


def fetch_global_year(year: str, dest: Path) -> bool:
    """Download the global CHIRPS year file (~1 GB) to disk.
    Returns True on success, False if the year doesn't exist on
    the remote (e.g. asking for a future year)."""
    url = CHIRPS_URL_TEMPLATE.format(year=year)
    print(f"Fetching {url}")
    print(f"  -> {dest}")

    with requests.get(url, stream=True, timeout=600) as r:
        if r.status_code == 404:
            print(f"  404 — year {year} not yet published. Skipping.")
            return False
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (100 * 1024 * 1024) < 1024 * 1024:
                    pct = 100 * downloaded / total
                    print(f"  {downloaded // (1024**2)} MB ({pct:.0f}%)")

    size_mb = dest.stat().st_size / (1024 ** 2)
    print(f"Downloaded {size_mb:.1f} MB")
    return True


def crop_to_caribbean(in_file: Path, out_file: Path) -> dict:
    print(f"Opening {in_file} ...")
    ds = xr.open_dataset(in_file)
    print(f"Global dataset shape: {dict(ds.sizes)}")

    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude"  if "latitude"  in ds.coords else "lat"

    cropped = ds.sel({
        lon_name: slice(CARIBBEAN_BBOX["lon_min"], CARIBBEAN_BBOX["lon_max"]),
        lat_name: slice(CARIBBEAN_BBOX["lat_min"], CARIBBEAN_BBOX["lat_max"]),
    })
    print(f"Cropped shape:        {dict(cropped.sizes)}")

    encoding = {
        var: {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32",
            "_FillValue": np.float32(-9999.0),
        }
        for var in cropped.data_vars
    }

    print(f"Writing {out_file} ...")
    cropped.to_netcdf(out_file, encoding=encoding)
    out_size_mb = out_file.stat().st_size / (1024 ** 2)
    print(f"Compressed output: {out_size_mb:.1f} MB")

    return {
        "global_size_mb":  in_file.stat().st_size / (1024 ** 2),
        "cropped_size_mb": out_size_mb,
        "cropped_dims":    dict(cropped.sizes),
    }


def push_to_hf(local_file: Path, remote_path: str, token: str) -> None:
    print(f"Uploading {local_file.name} -> hf://datasets/{HF_REPO_ID}/{remote_path}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=remote_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Update {remote_path}",
    )
    print("Upload complete.")


def list_existing_years(token: str) -> set:
    """Return set of years currently in the HF dataset."""
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"  warning: could not list HF repo files: {e}")
        return set()
    years = set()
    for f in files:
        if f.endswith(".nc"):
            stem = f.rsplit(".", 1)[0]
            if stem.isdigit() and len(stem) == 4:
                years.add(stem)
    return years


def process_one_year(year: str, token: str) -> bool:
    """Download, crop, upload one year. Returns True on success,
    False if the year wasn't available remotely."""
    print(f"\n--- Year {year} ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        global_file  = tmpdir / f"chirps-v2.0.{year}.days_p05.nc"
        cropped_file = tmpdir / f"{year}.nc"

        ok = fetch_global_year(year, global_file)
        if not ok:
            return False

        summary = crop_to_caribbean(global_file, cropped_file)
        print(f"Summary: {summary}")

        push_to_hf(cropped_file, f"{year}.nc", token)
    return True


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is required.", file=sys.stderr)
        return 1

    # Scheduled refresh path: always rebuild the CURRENT year, since
    # that's where new daily values accumulate as CHIRPS publishes
    # them. Historical years (1981 through last calendar year) don't
    # change, so we don't touch them on cron runs.
    scheduled_refresh = (
        os.environ.get("SCHEDULED_REFRESH", "false").lower() == "true"
    )

    if scheduled_refresh:
        from datetime import datetime, timezone
        current_year = str(datetime.now(timezone.utc).year)
        print(f"=== Scheduled refresh of CHIRPS {current_year} ===")
        print(f"Bbox: {CARIBBEAN_BBOX}")
        years         = [current_year]
        skip_existing = False  # force-overwrite the current year
    else:
        skip_existing = os.environ.get("SKIP_EXISTING", "false").lower() == "true"

        # Resolve which years to process from manual-trigger inputs
        single_year = os.environ.get("YEAR", "").strip()
        year_start  = os.environ.get("YEAR_START", "").strip()
        year_end    = os.environ.get("YEAR_END",   "").strip()
        mode        = os.environ.get("MODE", "single").strip()

        if mode == "single" and single_year:
            years = [single_year]
        elif mode == "range" and year_start and year_end:
            years = [str(y) for y in range(int(year_start),
                                           int(year_end) + 1)]
        elif single_year:
            # MODE not set; fall back to single_year if provided
            years = [single_year]
        elif year_start and year_end:
            years = [str(y) for y in range(int(year_start),
                                           int(year_end) + 1)]
        else:
            print("ERROR: must set YEAR or both YEAR_START and YEAR_END.",
                  file=sys.stderr)
            return 1

        print(f"=== Building CHIRPS years {years[0]}..{years[-1]} "
              "for Caribbean ===")
        print(f"Bbox: {CARIBBEAN_BBOX}")
        print(f"Skip existing: {skip_existing}")

    if skip_existing:
        existing = list_existing_years(token)
        print(f"Years already on HF: {sorted(existing)}")
        years = [y for y in years if y not in existing]
        print(f"Years to process:    {years}")

    succeeded = []
    skipped   = []
    failed    = []

    for y in years:
        try:
            ok = process_one_year(y, token)
            if ok:
                succeeded.append(y)
            else:
                skipped.append(y)
        except Exception as e:
            print(f"  ERROR processing year {y}: {e}")
            failed.append(y)

    print("\n=== Final summary ===")
    print(f"  succeeded: {succeeded}")
    print(f"  skipped (404 / not yet published): {skipped}")
    print(f"  failed: {failed}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
