#!/usr/bin/env python3
"""
build_chirts_year.py

Fetches CHIRTS-ERA5 daily Tmax and Tmin for a year or range of
years, crops to the Caribbean bbox, and pushes the resulting
NetCDFs to the Hugging Face dataset cariflux/chirts-caribbean.

Two files are written per year:
    <YEAR>-tmax.nc
    <YEAR>-tmin.nc

This mirrors the CHIRPS pipeline structure. CHIRTS-ERA5 is the
operational product (1980-near-present, 5-day latency); the older
CHIRTS-daily v1.0 covered only 1983-2016 and is not used here.

Inputs (env vars):
    HF_TOKEN           Hugging Face access token. Required.

    For manual runs:
      MODE             "single" or "range". Default "single".
      YEAR             Single year to fetch (e.g. "2015").
      YEAR_START       Range mode: first year.
      YEAR_END         Range mode: last year (inclusive).
      SKIP_EXISTING    "true" to skip years already on HF.

    For scheduled runs (cron):
      SCHEDULED_REFRESH "true" forces a single-year run for the
                       current calendar year. All other inputs
                       ignored.

Output:
    Pushes two .nc files per year to cariflux/chirts-caribbean.

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

# CHIRTS-ERA5 URL pattern. The operational product lives under
# /experimental/CHIRTS-ERA5/. Two variables per year.
# Example file names follow the pattern:
#   Tmax.YYYY.nc / Tmin.YYYY.nc (uppercase variable, by year)
CHIRTS_URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/"
    "global_netcdf_p05/{var}/{var}.{year}.nc"
)

HF_REPO_ID = "cariflux/chirts-caribbean"

# The two variables we mirror. Source-side capitalization is
# Tmax/Tmin (matching the URL); we rename to lowercase tmax/tmin
# in the output NetCDFs so the R-side variable names follow the
# CARIFLUX convention.
VARIABLES = [
    {"src_name": "Tmax", "out_name": "tmax",
     "longname": "CHIRTS-ERA5 daily maximum 2 m temperature"},
    {"src_name": "Tmin", "out_name": "tmin",
     "longname": "CHIRTS-ERA5 daily minimum 2 m temperature"},
]


def fetch_global_year(var: str, year: str, dest: Path) -> bool:
    """Download the global CHIRTS-ERA5 year file for one variable.
    Returns True on success, False if the year doesn't exist
    remotely (e.g. asking for a future year)."""
    url = CHIRTS_URL_TEMPLATE.format(var=var, year=year)
    print(f"Fetching {url}")
    print(f"  -> {dest}")

    with requests.get(url, stream=True, timeout=600) as r:
        if r.status_code == 404:
            print(f"  404 — {var} {year} not yet published. Skipping.")
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


def crop_to_caribbean(in_file: Path, out_file: Path,
                       src_var_name: str, out_var_name: str,
                       longname: str) -> dict:
    """Open global CHIRTS-ERA5 year, crop to Caribbean bbox,
    rename the variable, write compressed NetCDF.
    """
    print(f"Opening {in_file} ...")
    ds = xr.open_dataset(in_file)
    print(f"Global dataset shape: {dict(ds.sizes)}")
    print(f"Available data vars:  {list(ds.data_vars)}")

    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude"  if "latitude"  in ds.coords else "lat"

    # CHIRTS-ERA5 latitudes may be ascending or descending. Use
    # min/max ordering with sortby() to make the slice robust.
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        # Latitude descending — slice in descending order
        cropped = ds.sel({
            lon_name: slice(CARIBBEAN_BBOX["lon_min"], CARIBBEAN_BBOX["lon_max"]),
            lat_name: slice(CARIBBEAN_BBOX["lat_max"], CARIBBEAN_BBOX["lat_min"]),
        })
    else:
        cropped = ds.sel({
            lon_name: slice(CARIBBEAN_BBOX["lon_min"], CARIBBEAN_BBOX["lon_max"]),
            lat_name: slice(CARIBBEAN_BBOX["lat_min"], CARIBBEAN_BBOX["lat_max"]),
        })
    print(f"Cropped shape:        {dict(cropped.sizes)}")

    # Find the source variable. CHIRTS-ERA5 sometimes uses just
    # the variable name as a data var; sometimes there are extras
    # like crs/projection. Pick the first 3D variable matching the
    # source name pattern.
    candidate_vars = [v for v in cropped.data_vars
                      if v.lower() == src_var_name.lower()]
    if not candidate_vars:
        # Fallback: any 3D float var
        candidate_vars = [v for v in cropped.data_vars
                          if cropped[v].ndim == 3]
    if not candidate_vars:
        raise RuntimeError(
            f"Could not find a {src_var_name} variable in {in_file}. "
            f"Available: {list(cropped.data_vars)}"
        )
    src_var = candidate_vars[0]
    print(f"Source variable name: {src_var} -> renaming to {out_var_name}")

    # Subset to just the variable of interest, then rename
    out_ds = cropped[[src_var]].rename({src_var: out_var_name})
    # Tag CF-style metadata
    out_ds[out_var_name].attrs["units"] = "degree_Celsius"
    out_ds[out_var_name].attrs["long_name"] = longname

    encoding = {
        out_var_name: {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32",
            "_FillValue": np.float32(-9999.0),
        }
    }

    print(f"Writing {out_file} ...")
    out_ds.to_netcdf(out_file, encoding=encoding)
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


def list_existing_year_files(token: str) -> set:
    """Return set of (year, var) tuples currently in HF."""
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"  warning: could not list HF repo files: {e}")
        return set()

    year_var_pairs = set()
    for f in files:
        if f.endswith(".nc"):
            stem = f.rsplit(".", 1)[0]
            # Expecting "<YYYY>-tmax" or "<YYYY>-tmin"
            parts = stem.split("-")
            if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4:
                year_var_pairs.add((parts[0], parts[1]))
    return year_var_pairs


def process_one_year_one_var(year: str, var_spec: dict,
                             token: str) -> bool:
    """Download, crop, upload one year's worth of one variable.
    Returns True on success, False if the year wasn't available."""
    var_src  = var_spec["src_name"]
    var_out  = var_spec["out_name"]
    longname = var_spec["longname"]

    print(f"\n--- {var_out} {year} ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        global_file  = tmpdir / f"{var_src}.{year}.nc"
        cropped_file = tmpdir / f"{year}-{var_out}.nc"

        ok = fetch_global_year(var_src, year, global_file)
        if not ok:
            return False

        summary = crop_to_caribbean(global_file, cropped_file,
                                     var_src, var_out, longname)
        print(f"Summary: {summary}")

        push_to_hf(cropped_file, f"{year}-{var_out}.nc", token)
    return True


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is required.", file=sys.stderr)
        return 1

    scheduled_refresh = (
        os.environ.get("SCHEDULED_REFRESH", "false").lower() == "true"
    )

    if scheduled_refresh:
        from datetime import datetime, timezone
        current_year = str(datetime.now(timezone.utc).year)
        print(f"=== Scheduled refresh of CHIRTS-ERA5 {current_year} ===")
        print(f"Bbox: {CARIBBEAN_BBOX}")
        years         = [current_year]
        skip_existing = False
    else:
        skip_existing = os.environ.get("SKIP_EXISTING", "false").lower() == "true"

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
            years = [single_year]
        elif year_start and year_end:
            years = [str(y) for y in range(int(year_start),
                                           int(year_end) + 1)]
        else:
            print("ERROR: must set YEAR or both YEAR_START and YEAR_END.",
                  file=sys.stderr)
            return 1

        print(f"=== Building CHIRTS-ERA5 years {years[0]}..{years[-1]} ===")
        print(f"Bbox: {CARIBBEAN_BBOX}")
        print(f"Skip existing: {skip_existing}")

    # Build (year, var) work list
    work_items = []
    for y in years:
        for v in VARIABLES:
            work_items.append((y, v))

    if skip_existing:
        existing = list_existing_year_files(token)
        print(f"Year/var pairs already on HF: {len(existing)}")
        work_items = [(y, v) for (y, v) in work_items
                      if (y, v["out_name"]) not in existing]
        print(f"Items to process: {len(work_items)}")

    succeeded = []
    skipped   = []
    failed    = []

    for (y, v) in work_items:
        label = f"{v['out_name']} {y}"
        try:
            ok = process_one_year_one_var(y, v, token)
            if ok:
                succeeded.append(label)
            else:
                skipped.append(label)
        except Exception as e:
            print(f"  ERROR processing {label}: {e}")
            failed.append(label)

    print("\n=== Final summary ===")
    print(f"  succeeded: {succeeded}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
