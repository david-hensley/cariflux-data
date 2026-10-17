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
# /experimental/CHIRTS-ERA5/. Two variables: tmax and tmin.
# Each variable has its own subdirectory, with daily NetCDFs
# organized as one file per year:
#   /experimental/CHIRTS-ERA5/<lower>/netcdf/daily/
#       CHIRTS-ERA5.daily_<Title>.<YYYY>.nc
#
# Note the case mismatch: directories are lowercase (tmax, tmin),
# but the filename uses Tmax/Tmin. Each year file is ~6.7 GiB
# global float64; we crop to ~MB after Caribbean clip.
CHIRTS_URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/"
    "{var_dir}/netcdf/daily/CHIRTS-ERA5.daily_{var_file}.{year}.nc"
)

HF_REPO_ID = "cariflux/chirts-caribbean"

# The two variables we mirror. CHIRTS-ERA5's filesystem uses
# lowercase directory names (tmax, tmin) but Title-case in
# filenames (Tmax, Tmin). We track both. Output variable name
# in our NetCDF is lowercase, matching CARIFLUX convention.
VARIABLES = [
    {"var_dir": "tmax", "var_file": "Tmax", "out_name": "tmax",
     "longname": "CHIRTS-ERA5 daily maximum 2 m temperature"},
    {"var_dir": "tmin", "var_file": "Tmin", "out_name": "tmin",
     "longname": "CHIRTS-ERA5 daily minimum 2 m temperature"},
]


# ---- Download robustness settings -------------------------------
# CHC's server (data.chc.ucsb.edu) is unreliable for the large global
# year files: connections stall mid-stream (silent socket) or drop
# part-way (IncompleteRead). These settings make the download resilient:
#   - a (connect, read) timeout so a stalled socket is killed after
#     READ_TIMEOUT seconds of silence instead of hanging forever;
#   - resume via HTTP Range requests so a mid-stream drop continues from
#     where it stopped rather than restarting;
#   - several attempts with linear backoff;
#   - a final size check against the server's Content-Length so a
#     truncated transfer is detected and retried, never passed on.
CONNECT_TIMEOUT = 60          # seconds to establish the connection
READ_TIMEOUT    = 120         # seconds of silence before giving up a read
MAX_ATTEMPTS    = 5
BACKOFF_SEC     = 20          # multiplied by attempt number


def _remote_size(url: str):
    """Return the server's Content-Length in bytes, or None if the
    server doesn't report it. A HEAD request avoids downloading."""
    try:
        h = requests.head(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                          allow_redirects=True)
        if h.status_code == 200:
            cl = h.headers.get("content-length")
            return int(cl) if cl is not None else None
    except requests.RequestException:
        pass
    return None


def fetch_global_year(var_dir: str, var_file: str,
                       year: str, dest: Path) -> bool:
    """Download the global CHIRTS-ERA5 year file for one variable,
    resumably and with retries. Returns True on success, False if the
    year doesn't exist remotely (404, e.g. a future year). Raises on a
    genuine, unrecoverable download failure so the caller records it as
    a failure (not a silent skip)."""
    url = CHIRTS_URL_TEMPLATE.format(
        var_dir=var_dir, var_file=var_file, year=year
    )
    print(f"Fetching {url}")
    print(f"  -> {dest}")

    expected = _remote_size(url)
    if expected is not None:
        print(f"  server reports {expected // (1024**2)} MB")

    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = dest.stat().st_size if dest.exists() else 0
        # If we already have the whole file from a previous attempt, stop.
        if expected is not None and have >= expected and have > 0:
            break
        headers = {}
        mode = "wb"
        if have > 0 and expected is not None:
            # Resume: ask only for the remaining bytes, append to file.
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
            print(f"  attempt {attempt}: resuming at "
                  f"{have // (1024**2)} MB")
        else:
            if have > 0:
                # Can't resume safely without knowing total; restart.
                dest.unlink(missing_ok=True)
                have = 0
            print(f"  attempt {attempt}: downloading")

        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                if r.status_code == 404:
                    print(f"  404 — {var_file} {year} not yet "
                          f"published. Skipping.")
                    return False
                # 416 = range not satisfiable: we already have it all.
                if r.status_code == 416:
                    break
                r.raise_for_status()
                if expected is None:
                    cl = r.headers.get("content-length")
                    if cl is not None:
                        expected = int(cl) + have
                downloaded = have
                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if (expected and
                                downloaded % (200 * 1024 * 1024)
                                < 1024 * 1024):
                            pct = 100 * downloaded / expected
                            print(f"  {downloaded // (1024**2)} MB "
                                  f"({pct:.0f}%)")
            # Completed a pass without exception. Check completeness.
            have = dest.stat().st_size
            if expected is None or have >= expected:
                last_err = None
                break
            last_err = (f"incomplete: have {have} of {expected} bytes")
            print(f"  attempt {attempt} {last_err}")
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"  attempt {attempt} failed: {last_err}")

        if attempt < MAX_ATTEMPTS:
            wait = BACKOFF_SEC * attempt
            print(f"  waiting {wait}s before retry")
            import time
            time.sleep(wait)

    if last_err is not None:
        raise RuntimeError(
            f"download of {var_file} {year} failed after "
            f"{MAX_ATTEMPTS} attempts: {last_err}"
        )

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


# Plausible physical range for Caribbean daily air temperature in
# degrees Celsius. Used only to reject obviously-broken data (all-NA
# stubs, wrong units, fill-value fields), not to filter real values.
TEMP_MIN_PLAUSIBLE = -10.0
TEMP_MAX_PLAUSIBLE = 55.0


def validate_cropped(out_file: Path, out_var_name: str) -> None:
    """Guard against uploading a hollow or corrupt file. CHC sometimes
    serves a structurally-valid NetCDF whose data array is entirely
    missing (this is how an all-NA stub reached the mirror before this
    check existed). Raise if the cropped data has no finite values, or
    if none of its finite values fall in a physically plausible range.

    Only the sign of there being *some* real data is required -- we do
    not reject individual out-of-range cells, only a file that is
    entirely empty or entirely implausible.
    """
    ds = xr.open_dataset(out_file)
    try:
        if out_var_name not in ds.data_vars:
            raise RuntimeError(
                f"validation: variable {out_var_name} missing from "
                f"{out_file.name}; vars present: {list(ds.data_vars)}"
            )
        arr = ds[out_var_name].values
        finite = np.isfinite(arr)
        n_finite = int(finite.sum())
        if n_finite == 0:
            raise RuntimeError(
                f"validation: {out_file.name} has NO finite data values "
                f"(all-NA/empty). Refusing to upload a hollow file -- "
                f"the upstream source likely served a stub."
            )
        vals = arr[finite]
        vmin = float(vals.min())
        vmax = float(vals.max())
        in_range = ((vals >= TEMP_MIN_PLAUSIBLE) &
                    (vals <= TEMP_MAX_PLAUSIBLE))
        frac_ok = float(in_range.mean())
        print(f"  validation: {n_finite} finite values, "
              f"range {vmin:.1f}..{vmax:.1f} C, "
              f"{100*frac_ok:.0f}% in plausible range")
        if frac_ok == 0.0:
            raise RuntimeError(
                f"validation: {out_file.name} has finite values but NONE "
                f"in the plausible range [{TEMP_MIN_PLAUSIBLE}, "
                f"{TEMP_MAX_PLAUSIBLE}] C (min {vmin:.1f}, max {vmax:.1f}). "
                f"Refusing to upload likely-corrupt data."
            )
    finally:
        ds.close()


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
    Returns True on success, False if the year wasn't available.

    The TemporaryDirectory context manager guarantees that the
    big global file (~6.7 GiB for CHIRTS) is deleted before this
    function returns, so disk usage stays bounded between calls.
    """
    var_dir  = var_spec["var_dir"]
    var_file = var_spec["var_file"]
    var_out  = var_spec["out_name"]
    longname = var_spec["longname"]

    print(f"\n--- {var_out} {year} ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        global_file  = tmpdir / f"CHIRTS-ERA5.daily_{var_file}.{year}.nc"
        cropped_file = tmpdir / f"{year}-{var_out}.nc"

        ok = fetch_global_year(var_dir, var_file, year, global_file)
        if not ok:
            return False

        summary = crop_to_caribbean(global_file, cropped_file,
                                     var_file, var_out, longname)
        print(f"Summary: {summary}")

        # Gate: never upload a hollow/corrupt file over good mirror data.
        validate_cropped(cropped_file, var_out)

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
