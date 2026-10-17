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


def fetch_global_year(year: str, dest: Path) -> bool:
    """Download the global CHIRPS year file resumably and with retries.
    Returns True on success, False if the year doesn't exist remotely
    (404, e.g. a future year). Raises on a genuine, unrecoverable
    download failure so the caller records it as a failure (not a
    silent skip)."""
    url = CHIRPS_URL_TEMPLATE.format(year=year)
    print(f"Fetching {url}")
    print(f"  -> {dest}")

    expected = _remote_size(url)
    if expected is not None:
        print(f"  server reports {expected // (1024**2)} MB")

    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected is not None and have >= expected and have > 0:
            break
        headers = {}
        mode = "wb"
        if have > 0 and expected is not None:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
            print(f"  attempt {attempt}: resuming at {have // (1024**2)} MB")
        else:
            if have > 0:
                dest.unlink(missing_ok=True)
                have = 0
            print(f"  attempt {attempt}: downloading")

        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                if r.status_code == 404:
                    print(f"  404 \u2014 year {year} not yet published. Skipping.")
                    return False
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
                                downloaded % (100 * 1024 * 1024)
                                < 1024 * 1024):
                            pct = 100 * downloaded / expected
                            print(f"  {downloaded // (1024**2)} MB "
                                  f"({pct:.0f}%)")
            have = dest.stat().st_size
            if expected is None or have >= expected:
                last_err = None
                break
            last_err = f"incomplete: have {have} of {expected} bytes"
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
            f"download of CHIRPS {year} failed after "
            f"{MAX_ATTEMPTS} attempts: {last_err}"
        )

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


# Plausible physical range for Caribbean daily precipitation in mm.
# Used only to reject obviously-broken data (all-NA stubs, fill-value
# fields, wrong units), not to filter real values. Daily totals are
# non-negative; an upper bound well above any real daily total catches
# fill-value corruption without rejecting genuine extremes.
PRECIP_MIN_PLAUSIBLE = 0.0
PRECIP_MAX_PLAUSIBLE = 2000.0


def validate_cropped(out_file: Path) -> None:
    """Guard against uploading a hollow or corrupt file. CHC sometimes
    serves a structurally-valid NetCDF whose data array is entirely
    missing. Raise if the cropped data has no finite values, or if none
    of its finite values fall in a physically plausible range. Only the
    presence of *some* real data is required.
    """
    ds = xr.open_dataset(out_file)
    try:
        # Main data variable = the first 3D (time,lat,lon) float var.
        cand = [v for v in ds.data_vars if ds[v].ndim == 3]
        if not cand:
            cand = list(ds.data_vars)
        if not cand:
            raise RuntimeError(
                f"validation: {out_file.name} has no data variables."
            )
        var = cand[0]
        arr = ds[var].values
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
        in_range = ((vals >= PRECIP_MIN_PLAUSIBLE) &
                    (vals <= PRECIP_MAX_PLAUSIBLE))
        frac_ok = float(in_range.mean())
        print(f"  validation: var '{var}', {n_finite} finite values, "
              f"range {vmin:.1f}..{vmax:.1f} mm, "
              f"{100*frac_ok:.0f}% in plausible range")
        if frac_ok == 0.0:
            raise RuntimeError(
                f"validation: {out_file.name} has finite values but NONE "
                f"in the plausible range [{PRECIP_MIN_PLAUSIBLE}, "
                f"{PRECIP_MAX_PLAUSIBLE}] mm (min {vmin:.1f}, "
                f"max {vmax:.1f}). Refusing to upload likely-corrupt data."
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

        # Gate: never upload a hollow/corrupt file over good mirror data.
        validate_cropped(cropped_file)

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
