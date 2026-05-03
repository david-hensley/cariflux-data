#!/usr/bin/env python3
"""
build_chirps_year.py

Fetches one year of CHIRPS v2 daily precipitation, crops to the
Caribbean bbox, applies an ocean mask (cells far from land set to
NA so NetCDF compression is more efficient), and pushes the
resulting NetCDF to the Hugging Face dataset
cariflux/chirps-caribbean.

Inputs (env vars):
    HF_TOKEN   Hugging Face access token with write access to
               the cariflux org. Required.
    YEAR       Year to fetch (e.g. "2015"). Required.

Output:
    Pushes <YEAR>.nc to cariflux/chirps-caribbean on Hugging Face.

Caribbean bbox:
    lon: -89 to -58    (Honduran Bay Islands to eastern Lesser Antilles)
    lat:   9.5 to 28   (Trinidad/ABC islands to top of Bahamas)
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from huggingface_hub import HfApi

# Geometry libraries for the land mask
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import from_origin

# ---- Config -----------------------------------------------------
CARIBBEAN_BBOX = dict(lon_min=-89.0, lon_max=-58.0,
                      lat_min=9.5,   lat_max=28.0)

CHIRPS_URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_daily/netcdf/p05/chirps-v2.0.{year}.days_p05.nc"
)

HF_REPO_ID = "cariflux/chirps-caribbean"

# How far from land (in degrees) to keep cells. ~0.25 deg is
# roughly 25 km in the Caribbean. Anything further offshore is
# effectively "open ocean" and not useful for island hydrology.
LAND_BUFFER_DEG = 0.25


def fetch_global_year(year: str, dest: Path) -> None:
    """Download the global CHIRPS year file (~1 GB) to disk."""
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


def build_land_mask(lon_arr: np.ndarray, lat_arr: np.ndarray,
                    buffer_deg: float) -> np.ndarray:
    """Create a 2D boolean mask: True for cells within
    `buffer_deg` of any landmass.

    Uses Natural Earth's medium-resolution land polygons.
    """
    print(f"Building land mask (buffer = {buffer_deg} deg)...")

    ne_url = (
        "https://naciscdn.org/naturalearth/110m/physical/"
        "ne_110m_land.zip"
    )
    print(f"  fetching coastlines from {ne_url}")
    land = gpd.read_file(ne_url)
    print(f"  loaded {len(land)} land polygons globally")

    # Crop the global land geometry to a slightly larger box than
    # our AOI so we don't lose buffer near edges
    aoi = box(lon_arr.min() - buffer_deg - 1,
              lat_arr.min() - buffer_deg - 1,
              lon_arr.max() + buffer_deg + 1,
              lat_arr.max() + buffer_deg + 1)
    land_crop = land.clip(aoi)
    print(f"  {len(land_crop)} land polygons in / near AOI")

    # Buffer the land. Note: degrees as units are not equal-area
    # but for our purposes a flat-degree buffer is fine — Caribbean
    # latitudes are far enough from the poles for this not to matter.
    print(f"  buffering land by {buffer_deg} deg")
    buffered = land_crop.geometry.buffer(buffer_deg)
    union = unary_union(buffered)

    # Build a regular raster over the CHIRPS grid and rasterize
    # the buffered land into it.
    n_lat = len(lat_arr)
    n_lon = len(lon_arr)
    dlon  = float(lon_arr[1] - lon_arr[0])
    dlat  = float(lat_arr[1] - lat_arr[0])

    # rasterio transform expects upper-left origin, positive dy
    # downward. Build the transform based on whether lat axis
    # ascends or descends.
    if dlat < 0:
        # Lat decreasing: row 0 is northernmost
        top    = float(lat_arr[0]) - dlat / 2  # upper edge of row 0
        left   = float(lon_arr[0]) - dlon / 2
        transform = from_origin(left, top, abs(dlon), abs(dlat))
    else:
        # Lat increasing: row 0 is southernmost. Rasterize in
        # image coords (top = max lat) and flip at the end.
        top    = float(lat_arr[-1]) + dlat / 2
        left   = float(lon_arr[0])  - dlon / 2
        transform = from_origin(left, top, abs(dlon), abs(dlat))

    mask_img = rasterize(
        [(union, 1)],
        out_shape=(n_lat, n_lon),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    if dlat > 0:
        mask_img = np.flipud(mask_img)

    n_kept = int(mask_img.sum())
    n_total = int(mask_img.size)
    print(f"  mask kept {n_kept:,} / {n_total:,} cells "
          f"({100 * n_kept / n_total:.1f}%)")

    return mask_img


def crop_and_mask(in_file: Path, out_file: Path) -> dict:
    """Open global CHIRPS year, crop to Caribbean, mask cells far
    from land to NA, write compressed NetCDF.
    """
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

    # Build land mask aligned to the cropped grid
    lon_vals = cropped[lon_name].values
    lat_vals = cropped[lat_name].values
    land_mask = build_land_mask(lon_vals, lat_vals, LAND_BUFFER_DEG)

    # Apply mask to all data variables. Cells outside the mask
    # become NaN; NetCDF compression handles uniform NaN runs
    # very efficiently.
    data_vars_pre = {k: int(np.isfinite(cropped[k].values).sum())
                     for k in cropped.data_vars}

    for varname in cropped.data_vars:
        var = cropped[varname]
        if lat_name in var.dims and lon_name in var.dims:
            cropped[varname] = var.where(
                xr.DataArray(
                    land_mask,
                    coords={lat_name: cropped[lat_name],
                            lon_name: cropped[lon_name]},
                    dims=(lat_name, lon_name),
                )
            )

    data_vars_post = {k: int(np.isfinite(cropped[k].values).sum())
                      for k in cropped.data_vars}
    print(f"Valid cells before mask: {data_vars_pre}")
    print(f"Valid cells after  mask: {data_vars_post}")

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
        "land_buffer_deg": LAND_BUFFER_DEG,
    }


def push_to_hf(local_file: Path, remote_path: str, token: str) -> None:
    print(f"Uploading {local_file.name} -> hf://datasets/{HF_REPO_ID}/{remote_path}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=remote_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {remote_path} (bbox + land mask)",
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
    print(f"Land buffer: {LAND_BUFFER_DEG} deg")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        global_file  = tmpdir / f"chirps-v2.0.{year}.days_p05.nc"
        cropped_file = tmpdir / f"{year}.nc"

        fetch_global_year(year, global_file)
        summary = crop_and_mask(global_file, cropped_file)

        print("\n=== Crop / mask summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        push_to_hf(cropped_file, f"{year}.nc", token)

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
