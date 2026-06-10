# cariflux-data

Pipelines that build and refresh the data mirrors used by the
[CARIFLUX](https://github.com/david-hensley/cariflux) R package.

## What lives here

This repo holds **only the pipeline code**. The actual data lives
on Hugging Face:

- [cariflux/chirps-caribbean](https://huggingface.co/datasets/cariflux/chirps-caribbean)
  — daily precipitation, Caribbean bbox, from CHIRPS v2.
- [cariflux/chirts-caribbean](https://huggingface.co/datasets/cariflux/chirts-caribbean)
  — daily temperature, Caribbean bbox, from CHIRTS-ERA5.
- [cariflux/soilgrids-caribbean](https://huggingface.co/datasets/cariflux/soilgrids-caribbean)
  — gridded soil properties, Caribbean bbox, from SoilGrids 2.0.
- [cariflux/landcover-caribbean](https://huggingface.co/datasets/cariflux/landcover-caribbean)
  — 10 m land cover tiles, Caribbean region, from ESA WorldCover v200.

## Why a mirror

The original CHIRPS data lives at `data.chc.ucsb.edu`, which uses
CrowdSec to deter scrapers. That works for one-off fetches by
researchers but breaks programmatic use from R packages —
periodically, IPs get banned and the package breaks for that user
until the ban lifts.

SoilGrids' upstream infrastructure at `files.isric.org` has shown
intermittent reliability issues: VRT index files and individual
tile files sporadically become unreachable, and the WCS endpoint
has produced corrupted tiled-TIFF outputs. The Caribbean mirror
insulates CARIFLUX users from these upstream issues.

ESA WorldCover is served from the AWS Open Data bucket, which is
reliable. We mirror it anyway for **consistency and
reproducibility**: with all four data types on Hugging Face, the
CARIFLUX data layer is a single, self-contained, version-pinned
snapshot that does not depend on any external provider's bucket
paths or program lifecycle remaining stable over the years.

We mirror Caribbean subsets on Hugging Face so users of CARIFLUX
always get instant, frictionless access.

The Caribbean bbox we mirror for CHIRPS / CHIRTS is approximately:

- Longitude: -89 to -58
- Latitude:    9.5 to 27

For SoilGrids it is slightly wider:

- Longitude: -90 to -58
- Latitude:    9 to 28

Both cover the Greater Antilles, Lesser Antilles, the southern
Bahamas, coastal Yucatán/Belize, and northern South America.

## Citations

CARIFLUX users should cite the underlying data sources:

CHIRPS:
> Funk, C. *et al.* (2015). The climate hazards infrared
> precipitation with stations — a new environmental record for
> monitoring extremes. *Scientific Data*, 2, 150066.
> https://doi.org/10.1038/sdata.2015.66

CHIRTS:
> Funk, C. *et al.* (2019). A high-resolution 1983–2016
> Tmax climate data record based on infrared temperatures and
> stations by the Climate Hazards Center. *Journal of Climate*,
> 32(17), 5639–5658.

SoilGrids:
> Poggio, L. *et al.* (2021). SoilGrids 2.0: producing soil
> information for the globe with quantified spatial
> uncertainty. *SOIL*, 7, 217–240.
> https://doi.org/10.5194/soil-7-217-2021

ESA WorldCover:
> Zanaga, D. *et al.* (2022). ESA WorldCover 10 m 2021 v200.
> https://doi.org/10.5281/zenodo.7254221

CHIRPS is in the public domain (CC0). CHIRTS is freely
distributed (we redistribute under the same terms). SoilGrids
is licensed CC-BY 4.0 — attribution is required and provided
above. ESA WorldCover is licensed CC-BY 4.0 — attribution is
required and provided above.

## Refresh pipelines

CHIRPS and CHIRTS are refreshed via scheduled GitHub Actions
workflows. The schedule and current status are visible on the
[Actions tab](../../actions). Failed runs send notification
emails to the repo owner.

**SoilGrids and land cover are not on a refresh schedule.**
SoilGrids 2.0 and ESA WorldCover v200 are static published
datasets; their mirrors do not need periodic rebuilds. A new
upstream version (e.g. SoilGrids 3.0 or a new WorldCover release)
would trigger a manual rebuild.

## Build / rebuild procedures

### CHIRPS

```bash
HF_TOKEN=... YEAR=2024 python scripts/build_chirps_year.py
```

See script docstring for range-mode inputs.

### CHIRTS

```bash
HF_TOKEN=... YEAR=2024 python scripts/build_chirts_year.py
```

### SoilGrids

SoilGrids is a multi-step manual rebuild because the upstream API
is unreliable. The flow is:

1. **Acquisition (manual web interface).** Open
   https://maps.isric.org/ in a browser. For each of the 8
   properties (`bdod`, `clay`, `silt`, `sand`, `soc`, `cfvo`, `cec`, `phh2o`)
   and each of the 6 depths (`0-5cm`, `5-15cm`, `15-30cm`,
   `30-60cm`, `60-100cm`, `100-200cm`), download two tiles
   covering the Caribbean as W/E halves:

   | Tile | xmin | ymin | xmax | ymax |
   |------|------|------|------|------|
   | W    | -90    | 9 | -73.95 | 28 |
   | E    | -74.05 | 9 | -58    | 28 |

   The 0.1° overlap zone around -74 longitude ensures the
   subsequent mosaic step produces a seamless result.

   That's 96 downloads. Specify SoilGrids' native 250 m output
   resolution explicitly to keep each tile under the WCS
   16,384-pixel cap.

   Name the downloads `{tile}_{property}_{depth}.tif` where
   `tile` is `east` or `west` and `depth` uses underscores
   (e.g. `west_silt_5_15.tif`).

2. **Mosaicking (R, in the cariflux repo).** Run
   `data-raw/mosaic_soilgrids_for_mirror.R` from the cariflux
   R package's root. This reads the 96 input tiles, mosaics
   each W/E pair into a Caribbean-wide GeoTIFF, validates each
   output's value range, and writes 48 COG-compressed
   `{property}_{depth}.tif` files to
   `data-raw/cache/soilgrids_caribbean_mirror/mosaicked/`.

3. **Upload (this repo).** Create a *local* working directory
   for staging — this is intentionally `.gitignore`'d, so its
   contents stay on your machine and only end up on Hugging Face,
   never on GitHub:

   ```bash
   mkdir -p input_soilgrids  # local-only, in .gitignore

   # Copy the 48 mosaicked .tif files into the staging dir
   cp /path/to/cariflux/data-raw/cache/soilgrids_caribbean_mirror/mosaicked/*.tif input_soilgrids/

   HF_TOKEN=... python scripts/build_soilgrids.py
   ```

   The script reads from `input_soilgrids/`, uploads to Hugging
   Face, and that's the end of it. Nothing in `input_soilgrids/`
   is ever committed to this repo. By default the script skips
   any files already on the HF dataset, so reruns are idempotent.

### Land cover

Land cover is a single automated step -- no manual acquisition
needed, because ESA WorldCover is served reliably from AWS Open
Data. The script downloads each Caribbean land tile from AWS and
re-uploads it to Hugging Face unchanged:

```bash
HF_TOKEN=... python scripts/build_landcover.py
```

The script scans every 3x3 degree tile in the Caribbean bounding
box, skips ocean-only cells (which return 404 on AWS), downloads
the land-containing tiles, and pushes them to
`cariflux/landcover-caribbean` with their original WorldCover
filenames. By default each downloaded tile is deleted locally
after a successful upload (set `KEEP_LOCAL=true` to retain them in
`input_landcover/`). Reruns skip tiles already on HF.

## License

Pipeline code: MIT (see LICENSE).
Data: each mirror inherits the upstream license.
- CHIRPS: CC0-1.0
- CHIRTS: CC0-1.0 (per CHC distribution terms)
- SoilGrids: CC-BY-4.0 (attribution required, see citations)
