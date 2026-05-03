# cariflux-data

Pipelines that build and refresh the data mirrors used by the
[CARIFLUX](https://github.com/david-hensley/cariflux) R package.

## What lives here

This repo holds **only the pipeline code**. The actual data lives
on Hugging Face:

- [cariflux/chirps-caribbean](https://huggingface.co/datasets/cariflux/chirps-caribbean)
  — daily precipitation, Caribbean bbox, from CHIRPS v2.

## Why a mirror

The original CHIRPS data lives at `data.chc.ucsb.edu`, which uses
CrowdSec to deter scrapers. That works for one-off fetches by
researchers but breaks programmatic use from R packages —
periodically, IPs get banned and the package breaks for that user
until the ban lifts. We mirror a Caribbean subset on Hugging Face
so users of CARIFLUX always get instant, frictionless access.

The Caribbean bbox we mirror is approximately:

- Longitude: -87 to -55
- Latitude:    7 to  27

This covers the Greater Antilles, Lesser Antilles, the southern
Bahamas, coastal Yucatán/Belize, and northern South America.

## Citation

CARIFLUX users should cite the underlying CHIRPS data source:

> Funk, C. *et al.* (2015). The climate hazards infrared
> precipitation with stations — a new environmental record for
> monitoring extremes. *Scientific Data*, 2, 150066.
> https://doi.org/10.1038/sdata.2015.66

CHIRPS is in the public domain (CC0). We redistribute as a
convenience; original credit and authority remain with the UCSB
Climate Hazards Center.

## Refresh pipeline

The Hugging Face dataset is refreshed via a scheduled GitHub
Actions workflow. The schedule and current status are visible on
the [Actions tab](../../actions). Failed runs send notification
emails to the repo owner.

## License

Pipeline code: MIT (see LICENSE).
Data: CC0-1.0, matching CHIRPS.
