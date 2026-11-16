# External data

Open datasets used for development. None of it comes from the competition's match, and none of it
is committed: everything sits under the gitignored `external/`. `scripts/fetch_external.sh` fetches
all of it again and checks it. Manifests that pin the exact content are in `docs/manifests/`.

| source | where | pinned by | licence |
|---|---|---|---|
| Metrica sample data | `external/metrica` | commit in `docs/manifests/metrica.commit` | no licence file |
| IDSSE (DFL) open data | `external/idsse` | `docs/manifests/idsse_manifest.tsv`, `idsse_listing.json` | CC BY 4.0 |
| SkillCorner open data | `external/skillcorner` | `docs/manifests/skillcorner.commit`, `skillcorner_lfs.tsv` | MIT |

## Metrica sample data

- URL: https://github.com/metrica-sports/sample-data
- Commit: `e706dd506b360d69d9d123d5b8026e7294b13996`
- Content: three anonymised matches with tracking and events. Game 3 is in the FIFA/EPTS format.
- Licence: the repository has no licence file. Its README asks for responsible use and for the
  source to be acknowledged in anything public.
- Labels: tracking and events come from Metrica Sports. The README does not say how the events are
  produced. There is no per-frame possession label; it would have to be derived from the events.
- Use here: the structure of `Sample_Game_3_metadata.xml` is the template for the twin's ATD
  metadata. No Metrica numbers enter the twin.

## IDSSE (DFL) open data

- URL: https://huggingface.co/datasets/pysport/idsse-data, a mirror of
  https://doi.org/10.6084/m9.figshare.28196177 (Figshare blocks scripted downloads).
- Paper: Bassek et al. 2025, "An integrated dataset of synchronized spatiotemporal and event data
  in elite soccer", Scientific Data 12, 195, https://doi.org/10.1038/s41597-025-04505-y
- Pinned by: `docs/manifests/idsse_listing.json` (the Hugging Face tree listing) and
  `docs/manifests/idsse_manifest.tsv` (path, size, digest). Seven position files are verified
  against their LFS sha256. The other 16 files are not under LFS, so their git blob sha1 is verified.
- Content: 7 Bundesliga and 2. Bundesliga matches of 2022/23, 21 data files, 2,627,505,889 bytes,
  plus README and .gitattributes. Match ids: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9.
- Licence: CC BY 4.0, data owner Deutsche Fussball Liga (DFL). Credit the paper and the DFL.
- Labels: each ball frame carries `BallStatus` (0 inactive, 1 active) and `BallPossession` (1 home,
  2 away). They come with the official DFL position feed, collected by Sportec Solutions with the
  TRACAB Gen5 multi-camera system; the paper defines possession by the player in control of the
  ball. The sources read here do not say how much of each flag is automatic and how much operators
  set. Event data comes from trained human operators.
- Use here: `scripts/export_idsse.py` turns these flags into H/A/D frame labels, written to
  `external/processed/idsse/`. Per-match figures are in `docs/manifests/idsse_export_stats.tsv`.
  The twin (`twin_builder/`) is built from match J03WMX, 1. FC Koeln vs FC Bayern Muenchen.

## SkillCorner open data

- URL: https://github.com/SkillCorner/opendata
- Commit: `4340d274572876239c154c90bc507a9b3250a656`
- Pinned by: `docs/manifests/skillcorner.commit` and `docs/manifests/skillcorner_lfs.tsv` (path, size
  and sha256 of the 20 LFS files, 1,823,448,331 bytes, taken from the pointers in the git objects).
  The real files come from media.githubusercontent.com because git-lfs is not installed here.
- Content: 20 tracking files from 2024/25 Australian A-League matches, one directory per match. The
  repository's README still says 10 matches.
- Licence: MIT. The README asks users to credit SkillCorner.
- Labels: SkillCorner's own computer-vision pipeline on broadcast video. Each frame has a
  `possession` field naming a player and a group. Accuracy is not documented in what was read here,
  and it is a different provider from the competition's.
- Use here: tests the adapters (`tests/check_adapters.py`); nothing is trained on it, and the twin
  does not depend on it.

## How this relates to the competition

External data is used to develop and test code. The twin in `twin/` is a synthetic stand-in built
from the IDSSE match above and marked `TWIN_NOT_REAL_DATA.txt`. Its labels come from DFL flags, not
from the competition's corrected tracking, so anything tuned on it must be justified separately.
The competition's match data is never mixed into these folders or processed by these scripts.
