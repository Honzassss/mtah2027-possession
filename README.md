# Ball possession per frame: More than a Hack 2027, Phase 1

For every frame of the challenge match this code says who has the ball: `H` (home) or `A` (away) while
the ball is alive, `D` while it is dead, and `X` outside the two halves, which are not graded. It reads
the automatic tracking in the challenge's tactical bundle and writes the submission file
(`frame,label`, 199,319 rows).

The model is trained on open data only: tracking of other professional matches (IDSSE, DFL labels),
degraded on purpose to look like the challenge's uncorrected tracking. Nothing is learned from the
challenge match, and none of its data is in this repository. The approach, the possession rules and
the limitations are in the technical document, [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Regenerate the submission file

Tested on macOS (Apple silicon) with the python.org build of Python 3.14.7. On Linux (arm64, and x86-64
under Rosetta) in the `python:3.14.7-slim` container (with `libgomp1`), the same pins write the same
file byte for byte.
On macOS the LightGBM wheel loads OpenMP from Homebrew, so install that first:

```bash
brew install libomp
```

Then, from a clone of this repository:

```bash
python3.14 -m venv .venv
.venv/bin/pip install --only-binary :all: -r requirements.txt
.venv/bin/python -I scripts/predict_match.py <tactical_dir> <team_name>.csv
```

`<tactical_dir>` is the folder of the data bundle that holds `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt` (the homography and the videos are not read). The command prints three
lines:

```
PASS
rows 199319
spells <number of possessions in the file>
```

It writes `<team_name>.csv` and `<team_name>.csv.sha256` beside it, so the result can be checked with
`shasum -a 256 -c <team_name>.csv.sha256`. Before keeping the file it reads it back from disk and checks
it against the format in the challenge's SUBMISSION.md: the header, 199,319 rows, frames 0 to 199318 in
order, only `H`, `A`, `D` and `X`, no `X` on a graded frame, and `D` on the published dead block before
the second-half kick-off. A run that fails prints `FAIL <stage>: <reason>` and leaves any earlier file
untouched. The challenge repository's own `evaluation/validate_submission.py` should then accept the
file as well.

The run is deterministic: the same bundle gives the same bytes. It takes about ten seconds.

## What the command does

1. **Read** the tracking (`pipeline/epts.py`): the metadata decides which field of a raw line belongs to
   which slot and team; positions go to metres on the published pitch and timeline.
2. **Clean** (`pipeline/clean.py`): drop ball observations that imply more than 40 m/s, split tracks at
   gaps and jumps, find the attack direction and the keepers, recompute speeds from positions.
3. **Features at 5 Hz** (`pipeline/features.py`): 141 per 0.2 s step, from the ball, the players near it
   and the teams' shape, looking backwards and forwards; computed in both team views.
4. **Three LightGBM heads** (`pipeline/model.py`): dead ball or live, which team has the ball, and which
   team restarts after a stoppage, each averaged over both views so neither side is favoured.
5. **Decode** (`pipeline/decode.py`): a four-state Viterbi (home, away, dead with home to restart, dead
   with away to restart) in whole milli-nats, so every machine finds the same path. The side to restart
   may change during a stoppage, and the published dead block stays dead.
6. **Write and check** (`pipeline/submission.py`): one label per frame, then the checks above.

`pipeline/predict.py` is that path as one function, `frame_labels(match, load_model("models", "v3"))`.

## The model files

`models/` holds the published model, v3: `v3_decoder.json` with the decoder's transition table, its one
free number (alpha) and the SHA-256 of each head it uses, and the three heads it shares with v2
(`v2_dead.txt`, `v2_team.txt`, `v2_restart.txt`, LightGBM text files). The script refuses heads that do
not match those hashes.

The heads learn from labels of convention T = 1: in live play, a run of one team shorter than one second does
not change possession and stays with the team that had the ball before it, the way the challenge's
rules treat loose and contested balls; the first run after a stoppage stays the restarter's. The rules
give no length for such a run; on open data, T = 1 is the choice that loses least when the truth
follows another one (`analysis/test1_convention.py`).

v3's decoder is v2's with one change: the side to restart may change during a stoppage, at the cost of
staying dead. A dead spell in the labels always takes the side of the play after it, so the counted
table has no such step, and v2's decoder can change its mind about the restarter only through a live
step of 0.2 s, which the metric counts as a spell of its own and never matches. `models/` also keeps
v2 (`--tag v2`) and v0 (`--tag v0`), the same recipe on the DFL labels as they are (T = 0).

## How the model was made

Everything below runs on open data and is only needed to rebuild `models/`, not to regenerate the file:

```bash
.venv/bin/pip install --only-binary :all: -r requirements-twin.txt -r requirements-model.txt
scripts/fetch_external.sh idsse
.venv/bin/python -I scripts/export_idsse.py external/idsse external/processed/idsse
.venv/bin/python -I analysis/model_v0.py build     # degraded copies of the matches and their features
.venv/bin/python -I analysis/model_v0.py cv        # leave one match out: boosting rounds and alpha
.venv/bin/python -I analysis/model_v0.py final     # v0: train on five matches, score the two held out
.venv/bin/python -I analysis/test2_dead_shift.py predict   # v0's held-out predictions, which test 1 reads
.venv/bin/python -I analysis/test1_convention.py labels    # the labels of each convention T
.venv/bin/python -I analysis/test1_convention.py cv        # boosting rounds per T, one match out
.venv/bin/python -I analysis/test1_convention.py sweep     # alpha per T
.venv/bin/python -I analysis/test1_convention.py final     # heads per T on all five, held-out scores
.venv/bin/python -I analysis/model_v2.py bundle            # v2: the T = 1 heads with their decoder
.venv/bin/python -I analysis/model_v3.py bundle            # v3: v2's heads, the restarter may change
```

The stages write fresh sets into `results/models/`; run the script with `--models results/models` to
use them. Five IDSSE matches train the heads and two (J03WMX, J03WN1) are held out. A copy of a match is
degraded the way the challenge's documents describe its tracking (the ball missing on about 41 % of
graded frames, about 20 tracked objects, anonymous track slots, a camera that misplaces a ball in the
air), and the choices are made on the training matches only. On the held-out matches, degraded
the same way and scored against labels of T = 1, v3 reaches spellF1 0.646, v2 0.625 and v0 0.621;
against the DFL labels, v0 reaches 0.556. These are checks on open data, not a forecast for the challenge match.
LightGBM runs deterministically with 8 threads, but trees trained on another machine can differ in
their last digits, which is why the trained heads are committed rather than rebuilt for the
submission. On the machine that trained them, the commands above in a fresh clone with a fresh
virtual environment rebuild every file in `models/` (v0, v2 and v3) byte for byte, in about 46 minutes
after the download. In the `python:3.14.7-slim` container (arm64, and x86-64 under Rosetta),
`tests/check_predict.py` passes as on the Mac: from the features cached on the Mac, the committed models
give back all 54 held-out scores of v0 and of v2 and the 36 gate rows of v3 exactly, and a held-out copy
degraded from scratch in the container gets the same label on every frame.

## Layout

| path | what |
|---|---|
| `scripts/predict_match.py` | the command that writes the submission file |
| `pipeline/` | adapters (`epts.py`, `idsse.py`, `skillcorner.py`), cleaning, features, heads, decoder, file checks; `degrade.py` and `camera.py` are for training only |
| `models/` | the published model, v3, with v2 and v0 beside it |
| `analysis/` | training and evaluation on open data (`model_v0.py`, `model_v2.py`, `model_v3.py`, tests of the design choices) |
| `scripts/match_*.py` | label-free measurements run on the challenge data during development; they print only PASS or FAIL |
| `tests/` | self-checks, run on the open datasets and the stand-in |
| `twin_builder/` | builds the stand-in the self-checks run on: the challenge bundle's file format, filled from the open IDSSE match J03WMX (into `twin/`, not in the repository; see `docs/external-data.md`) |
| `docs/` | the technical document, the open datasets and the manifests that pin them |
| `upstream/` | not in the repository: a clone of the challenge's public repository, which comments cite and two scripts read (`match_test8.py` its baseline file, `tests/check_twin.py` its `games.py`) |

## Data and licences

- **IDSSE (DFL) open data**, the training data: Bassek et al. 2025, "An integrated dataset of
  synchronized spatiotemporal and event data in elite soccer", Scientific Data 12, 195,
  https://doi.org/10.1038/s41597-025-04505-y. CC BY 4.0, data owner Deutsche Fussball Liga (DFL). The
  labels are DFL's own ball status and ball possession fields. Mirror used:
  https://huggingface.co/datasets/pysport/idsse-data.
- **SkillCorner open data** (MIT) and **Metrica Sports sample data** (no licence file) are used only to
  test the adapters; nothing is trained on them.
- The challenge's data is not in this repository and must never be added: it is not redistributable.

[docs/external-data.md](docs/external-data.md) lists the sources, and `docs/manifests/` pins the files
that were fetched.
