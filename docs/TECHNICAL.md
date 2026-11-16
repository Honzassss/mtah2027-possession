# Possession from automatic tracking

**More than a Hack 2027, Phase 1: technical document.**
Team: **Gridminds**. Code: https://github.com/Honzassss/mtah2027-possession.

## Summary

We label every frame of the challenge match from its automatic tracking alone
(`BAR-ATH_tactical_atd.xml` and `_atd.txt`), with a model that is never fitted to that match: it
was trained, chosen and frozen on open data before it first ran on the challenge bundle. It learns
from open tracking of seven other professional matches with a possession label on every frame
(IDSSE, with DFL's labels): five to train on, two held out. That tracking is degraded on purpose to
look like the challenge's: the ball goes missing in long blocks, players become anonymous track
slots, and a virtual camera misplaces balls in the air. Features at 5 Hz feed three gradient-boosted
classifiers (is the ball dead, which team has it, which team restarts), and a four-state Viterbi
decoder turns their probabilities into possessions. In live play, a run of one team shorter than a
second does not change possession (the convention T = 1, Section 4.5).

On the two held-out matches, degraded the same way and scored against labels of that convention,
the shipped model (v3) scores spellF1 **0.646**, the mean over the two simulators we gate on
(Section 5.1); against DFL's labels as they are, 0.555. Those are checks on open data, not a
forecast for the challenge match. Every number in this document comes from open data. No label of
the challenge match is used to train, select or tune anything; the public five minutes serve only
to check the pipeline (Section 2).

## 1. Approach

### 1.1 The pipeline

| stage | code | what it does |
|---|---|---|
| read | `pipeline/epts.py` | Our own FIFA/EPTS parser. The metadata decides which field of a raw line belongs to which slot, each slot's team comes from its `teamId`, and no slot range is assumed. Positions become metres. |
| clean | `pipeline/clean.py` | Keep the largest set of ball observations that never implies more than 40 m/s; split tracks at gaps and at jumps over 25 m/s; find each half's attack direction and the keepers; keep at most 11 tracks per team; recompute speeds from positions (the raw speed fields are ignored). |
| features | `pipeline/features.py` | 141 features per 0.2 s step, in the home side's attacking frame, computed as they are and with the teams swapped. |
| heads | `pipeline/model.py` | Three LightGBM classifiers: **dead** (every step), **team** (home has the ball, on live steps), **restart** (home has the ball when play resumes, on dead steps). |
| decode | `pipeline/decode.py` | Viterbi over H, A, D-H and D-A, one half at a time, in integer milli-nats. |
| write | `pipeline/submission.py` | Each step's label on each of its frames, `X` outside the halves, `D` on the published dead stretch. The file is read back and checked before it is kept. |

**Features** describe the ball against the nearest players of each team (distance to the nearest
two, the same distance along the camera ray, players within 3 m and 10 m, relative speed); the ball
itself (seen or not, position, distance beyond the lines, speed); each team's shape (centroid,
spread, speed, players running, forward velocity, depth of the last line, length and width of the
block, the keeper, how close the opponents are); touches (time since and until a player of each
team was within 1.5 m of the ball); gaps (time since and until the ball was seen, what it was doing
at both ends, and a ball interpolated across the gap, measured against the players there); and
windows of 1, 3 and 10 s backwards and forwards over most of these. Left out on purpose: the clock,
the camera's view polygon, ball height and player identities.

**Heads.** LightGBM, deterministic on 8 CPU threads (learning rate 0.05, 63 leaves, at least 400
rows per leaf, L2 10), trained on the labels of convention T = 1. Steps within 1 s of a label
change weigh three times as much. Each head is trained on both team views, with the label flipped
for the swapped one, and predicts the mean of both. The boosting rounds are the median
early-stopping round over folds that leave one training match out (dead 508, team 291, restart
144).

**Decoder.** The state posteriors are products of the heads: H is (1 − p_dead) · p_home, A is
(1 − p_dead) · (1 − p_home), D-H is p_dead · p_restart and D-A is p_dead · (1 − p_restart).
Emissions are hybrid-HMM scaled likelihoods, α · (log posterior − log prior). Neighbouring steps
share most of their windows, so their evidence is far from independent, and α is the decoder's one
free number: 1.4, chosen on the training matches' out-of-fold predictions, where spellF1 stays
within 0.003 of its best from α = 1 to 2. Transitions are counted on the training labels and made
symmetric between home and away. A dead state can only resume as live play for the team it names,
but during a stoppage D-H and D-A may swap at the cost of staying dead, so the restart head can
change its mind without a live step in between (Section 4.3). After the logarithms everything is
integer arithmetic, with ties going to the lower state, so the path does not depend on
floating-point summation order.

### 1.2 Why this design

1. **Learn from labelled open data, not from rules set by eye.** The challenge's truth is
   mechanical, but its thresholds are not published, the match has no labels we may use, and the
   public window is not a tuning set. A model trained elsewhere was the only way we saw to learn how
   possession behaves while the ball is unseen. IDSSE is the open dataset we found with a ball
   status and a possession flag on every frame.
2. **Degrade the training data instead of repairing the test data.** The challenge's tracking may
   not be corrected by hand, and corrections could not be scored against its truth. Open data, by
   contrast, can be spoiled in known ways and scored against its own labels. A model that has only
   seen clean tracking learns cues that automatic tracking lacks: a ball that is always there, and
   a ball parked still during stoppages.
3. **Trees on windowed features, not a deep sequence model.** Seven labelled matches from one
   tracking system are few for a deep model. Gradient-boosted trees handle missing values natively
   (the ball is missing on 41 % of graded frames), train on a CPU in minutes and run
   deterministically.
4. **A decoder, not per-step decisions.** spellF1 matches possessions by temporal IoU, so a one-step
   flicker splits a possession and costs far more than its frames. With v0's heads on DFL's labels,
   the per-step argmax scores 0.456 (`camera`) and 0.408 (`plain`) on the held-out copies, with 678
   and 807 possessions per match against 506 true ones. Decoded, they score 0.565 and 0.546, with
   396 and 414 possessions. The two dead states encode the one rule CHALLENGE.md gives for
   restarts: there is always a clear possessor.
5. **5 Hz is fine enough.** In the IDSSE labels, runs of one team shorter than 0.4 s (outside
   restarts) cover at most 0.16 % of live frames.
6. **No home-side prior.** All five training matches are home games of one club, so a home prior
   would describe that club, not Barcelona. Hence both team views, symmetric transitions and no
   clock: where a half's published start falls against its kick-off differs between sources (the
   challenge's second half opens with a published dead stretch).
7. **The same file, not approximately.** Integer decoding, deterministic LightGBM, and trained
   heads committed and pinned by SHA-256.
8. **A gate against fooling ourselves.** A change after v0 must raise spellF1 by at least 0.02 on
   both held-out matches under both simulators: four cells, each a mean over nine copies, scored
   against labels of the chosen convention. The simulators are our own assumptions, so a gain
   under only one of them, or on one match, counts as noise, and retraining alone moves a cell by
   up to 0.013. Two steps ship short of the gate, by the team's decision, and Section 5.1 gives
   both: the switch to the labels of T = 1, and v3's decoder change, 0.0009 short in one cell.

## 2. Assets: used, tried and dropped

| asset | status | how, or why not |
|---|---|---|
| Tactical ATD (`_atd.xml`, `_atd.txt`) | **used** | The only input of the submission run, through our own parser. |
| Published boundaries | **used** | The halves (DATA.md) and the dead stretch before the second-half kick-off (FAQ.md): `X` outside the halves, `D` forced on the stretch. |
| Homography row printed in DATA.md | **used in training** | Solved as a pan camera with square pixels and no roll: 30.4 m high, 47.3 m behind the near touchline (`pipeline/camera.py`). The degrader uses it for parallax, pixel noise and the field of view. The features use it for the distance along the camera ray: a ball seen at a ground point may be anywhere on its ray up to 3 m high, and the distance is to the nearest point of that stretch. |
| Tactical homography, per frame | **not in the model** | Read only by the label-free checks below. |
| Smart Tagging | **dropped** | Inferred, not truth; `Team` is `N/A` on every throw-in, the restarts where the side matters most; and its intervals are phases, not possessions. None of our labelled open data has Smart Tagging, so how far to trust a tag could only be learned on the challenge match, which we do not tune on. |
| Video, tactical and broadcast | **dropped** | Filling the ball in from video needs a detector trained and checked on footage like this, with labels we did not have. The broadcast is not synchronised to the tactical video and has no usable homography on a third of its frames. |
| Broadcast ATD and homography | **dropped** | A separate timeline with cuts and replays; mapping it to tactical frames was left for later. |
| kloppy | **replaced** for the challenge bundle | Our parser lets the metadata decide which field belongs to which slot (Metrica's game 3 changes layout at every substitution), filters nothing silently and never prints a value from the file. kloppy 3.19.1 cross-checks it on that game and reads the IDSSE positions (`only_alive=False`, so dead frames stay). The IDSSE labels come from our own pass over the raw XML, because kloppy turns a missing BallPossession into "away". |
| IDSSE (DFL) open data | **used** | All training and validation data (Section 3). |
| SkillCorner open data | **adapter tests only** | 20 A-League matches of broadcast tracking at 10 fps. Its possession group is set only while a player holds the ball (null for about half of in-play time), and it has no dead-ball flag like the challenge's `D`. |
| Metrica sample data | **parser tests only** | Game 3 (FIFA/EPTS) tests our parser and is the template for a stand-in bundle. Not trained on: it has no per-frame possession label, and its ball vanishes in stoppages, which would teach that no ball means dead ball. |
| Format stand-in | **code tests only** | A synthetic bundle in the challenge's format, built from IDSSE match J03WMX, shows that the code runs end to end. It never chooses a model. |
| Public five minutes | **a pipeline check, never trained or tuned on** | The frozen model's file was scored once on the window, to check alignment and format: the sides and the timing lined up, and nothing changed after it. Not used to choose the convention (Section 4.5). |

**Label-free checks on the challenge bundle.** Four scripts measure the bundle without reading any
label and print only PASS or FAIL lines. `scripts/match_test8.py` checks integrity: our reader,
pushed through the starter baseline's nearest-player rule, must reproduce the baseline file shipped
in the challenge repository on every frame, and the kick-offs, attack direction and homography must
agree with the published boundaries. `match_test7.py` measures how the players move during ball
gaps of each length, against degraded open copies whose truth is known; `match_test9.py` the
camera's aim against the visible ball; `match_parallax.py` the signature of misplaced high balls.
They were written to settle two degrader settings that the documents do not give: how much of the
missing-ball time is live play, and whether balls in the air are misplaced. They do not change the
frozen model: the training copies keep drawing both settings at random (Section 3.1), and the
checks only describe the match's tracking.
Test 8 passed everything that touches our reader: it reproduces the baseline file on every frame of
both halves, and the attack direction, the homography and the first-half kick-off agree with the
published boundaries (the metadata carries no half boundaries, so the published ones are used). At
the second-half kick-off the teams stand in their own halves on the published frame, which the
challenge's own anchor puts on the kick to within a frame, but the tracked ball is not on the centre
spot; we read that as the tracker misplacing the ball at that restart, and changed nothing.
Tests 7, 9 and parallax ran on the bundle as well. With the model frozen their reports were not
used: the two settings they were written to settle stay drawn at random in training.

## 3. Training data

All labelled data is IDSSE: Bassek et al. 2025, *An integrated dataset of synchronized
spatiotemporal and event data in elite soccer*, Scientific Data 12, 195,
https://doi.org/10.1038/s41597-025-04505-y, CC BY 4.0, data owner Deutsche Fussball Liga. We used
the Hugging Face mirror `pysport/idsse-data`; `docs/manifests/` pins every file by size and digest.
None of it comes from the provided game, and we labelled nothing ourselves, on any game.

| match | competition, 2022/23 | fixture | role | frames in the halves (25 Hz) |
|---|---|---|---|---|
| J03WOH | 2. Bundesliga | Fortuna Düsseldorf v SSV Jahn Regensburg | training | 137,214 |
| J03WOY | 2. Bundesliga | Fortuna Düsseldorf v F.C. Hansa Rostock | training | 142,536 |
| J03WPY | 2. Bundesliga | Fortuna Düsseldorf v 1. FC Nürnberg | training | 146,211 |
| J03WQQ | 2. Bundesliga | Fortuna Düsseldorf v FC St. Pauli | training | 142,345 |
| J03WR9 | 2. Bundesliga | Fortuna Düsseldorf v 1. FC Kaiserslautern | training | 146,810 |
| J03WMX | Bundesliga | 1. FC Köln v FC Bayern München | held out | 145,967 |
| J03WN1 | Bundesliga | VfL Bochum 1848 v Bayer 04 Leverkusen | held out | 141,561 |

That is 715,116 training frames (7.9 hours) and 287,528 held-out frames (3.2 hours), the latter
from the other division and other teams. In DFL's labels `D` covers 41–52 % of frames, a match has
370–511 possessions, and the median possession lasts 3.0–4.7 s.

- **Who labelled it.** Every frame of the official DFL position feed carries `BallStatus`
  (0 inactive, 1 active) and `BallPossession` (1 home, 2 away). Sportec Solutions collects the feed
  with the TRACAB Gen5 multi-camera system, and the IDSSE paper defines possession by the player in
  control of the ball. The sources we read do not say how much of either flag is set automatically
  and how much by operators.
- **Our labels from it.** `D` where `BallStatus` is 0, otherwise `H` for `BallPossession` 1 and `A`
  for 2; any other value stops the export (`scripts/export_idsse.py`). Every live frame has a
  possession value. A fixed rule then relabels the same frames for a convention T (Section 4.5).
  The shipped model is trained and gated on T = 1, which changes 0.5–1.4 % of live frames; the
  convention test also uses T = 0.4 and 2 s.
- **Not used.** The IDSSE event data (entered by human operators) and player identities. Ball height
  drives only the degrader's camera and never reaches a feature.
- **Copies.** Each training match is degraded three times (seeds 1–3) under the `camera` simulator,
  with the live share of the missing time drawn per copy: 15 copies, 429,081 steps. Each held-out
  match is degraded under `camera`, `plain` and `camera-parallax`, at live shares 0.1, 0.3 and 0.5,
  with seeds 1–3: 54 copies, of which the 36 under `camera` and `plain` form the gate. Boosting
  rounds and α come from the training folds; the held-out matches set no number and only gate
  whole models (Section 1.2).

### 3.1 Making open tracking look like the challenge's

`pipeline/degrade.py` keeps the labels and spoils what a model can learn from:

- **Ball.** It goes missing in contiguous, heavy-tailed blocks, with separate hazards for live and
  dead play and a visibility that drifts between five-minute blocks, until the printed 41 % of
  graded frames is reached. How much of the missing time is live play is not published, so it is
  drawn per training copy (5–55 %), and absence cannot be learned as a stoppage.
- **Stoppages.** DFL's ball stands still during a stoppage and jumps to the restart spot. In the
  copies it coasts, is carried near a player and arrives at the spot, so neither a still ball nor a
  jump marks dead play.
- **Camera.** The camera solved from the homography row pans after the ball with a lag, whatever
  leaves the image is lost, and positions get pixel noise, which is larger in depth and on the far
  side. Whether the real tracker corrects for height is not known, so each copy has an even chance
  that a ball at height z lands on the ground where its camera ray meets it, D · z / (H − z)
  further away, for a ball D metres from the foot of a camera H metres high.
- **Identities.** People become anonymous tracklets in 46 slots, with breaks, swaps at occlusions,
  tracklets on the wrong team, keeper time on the other team, duplicates and false tracks, at about
  20.5 tracked objects per frame.

Only figures printed in the challenge's documents are targets (41 %, 20.5, 46 slots, the homography
row); everything else is an assumption, kept wide and drawn per copy. The second simulator,
`plain`, has no camera at all: noise in metres, no field of view, no parallax, and no ball lost in
the air or in crowds. `camera-parallax` always misplaces high balls and is reported beside the gate.

## 4. Possession rules

The labels come from DFL's feed, so most of where the line falls is DFL's: the model learns when
play is dead from DFL's ball status, and who owns a live ball from its possession flag. On top of
that we fixed the following.

**4.1 Three states.** Every graded frame gets `H`, `A` or `D`; `X` appears only outside the halves.
spellF1 replaces an `X` with the previous label, so inside a possession it repeats what the decoder
says anyway, and at a change it drags the old label across; macroF1 counts it as a miss.

**4.2 Dead ball.** `D` where the dead head and the decoder put it, plus the published stretch
before the second-half kick-off. Who owns a dead ball is never reported, but the decoder keeps it
internally as D-H or D-A: the team that will have the ball when play resumes. Shifting v0's
dead-ball evidence by ±1 nat per step (about ±0.36 nat of the dead head's log-odds) costs at most
0.002 spellF1 on the training folds and 0.004 on the held-out gate. The best shift gains 0.002 on
the folds, under the +0.005 we had fixed in advance as the bar for a candidate, so there is no
calibration parameter (`analysis/test2_dead_shift.py`). Too much `D` hurts more than too little:
two nats towards `D` cost 0.010 on the gate, two nats away from it gain 0.003.

**4.3 Restarts.** "Whenever play restarts after a stoppage, there is a clear possessor"
(CHALLENGE.md). A dead state can only end in live play for the team it names, so the restart head,
which reads where the players stand and move while the ball is dead, decides who comes out of a
stoppage. It may change its mind while the ball is dead. In v2 a dead spell kept the restarter it
began with, so the decoder could change sides only through a single live step: a possession of
0.2 s that the metric counts and that never matches (152 of them out of fold on the training
copies, none in the truth). v3 lets D-H and D-A swap at the cost of staying dead, which removes
them all and fits no new number. Under the relabelled conventions, the first run after a stoppage
or at the start of a half belongs to the restarting team whatever its length.

**4.4 Loose balls and challenges.** There is no "contested" state, just as in the truth: while the
ball is loose or contested, possession stays with the last team (CHALLENGE.md). In the decoder, the
current state holds until the team evidence outweighs the cost of a switch, counted from the
training labels, for long enough. A touch without control, a deflected pass or an aerial duel won
without control should therefore not flip the label. How short a run may be before it counts is the
convention T.

**4.5 The convention T.** In live play, a run of one team shorter than T seconds does not change
possession; it takes the previous team's label. T = 0 is DFL's label as it is. The challenge's text
points away from T = 0 (the three cases that do not count as a change), but its truth mechanically
takes the owner that Metrica's corrected tracking names, and how that owner treats a short touch is
not published. We trained a v0-style
model for each T in 0, 0.4, 1 and 2 s on its own labels (α re-chosen per T on the folds: 2.8, 2.8,
1.4 and 1.0) and scored each against every convention (`analysis/test1_convention.py`). spellF1 on
the training folds, out of fold:

| model \ truth | T = 0 | T = 0.4 | T = 1 | T = 2 | worst loss against the matched model |
|---|---|---|---|---|---|
| T = 0 (v0) | **0.614** | 0.628 | 0.647 | 0.647 | 0.053 |
| T = 0.4 | 0.617 | **0.631** | 0.650 | 0.649 | 0.050 |
| T = 1 | 0.606 | 0.622 | **0.655** | 0.693 | **0.009** |
| T = 2 | 0.563 | 0.581 | 0.620 | **0.699** | 0.051 |

A wrong T costs up to 0.053 on the folds and 0.075 on the held-out matches. T = 1 is the minimax:
within 0.009 of the matched model under every tested truth on the folds (0.016 on the held-out
matches). The conventions also count very differently: the held-out matches have on average 506,
470, 396 and 307 true possessions for T = 0, 0.4, 1 and 2. Nothing in IDSSE can tell which
convention the challenge's truth follows, so T is a decision, not a measurement.

We ship **T = 1**, the choice that loses least whichever of the four conventions holds. It rests
on open data alone: no model was scored on the public five minutes to choose it.

**4.6 Boundaries** fall between 0.2 s steps, and each frame gets its step's label.

## 5. Results on open data, and limitations

### 5.1 Numbers

The shipped model, v3, on the held-out gate copies (18 per simulator; 396 true possessions per
match under T = 1) and out of fold on the training matches (15 copies; 353 true possessions per
copy), scored against labels of T = 1 with `analysis/spellmetric.py`, our clean-room
reimplementation of the metrics in EVALUATION.md:

| | camera | plain | gate (mean) | training folds |
|---|---|---|---|---|
| spellF1 | 0.652 | 0.640 | **0.646** | 0.667 |
| macroF1 | 0.895 | 0.886 | 0.890 | – |
| possessions predicted per match | 316 | 323 | 319 | 290 |

How v3 came about, on the same 36 gate copies (v1, the change heads in Section 5.3, was dropped):

| model | what changed | gate, truth T = 0 | gate, truth T = 1 | possessions predicted per match |
|---|---|---|---|---|
| v0 | trained on DFL's labels (T = 0) | 0.556 | 0.621 | 405 |
| v2 | v0's recipe on the labels of T = 1 | 0.540 | 0.625 | 345 |
| v3 | v2's heads; the restarter may change during a stoppage | 0.555 | **0.646** | 319 |

On T = 1 labels v2 leads v0 by only 0.005 (−0.004 to +0.013 per cell). It ships anyway because the
convention comes before the gate: once T = 1 is chosen, the model trained for it becomes the
reference. v3 leads v2 by 0.021 and is better on all 36 copies, but one cell (`camera`, J03WN1)
gains +0.0191, 0.0009 short of the bar, while the other three gain +0.021 to +0.022. It ships too,
by the team's decision: its one change fits nothing new and removes a failure the truth never shows
(Section 4.3). Against DFL's labels v3 is level with v0, the model made for them (0.555 against
0.556), so shipping it at T = 1 costs next to nothing if the truth follows T = 0.

### 5.2 Where it fails

- **Short possessions after a turnover.** v3 finds about 320 possessions per held-out match where
  the labels of T = 1 have 396: short possessions merge into their neighbours. Out of fold it finds
  (at IoU 0.5) 17 % of the true possessions of 1–2 s that follow a turnover and 40 % of those of
  2–4 s, and such runs are a third of all possessions. In the ones it misses, the ball is seen on
  two thirds of the steps, yet the team head favours the right team on only 43 % (1–2 s) and 55 %
  (2–4 s) of them. No boundary is finer than 0.2 s.
- **Live play inside ball gaps.** The more of the missing time is live play, the worse it gets:
  v3 scores 0.682, 0.648 and 0.625 at live shares of 10, 30 and 50 % under `camera`, and 0.675,
  0.637 and 0.609 under `plain`. The real share is not published.
- **Balls in the air.** If the tracker puts high balls on the ground, they land metres away: a
  median 3.1–5.2 m for balls 1–2 m high and 18–28 m for balls 5–10 m high, from the near to the far
  third. The 40 m/s gate then deletes 2.5 % of live ball observations instead of 0.05 %, and for a
  ball 1–2.5 m up with a player in reach the apparent nearest team is the wrong one 31 % of the
  time. The distance along the camera ray brings that to 11 % but is worse for balls on the ground
  (6 % against 1 %), so it sits beside the plain distance, not in its place
  (`analysis/test3_virtual_camera.py`; an upper bound, since it assumes every high ball is detected
  and none is corrected). For v0 on DFL's labels, held-out copies that always misplace high balls
  score 0.544, against 0.565 under `camera`, where 3 of the 18 held-out copies drew the
  misplacement.
- **Keepers.** Cleaning returns only 79 % (`camera`) and 91 % (`plain`) of keeper time to the right
  team, pooled over 21 copies, and nothing at all on the worst copy.
- **Domain.** The training data is five home matches of one second-division club in one season,
  tracked by one system and labelled by DFL; the challenge's truth comes from Metrica's corrected
  tracking. Wherever the two disagree on what "dead" and "owning" mean, we inherit DFL's line.
- **The convention** (Section 4.5): if the truth follows another of the tested conventions, the
  T = 1 model loses up to 0.009 on the folds and 0.016 on the held-out matches against the model
  made for it. A convention outside the tested ones could cost more.
- **No way to see our own failures on the match.** There are no labels to see them with, and five
  public minutes are far too few to judge a model on, nor are they meant for it.

### 5.3 What we would fix with more time

1. **New evidence for short possessions.** Fed the true labels in place of one head, v3's decoder
   scores 0.942 out of fold with a perfect team head and 0.717 with a perfect dead-ball head,
   against 0.667: what is missing is evidence in the team head, not decoding. Decoder settings
   (cheaper switches, a rescaled team head, the T = 1 rule applied to the output) move the
   out-of-fold score by −0.016 to +0.004. The best retrained head, a second LightGBM stacked on the
   team head's output over ±3 s, gains 0.010 on the gate against v3 (+0.008 to +0.013 per cell),
   half the bar, and was dropped, as were three change heads (switch, stop, resume) driving v0's
   transitions, at +0.009. Next we would detect touches at 25 Hz and snap each change to one.
2. **Smart Tagging and the camera's aim as evidence**, once their reliability can be measured
   without the match's labels, and only through the gate.
3. **Video**: a ball detector for the gaps and, once the broadcast is aligned to the tactical
   timeline, its shots without the pitch (replays, close-ups) as evidence of stoppages.
4. **More labelled data** from other providers and leagues, ideally labelled the way the
   challenge's truth is.
5. **Keeper repair** that holds up on degraded tracking, and cleaning thresholds chosen through the
   gate rather than fixed in advance.

## 6. Reproduction

`README.md` has the installation (Python 3.14, pinned requirements, and LightGBM's OpenMP on macOS)
and the command:

```bash
.venv/bin/python -I scripts/predict_match.py <tactical_dir> <team_name>.csv
```

`<tactical_dir>` is the data bundle's `tactical/` folder, of which only `BAR-ATH_tactical_atd.xml`
and `BAR-ATH_tactical_atd.txt` are read. The command runs v3, the default, prints `PASS`,
`rows 199319` and `spells <n>`, and writes the CSV with its SHA-256 beside it. It is deterministic
(the same bundle gives the same bytes) and takes about ten seconds. Run twice on the challenge
bundle, it wrote the same file byte for byte.

The model files are committed in `models/`: v3's decoder bundle and the three LightGBM text files
it pins by SHA-256. They are committed rather than rebuilt because LightGBM trained on another
machine can differ in its last digits. Rebuilding them from open data is optional; `README.md`
lists the commands: fetch IDSSE, export the labels, degrade the matches and compute their
features, train v0, train the heads for each convention (`analysis/test1_convention.py`), then
bundle v2 and v3.

On the machine that trained them, those commands in a fresh clone with a fresh virtual environment
rebuild every file in `models/` byte for byte, in about 46 minutes after the download. On other
machines the committed files carry over: in the `python:3.14.7-slim` container, on arm64 and on
x86-64 under Rosetta, `tests/check_predict.py` passes as on the Mac. From the features cached on
the Mac, the committed models give back all 54 held-out scores of v0 and of v2 and the 36 gate
scores of v3 exactly, and a held-out copy degraded from scratch in the container gets the same
label on every frame. The format stand-in gives the same file, byte for byte, on the Mac, on both
Linux architectures and from the fresh clone, with the committed model files and with the rebuilt
ones. The other self-checks
(`tests/check_adapters.py`, `check_degrade.py`, `check_features.py`, `check_twin.py` and
`check_match_scripts.py`) also run on open data and the stand-in only.
