"""Possession pipeline. Every stage after the adapters reads pipeline.schema only.

    from pipeline.clean import clean
    from pipeline.epts import load_competition, load_epts
    from pipeline.idsse import load_idsse
    from pipeline.skillcorner import load_skillcorner

    match = load_competition("path/to/data/tactical")     # explicit path, no environment variable
    match = load_idsse("external/processed/idsse/J03WMX.npz")
    match = load_skillcorner("external/skillcorner/data/matches/1886347")
    cleaned = clean(match)

    from pipeline.degrade import degrade                    # training only, refuses unlabelled data
    copy = clean(degrade(load_idsse("external/processed/idsse/J03WMX.npz"), seed=1))

    from pipeline.features import features, step_index      # 5 Hz, from the clean schema only
    from pipeline.model import HEADS, predict                # LightGBM heads: dead, team, restart
    from pipeline.decode import decode                       # 4-state integer Viterbi (H, A, D-H, D-A)
    steps = step_index(cleaned)
    names, table = features(cleaned, steps)
    _, swapped = features(cleaned, steps, swap=True)

    from pipeline.predict import frame_labels, load_model   # a saved model onto any Match, end to end
    from pipeline import submission                         # write the frame,label file, read it back, check it
    labels = frame_labels(match, load_model("models", "v3"))           # what scripts/predict_match.py runs

tests/check_adapters.py checks the adapters and cleaning on the twin and the open datasets,
tests/check_degrade.py the camera and the degrader on IDSSE, tests/check_features.py the features, the
heads' targets and the decoder, tests/check_predict.py the path from saved files to the submission
file. analysis/model_v0.py trains and scores model v0 on IDSSE; analysis/model_v2.py bundles v2, the
same recipe on the labels of convention T = 1 (analysis/test1_convention.py), and analysis/model_v3.py
bundles v3, v2's heads with a restarter that may change during a stoppage.
"""
