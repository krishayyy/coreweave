# Run records

`experiment_headline.json` backs every figure quoted in the README: 30 type A,
24 type B and 12 type C scenarios, three independent detection-roll repeats,
all arms on identical scenarios with identical rolls.

    experiment_headline.json        the headline suite
    experiment_final.json       the deterministic arms it was assembled from
    experiment_cond.json        the System One arm it was assembled from
    experiment_llm_current.json the language model arm, same configuration
    time_curve.json             find rate against search budget
    learning_curve.json         case memory against a no-memory control

Intermediate runs from earlier configurations have been deleted rather than
kept, because a directory of near-identical files invites quoting whichever one
is most flattering. Everything here is reproducible:

    python scripts/experiment.py --n-a 30 --n-b 24 --n-c 12 --repeats 3
    python scripts/time_curve.py
    python scripts/learning_curve.py --n 16
