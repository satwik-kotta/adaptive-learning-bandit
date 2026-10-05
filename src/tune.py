"""Tune each policy's settings on a development population of simulated students.

    python -m src.tune

The evaluation in src.evaluate uses different students, so tuned settings are
not scored on the students they were tuned on.
"""
import json
from pathlib import Path

import numpy as np

from .policies import LearningProgressBandit, Staircase, ThompsonTarget
from .simulate import run_many
from .student import World

DEV_SEEDS = range(100_000, 100_250)
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "tuned_params.json"

SPACES = {
    Staircase: {"budget_factor": [0.8, 1.0, 1.2, 1.5, 2.0]},
    ThompsonTarget: {"target": [0.5, 0.6, 0.65, 0.7, 0.8], "discount": [0.9, 0.95, 0.97, 0.99],
                     "open_at": [0.5, 0.6, 0.7, 0.8], "close_at": [0.75, 0.8, 0.85, 0.9]},
    LearningProgressBandit: {"window": [4, 6, 8], "explore": [0.05, 0.1, 0.15, 0.25, 0.4],
                             "smooth": [0.3, 0.6, 0.8], "open_at": [0.5, 0.6, 0.7, 0.8],
                             "close_at": [0.75, 0.8, 0.85, 0.9], "temp": [0.5, 1.0, 2.0]},
}


def main(n_configs=40):
    rng = np.random.default_rng(0)
    world, tuned = World(), {}
    for cls, space in SPACES.items():
        keys = list(space)
        grid = np.prod([len(space[k]) for k in keys])
        if grid <= n_configs:
            import itertools
            configs = [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])]
        else:
            configs = [{k: rng.choice(space[k]).item() for k in keys} for _ in range(n_configs)]
        scored = [(run_many(cls, world, DEV_SEEDS, params=c)["final"].mean(), c) for c in configs]
        best = max(scored, key=lambda x: x[0])
        default = run_many(cls, world, DEV_SEEDS)["final"].mean()
        tuned[cls.__name__] = {"params": best[1], "dev_final_mastery": float(best[0]),
                               "dev_final_mastery_default": float(default), "configs_tried": len(configs)}
        print(f"{cls.name}: default {default:.3f} -> tuned {best[0]:.3f} with {best[1]}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(tuned, indent=1))


if __name__ == "__main__":
    main()
