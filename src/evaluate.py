"""Compare all policies on fresh simulated students and write results + dashboard data.

    python -m src.evaluate
"""
import json
from pathlib import Path

import numpy as np

from .policies import ALL, LearningProgressBandit, LinearCurriculum
from .simulate import run_many, run_student
from .student import LEVELS, N_LEVELS, N_TOPICS, TOPICS, World

ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = range(0, 2000)                 # disjoint from the tuning seeds (100000+)
SWEEP_SEEDS = range(5000, 5600)
HORIZON = 180

SWEEP = {
    "Baseline world": {},
    "Learning less forgiving of poor pitch": {"width": 0.10},
    "Learning more forgiving of poor pitch": {"width": 0.22},
    "Slow learners": {"lr_mean": 0.03},
    "Fast learners": {"lr_mean": 0.065},
    "Students differ more": {"lr_sigma": 0.7},
    "More guessing (4-option questions)": {"guess": 0.25},
    "Faster forgetting": {"forget": 0.003},
    "Softer prerequisite gate": {"prereq_gate": 3.0},
}


def tuned_params():
    path = ROOT / "artifacts" / "tuned_params.json"
    return {k: v["params"] for k, v in json.loads(path.read_text()).items()} if path.exists() else {}


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    params, world = tuned_params(), World()
    res = {P.name: run_many(P, world, TEST_SEEDS, HORIZON, params.get(P.__name__)) for P in ALL}
    base = res[LinearCurriculum.name]
    start = base["start"]

    # learning speed is hidden from every policy; recover it here only to slice results
    from .student import Student
    lr = np.array([Student(world, np.random.default_rng([s, 1])).lr for s in TEST_SEEDS])
    speed = np.digitize(lr, np.quantile(lr, [1 / 3, 2 / 3]))
    seg_names = ["Slowest third of learners", "Middle third", "Fastest third"]

    table = []
    for name, r in res.items():
        d = r["final"] - base["final"]
        table.append({"policy": name, "final_mastery": float(r["final"].mean()), "final_ci": boot_ci(r["final"]),
                      "gain": float((r["final"] - r["start"]).mean()), "post_test": float(r["post_test"].mean()),
                      "post_test_ci": boot_ci(r["post_test"]),
                      "vs_linear": float(d.mean()), "vs_linear_ci": boot_ci(d),
                      "share_of_students_better_than_linear": float((d > 0).mean()),
                      "post_test_vs_linear": float((r["post_test"] - base["post_test"]).mean()),
                      "post_test_vs_linear_ci": boot_ci(r["post_test"] - base["post_test"]),
                      "topics_above_half": float((r["by_topic"] >= 0.5).sum(1).mean()),
                      "by_topic": r["by_topic"].mean(0).round(3).tolist(),
                      "by_speed": [float(r["final"][speed == i].mean()) for i in range(3)],
                      "curve": r["curve"].mean(0).round(4).tolist()})
    print(f"{len(TEST_SEEDS)} students x {HORIZON} questions, start mastery {start.mean():.3f}")
    for row in sorted(table, key=lambda r: -r["final_mastery"]):
        print(f"  {row['policy']:40s} final {row['final_mastery']:.3f}  post-test {row['post_test']:.3f}  "
              f"vs linear {row['vs_linear']:+.3f} [{row['vs_linear_ci'][0]:+.3f}, {row['vs_linear_ci'][1]:+.3f}]  "
              f"by speed {np.round(row['by_speed'], 3)}")

    sweep = []
    for label, change in SWEEP.items():
        w = world.but(**change)
        lin = run_many(LinearCurriculum, w, SWEEP_SEEDS, HORIZON)["final"]
        ban = run_many(LearningProgressBandit, w, SWEEP_SEEDS, HORIZON, params.get("LearningProgressBandit"))["final"]
        d = ban - lin
        sweep.append({"world": label, "change": change, "linear": float(lin.mean()), "bandit": float(ban.mean()),
                      "diff": float(d.mean()), "ci": boot_ci(d)})
        print(f"  sweep {label:40s} linear {lin.mean():.3f} bandit {ban.mean():.3f} diff {d.mean():+.3f}")
    for h in (90, 120, 240, 360):
        lin = run_many(LinearCurriculum, world, SWEEP_SEEDS, h)["final"]
        ban = run_many(LearningProgressBandit, world, SWEEP_SEEDS, h, params.get("LearningProgressBandit"))["final"]
        d = ban - lin
        sweep.append({"world": f"{h} questions instead of {HORIZON}", "change": {"horizon": h}, "linear": float(lin.mean()),
                      "bandit": float(ban.mean()), "diff": float(d.mean()), "ci": boot_ci(d)})
        print(f"  sweep {h} questions: linear {lin.mean():.3f} bandit {ban.mean():.3f} diff {d.mean():+.3f}")

    # one example student under both policies: a slow learner, where pacing matters most
    example_seed = int(np.array(list(TEST_SEEDS))[np.argsort(lr)[len(lr) // 6]])
    traces = {}
    for P in (LinearCurriculum, LearningProgressBandit):
        t = run_student(P, world, example_seed, HORIZON, params.get(P.__name__), trace=True)
        traces[P.name] = {"final": t["final"], "steps": [[r[1], r[2], r[3], round(r[4], 3)] for r in t["trace"]],
                          "mastery": [list(map(float, r[5:])) for r in t["trace"]]}

    # where each policy spends its questions: share per topic in blocks of 20 questions
    blocks = HORIZON // 20
    spend = {}
    for P in (LinearCurriculum, LearningProgressBandit):
        counts = np.zeros((N_TOPICS, blocks))
        for s in range(300):
            for r in run_student(P, world, s, HORIZON, params.get(P.__name__), trace=True)["trace"]:
                counts[r[1], r[0] // 20] += 1
        spend[P.name] = (counts / counts.sum(0)).round(3).tolist()

    out = {"students": len(TEST_SEEDS), "horizon": HORIZON, "start_mastery": float(start.mean()),
           "topics": TOPICS, "levels": LEVELS.tolist(), "segments": seg_names, "table": table, "sweep": sweep,
           "tuned": json.loads((ROOT / "artifacts" / "tuned_params.json").read_text()),
           "example": {"seed": example_seed, "traces": traces}, "spend": spend,
           "world": World().__dict__}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    slim = {k: v for k, v in out.items() if k not in ("example", "spend")}
    slim["table"] = [{k: v for k, v in r.items() if k != "curve"} for r in table]
    (ROOT / "artifacts" / "results.json").write_text(json.dumps(slim, indent=1))
    (ROOT / "docs" / "data.js").write_text("window.TUTOR_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
