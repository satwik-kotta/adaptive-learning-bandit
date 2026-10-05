"""Run policies against simulated students."""
import numpy as np

from .student import N_LEVELS, N_TOPICS, Student, World


def run_student(policy_cls, world, seed, horizon=180, params=None, trace=False):
    """One student, one policy. The student is identical across policies for a given seed."""
    student = Student(world, np.random.default_rng([seed, 1]))
    policy = policy_cls(horizon, np.random.default_rng([seed, 2]), **(params or {}))
    start = student.mastery.mean()
    curve = np.empty(horizon + 1)
    curve[0] = start
    rows = []
    for t in range(horizon):
        topic, level = policy.choose(student)
        m_before = student.mastery[topic]
        correct = student.answer(topic, level)
        policy.update(topic, level, correct)
        curve[t + 1] = student.mastery.mean()
        if trace:
            rows.append((t, topic, level, int(correct), float(m_before), *student.mastery.round(4)))
    test = student.post_test(np.random.default_rng([seed, 3]))
    out = {"start": float(start), "final": float(student.mastery.mean()), "post_test": test,
           "final_by_topic": student.mastery.copy(), "curve": curve}
    if trace:
        out["trace"] = rows
    return out


def run_many(policy_cls, world, seeds, horizon=180, params=None):
    res = [run_student(policy_cls, world, s, horizon, params) for s in seeds]
    return {"start": np.array([r["start"] for r in res]), "final": np.array([r["final"] for r in res]),
            "post_test": np.array([r["post_test"] for r in res]),
            "by_topic": np.array([r["final_by_topic"] for r in res]),
            "curve": np.array([r["curve"] for r in res])}
