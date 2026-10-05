import numpy as np

from src.policies import ALL, LearningProgressBandit, LinearCurriculum, Oracle, Staircase
from src.simulate import run_many, run_student
from src.student import LEVELS, N_LEVELS, N_TOPICS, Student, World


def test_harder_questions_are_answered_correctly_less_often():
    s = Student(World(), np.random.default_rng(0))
    s.mastery[:] = 0.5
    p = [s.p_correct(0, d) for d in range(N_LEVELS)]
    assert all(a > b for a, b in zip(p, p[1:]))
    assert p[-1] >= World().guess and p[0] <= 1 - World().slip


def test_learning_peaks_just_above_current_mastery():
    s = Student(World(), np.random.default_rng(0))
    s.mastery[:] = 0.4
    gains = [s.expected_gain(0, d) for d in range(N_LEVELS)]
    assert int(np.argmax(gains)) == int(np.argmin(np.abs(LEVELS - 0.5)))


def test_prerequisite_blocks_learning():
    s = Student(World(), np.random.default_rng(0))
    s.mastery[:] = 0.1
    blocked = s.expected_gain(1, 0)
    s.mastery[0] = 0.9
    assert s.expected_gain(1, 0) > 10 * blocked


def test_same_seed_gives_the_same_student_to_every_policy():
    starts = {P.name: run_student(P, World(), 7, horizon=5)["start"] for P in ALL}
    assert len(set(starts.values())) == 1


def test_runs_are_reproducible():
    a = run_student(LearningProgressBandit, World(), 3)["final"]
    b = run_student(LearningProgressBandit, World(), 3)["final"]
    assert a == b


def test_policies_only_return_valid_arms_and_bandit_starts_easy():
    for P in ALL:
        t = run_student(P, World(), 1, horizon=60, trace=True)["trace"]
        assert all(0 <= r[1] < N_TOPICS and 0 <= r[2] < N_LEVELS for r in t)
    first = run_student(LearningProgressBandit, World(), 1, horizon=3, trace=True)["trace"][0]
    assert (first[1], first[2]) == (0, 0)


def test_linear_curriculum_covers_every_topic_equally():
    t = run_student(LinearCurriculum, World(), 1, horizon=180, trace=True)["trace"]
    assert np.bincount([r[1] for r in t], minlength=N_TOPICS).tolist() == [30] * N_TOPICS


def test_oracle_is_an_upper_bound_on_average():
    seeds = range(60)
    o = run_many(Oracle, World(), seeds)["final"].mean()
    for P in (LinearCurriculum, Staircase, LearningProgressBandit):
        assert o > run_many(P, World(), seeds)["final"].mean()


def test_post_test_does_not_change_mastery():
    s = Student(World(), np.random.default_rng(0))
    before = s.mastery.copy()
    s.post_test(np.random.default_rng(1))
    assert np.array_equal(before, s.mastery)
