"""Question-selection policies. Each sees only its own history of (topic, level, correct)."""
import numpy as np

from .student import N_LEVELS, N_TOPICS


class Policy:
    name = "policy"

    def __init__(self, horizon, rng):
        self.T, self.rng, self.t = horizon, rng, 0

    def choose(self, student=None):
        raise NotImplementedError

    def update(self, topic, level, correct):
        self.t += 1


class LinearCurriculum(Policy):
    """The textbook: equal time per topic, in order, difficulty stepping up on a fixed schedule."""
    name = "Linear curriculum"

    def choose(self, student=None):
        per_topic = self.T / N_TOPICS
        topic = min(int(self.t // per_topic), N_TOPICS - 1)
        level = min(int((self.t - topic * per_topic) // (per_topic / N_LEVELS)), N_LEVELS - 1)
        return topic, level


class RandomPolicy(Policy):
    name = "Random questions"

    def choose(self, student=None):
        return int(self.rng.integers(N_TOPICS)), int(self.rng.integers(N_LEVELS))


class Staircase(Policy):
    """Rule-based adaptive practice: two right in a row -> harder, one wrong -> easier.
    Moves to the next topic after three right in a row at the top level, or when the
    topic's time budget runs out. A strong non-learning baseline."""
    name = "Staircase rule"

    def __init__(self, horizon, rng, budget_factor=1.5):
        super().__init__(horizon, rng)
        self.topic, self.level, self.streak, self.spent = 0, 0, 0, 0
        self.budget = budget_factor * horizon / N_TOPICS
        self.final_level = np.zeros(N_TOPICS, dtype=int)
        self.done = False

    def choose(self, student=None):
        return self.topic, self.level

    def update(self, topic, level, correct):
        super().update(topic, level, correct)
        self.spent += 1
        self.streak = self.streak + 1 if correct else 0
        if not correct:
            self.level = max(0, self.level - 1)
        elif self.streak >= 2 and self.level < N_LEVELS - 1:
            self.level, self.streak = self.level + 1, 0
        self.final_level[self.topic] = self.level
        mastered = self.level == N_LEVELS - 1 and self.streak >= 3
        if mastered or self.spent >= self.budget:
            if not self.done and self.topic < N_TOPICS - 1:
                self.topic += 1
                self.level = 0
            else:                                   # everything visited: go back to the weakest topic
                self.done = True
                self.topic = int(np.argmin(self.final_level))
                self.level = int(self.final_level[self.topic])
            self.streak, self.spent = 0, 0


class LearningProgressBandit(Policy):
    """Multi-armed bandit over (topic, difficulty) arms, rewarded by learning progress.

    The tutor cannot see mastery, so it rewards an arm when the student's recent
    success rate on that arm is *changing*: an arm the student always gets right
    (mastered) or always gets wrong (too hard) earns nothing and fades out.
    Arms are sampled in proportion to their smoothed reward, with a little
    uniform exploration (EXP3-style). Only arms inside the current "zone of
    proximal development" are eligible: a harder level or the next topic opens
    when the student is succeeding on the one before it. After Clement et al.
    (2015), "Multi-Armed Bandits for Intelligent Tutoring Systems".
    """
    name = "Learning-progress bandit"

    def __init__(self, horizon, rng, window=6, explore=0.15, smooth=0.6, open_at=0.7, close_at=0.9, temp=1.0):
        super().__init__(horizon, rng)
        self.window, self.explore, self.smooth = window, explore, smooth
        self.open_at, self.close_at, self.temp = open_at, close_at, temp
        self.hist = [[[] for _ in range(N_LEVELS)] for _ in range(N_TOPICS)]
        self.value = np.full((N_TOPICS, N_LEVELS), 0.5)          # optimistic start
        self.active = np.zeros((N_TOPICS, N_LEVELS), dtype=bool)
        self.closed = np.zeros((N_TOPICS, N_LEVELS), dtype=bool)
        self.active[0, 0] = True
        self.log_active = []

    def _rate(self, k, d, n=None):
        h = self.hist[k][d][-(n or self.window):]
        return np.mean(h) if h else 0.0

    def choose(self, student=None):
        mask = self.active & ~self.closed
        if not mask.any():                                       # everything closed: reopen for review
            self.closed[:] = False
            mask = self.active
        w = np.where(mask, np.maximum(self.value, 1e-3) ** (1 / self.temp), 0.0)
        p = (1 - self.explore) * w / w.sum() + self.explore * mask / mask.sum()
        i = self.rng.choice(N_TOPICS * N_LEVELS, p=p.ravel())
        return int(i // N_LEVELS), int(i % N_LEVELS)

    def update(self, topic, level, correct):
        super().update(topic, level, correct)
        h = self.hist[topic][level]
        h.append(float(correct))
        recent = h[-self.window:]
        if len(recent) >= 4:
            half = len(recent) // 2
            progress = abs(np.mean(recent[half:]) - np.mean(recent[:half]))
            self.value[topic, level] = self.smooth * self.value[topic, level] + (1 - self.smooth) * progress
        n = len(recent)
        if n >= 3 and np.mean(recent) >= self.open_at:           # succeeding here: open what comes next
            if level + 1 < N_LEVELS:
                self.active[topic, level + 1] = True
            if level >= 1 and topic + 1 < N_TOPICS:
                self.active[topic + 1, 0] = True
        if n >= min(self.window, 4) and np.mean(recent) >= self.close_at:
            self.closed[topic, level] = True                     # mastered: stop spending questions here


class ThompsonTarget(Policy):
    """Discounted Thompson sampling toward a target success rate.

    Keeps a Beta posterior on P(correct) for every arm, discounted so old answers
    fade as the student learns. Each turn it samples from the posteriors and asks
    the open question whose sampled success rate is closest to the target (hard
    enough to teach, easy enough to succeed). Uses the same opening rule as the
    learning-progress bandit."""
    name = "Thompson sampling (target difficulty)"

    def __init__(self, horizon, rng, target=0.65, discount=0.97, open_at=0.7, close_at=0.9):
        super().__init__(horizon, rng)
        self.target, self.discount, self.open_at, self.close_at = target, discount, open_at, close_at
        self.a = np.ones((N_TOPICS, N_LEVELS))
        self.b = np.ones((N_TOPICS, N_LEVELS))
        self.active = np.zeros((N_TOPICS, N_LEVELS), dtype=bool)
        self.active[0, 0] = True

    def choose(self, student=None):
        mean = self.a / (self.a + self.b)
        n = self.a + self.b - 2
        mastered = (mean >= self.close_at) & (n >= 4)
        mask = self.active & ~mastered
        if not mask.any():
            mask = self.active
        draw = self.rng.beta(self.a, self.b)
        score = np.where(mask, -np.abs(draw - self.target), -np.inf)
        i = int(np.argmax(score))
        return i // N_LEVELS, i % N_LEVELS

    def update(self, topic, level, correct):
        super().update(topic, level, correct)
        self.a = 1 + (self.a - 1) * self.discount
        self.b = 1 + (self.b - 1) * self.discount
        self.a[topic, level] += correct
        self.b[topic, level] += 1 - correct
        mean = self.a[topic, level] / (self.a[topic, level] + self.b[topic, level])
        if self.a[topic, level] + self.b[topic, level] - 2 >= 2.5 and mean >= self.open_at:
            if level + 1 < N_LEVELS:
                self.active[topic, level + 1] = True
            if level >= 1 and topic + 1 < N_TOPICS:
                self.active[topic + 1, 0] = True


class Oracle(Policy):
    """Upper bound: sees true mastery and asks the question with the largest expected gain."""
    name = "Oracle (sees mastery)"

    def choose(self, student=None):
        gains = np.array([[student.expected_gain(k, d) for d in range(N_LEVELS)] for k in range(N_TOPICS)])
        i = int(np.argmax(gains))
        return i // N_LEVELS, i % N_LEVELS


ALL = [LinearCurriculum, RandomPolicy, Staircase, ThompsonTarget, LearningProgressBandit, Oracle]
