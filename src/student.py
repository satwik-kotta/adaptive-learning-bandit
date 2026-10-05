"""A simulated student working through a six-topic maths sequence.

The tutor only ever sees right/wrong answers. Mastery is hidden state.

Assumptions (each one is a parameter, varied in the robustness sweep):
  * answering:   P(correct) rises with (mastery - difficulty), with guessing and slips
  * learning:    a question teaches most when it sits slightly above current mastery
                 (too easy or too hard teaches little); wrong answers teach less
  * prerequisites: topic k is hard to learn until topic k-1 is mostly mastered
  * forgetting:  topics not practised decay slowly
  * students differ in starting knowledge and learning speed
"""
from dataclasses import dataclass, replace

import numpy as np

TOPICS = ["Fractions", "Decimals", "Ratios", "Percentages", "Linear equations", "Inequalities"]
LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])          # question difficulty on the mastery scale
N_TOPICS, N_LEVELS = len(TOPICS), len(LEVELS)


@dataclass(frozen=True)
class World:
    discrimination: float = 8.0     # how sharply P(correct) depends on mastery - difficulty
    guess: float = 0.15
    slip: float = 0.08
    stretch: float = 0.10           # best question sits this far above current mastery
    width: float = 0.15             # how forgiving learning is to badly pitched questions
    wrong_factor: float = 0.6       # share of the gain kept when the answer is wrong
    prereq_gate: float = 8.0        # steepness of the prerequisite gate
    forget: float = 0.001           # per-question decay of unpractised topics
    lr_mean: float = 0.045          # median mastery gained from a perfectly pitched question
    lr_sigma: float = 0.35          # spread of learning speed across students (log scale)

    def but(self, **kw):
        return replace(self, **kw)


class Student:
    def __init__(self, world: World, rng: np.random.Generator):
        self.w, self.rng = world, rng
        start = rng.uniform(0.05, 0.40)
        self.mastery = np.clip(start * 0.6 ** np.arange(N_TOPICS) + rng.normal(0, 0.03, N_TOPICS), 0.0, 1.0)
        self.lr = world.lr_mean * np.exp(rng.normal(0, world.lr_sigma))

    def p_correct(self, topic, level):
        w = self.w
        z = w.discrimination * (self.mastery[topic] - LEVELS[level])
        return w.guess + (1 - w.guess - w.slip) / (1 + np.exp(-z))

    def expected_gain(self, topic, level):
        """Expected mastery gain from one question (used by the oracle only)."""
        w = self.w
        gate = 1.0 if topic == 0 else 1 / (1 + np.exp(-w.prereq_gate * (self.mastery[topic - 1] - 0.5)))
        fit = np.exp(-((LEVELS[level] - (self.mastery[topic] + w.stretch)) ** 2) / (2 * w.width ** 2))
        p = self.p_correct(topic, level)
        return self.lr * gate * fit * (p + (1 - p) * w.wrong_factor) * (1 - self.mastery[topic]) ** 0.5

    def answer(self, topic, level):
        """Ask one question. Returns True/False and updates hidden mastery."""
        w = self.w
        correct = self.rng.random() < self.p_correct(topic, level)
        gate = 1.0 if topic == 0 else 1 / (1 + np.exp(-w.prereq_gate * (self.mastery[topic - 1] - 0.5)))
        fit = np.exp(-((LEVELS[level] - (self.mastery[topic] + w.stretch)) ** 2) / (2 * w.width ** 2))
        gain = self.lr * gate * fit * (1.0 if correct else w.wrong_factor) * (1 - self.mastery[topic]) ** 0.5
        self.mastery *= (1 - w.forget)
        self.mastery[topic] = min(1.0, self.mastery[topic] / (1 - w.forget) + gain)
        return bool(correct)

    def post_test(self, rng, per_topic=5):
        """A fixed 30-question test, one of each level per topic. No learning happens."""
        p = np.array([[self.p_correct(k, d) for d in range(N_LEVELS)] for k in range(N_TOPICS)])
        return float((rng.random((N_TOPICS, per_topic)) < p[:, :per_topic]).mean())
