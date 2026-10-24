"""Valve anti-oscillation policies, simulated offline on decision sequences.

Policy strings (used by e8_savings.py):
  none         every decision goes to the valve,
  time:MS      the deployed rule of modbus_controller (Magdalena-CV-Pipeline,
               branch pipeline_with_yolo): a state change is rejected if the
               previous change happened less than MS ms before,
  kofn:N/K     change the state only when K of the last N decisions disagree with it,
  asym:N/K     open at once on a single plant decision (fail-safe towards
               application), close only after K of the last N say no_plant.
"""

from collections import deque


class TimeDebouncer:
    def __init__(self, min_interval_s, initial=True):
        self.min_interval = min_interval_s
        self.state = initial
        self.last_change = float("-inf")

    def update(self, plant, t):
        if plant != self.state and t - self.last_change >= self.min_interval:
            self.state = plant
            self.last_change = t
        return self.state


class KofNDebouncer:
    def __init__(self, n=3, k=2, k_open=None, initial=True):
        self.k, self.k_open = k, k_open or k
        self.window = deque(maxlen=n)
        self.state = initial

    def update(self, plant, t=None):
        self.window.append(bool(plant))
        votes_open = sum(self.window)
        votes_close = len(self.window) - votes_open
        if not self.state and votes_open >= self.k_open:
            self.state = True
        elif self.state and votes_close >= self.k:
            self.state = False
        return self.state


def make(policy, initial=True):
    kind, _, arg = policy.partition(":")
    if kind == "none":
        return KofNDebouncer(1, 1, initial=initial)
    if kind == "time":
        return TimeDebouncer(float(arg) / 1000, initial)
    n, k = (int(x) for x in arg.split("/"))
    if kind == "kofn":
        return KofNDebouncer(n, k, initial=initial)
    if kind == "asym":
        return KofNDebouncer(n, k, k_open=1, initial=initial)
    raise ValueError(f"unknown policy {policy}")


def run(decisions, times, policy, initial=True):
    d = make(policy, initial)
    return [d.update(bool(x), t) for x, t in zip(decisions, times)]


def toggles(states):
    return sum(1 for a, b in zip(states, states[1:]) if a != b)
