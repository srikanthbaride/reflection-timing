import numpy as np

ACTIONS = ["up", "down", "left", "right"]

class GridWorld:
    def __init__(self, size=5, start=(0,0), goal=(4,4), step_penalty=-0.01, goal_reward=1.0, max_steps=40):
        self.size = size
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.pos = tuple(self.start)
        self.t = 0
        return self._obs()

    def _obs(self):
        # one-hot position in grid
        obs = np.zeros((self.size, self.size), dtype=np.float32)
        obs[self.pos] = 1.0
        return obs.flatten()  # size*size vector

    def step(self, action_idx: int):
        action = ACTIONS[action_idx]
        x, y = self.pos
        if action == "up":
            x = max(0, x-1)
        elif action == "down":
            x = min(self.size-1, x+1)
        elif action == "left":
            y = max(0, y-1)
        else: # right
            y = min(self.size-1, y+1)
        self.pos = (x, y)
        self.t += 1

        r = self.step_penalty
        done = False
        if self.pos == self.goal:
            r = self.goal_reward
            done = True
        if self.t >= self.max_steps:
            done = True

        return self._obs(), r, done, {}
