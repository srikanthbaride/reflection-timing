import os, csv, random
import numpy as np

from srpi.envs.gridworld import GridWorld, ACTIONS

class ReflectionMemory:
    """A tiny text-memory that stores simple 'lessons' and can bias actions."""
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.items = []

    def add(self, s, a, info:str):
        self.items.append({"state": s, "action": a, "info": info})
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def suggest(self, state_idx):
        """Return a small bias vector over actions based on 'lessons' that match state index."""
        # Very naive: if we have a lesson for this state, discourage repeating the same failed action
        bias = np.zeros(4, dtype=float)
        for it in self.items:
            if it["state"] == state_idx and "avoid_action" in it["info"]:
                a_bad = it["action"]
                bias[a_bad] -= 0.5
        return bias

def manhattan_policy(obs_flat, size):
    """Scripted greedy policy: move toward goal by Manhattan distance."""
    idx = int(np.argmax(obs_flat))
    x, y = divmod(idx, size)
    gx, gy = size-1, size-1
    # compute deltas
    dx = gx - x
    dy = gy - y
    # Prefer moving along the larger absolute difference first
    # Return a probability over 4 actions (up,down,left,right) as a softmax on scores
    scores = np.array([0.0,0.0,0.0,0.0])
    if dx < 0: scores[0] += 1.0   # up
    if dx > 0: scores[1] += 1.0   # down
    if dy < 0: scores[2] += 1.0   # left
    if dy > 0: scores[3] += 1.0   # right
    # mild softness
    probs = np.exp(scores) / np.exp(scores).sum()
    return probs

def reflection_string(event:str, state_idx:int, action:int, reward:float, done:bool):
    # Minimal reflection text
    base = f"event={event} state={state_idx} action={action} reward={reward:.2f} done={done}"
    if reward > 0 and done:
        base += " reached_goal"
    if reward < 0:
        base += " step_penalty"
    return base

def state_index_from_obs(obs_flat, size):
    return int(np.argmax(obs_flat))


def run_mode(mode:str, env_cfg:dict, episodes:int, memory_capacity:int, eps:float, csv_writer):
    env = GridWorld(size=env_cfg["size"],
                    start=tuple(env_cfg["start"]),
                    goal=tuple(env_cfg["goal"]),
                    step_penalty=env_cfg["step_penalty"],
                    goal_reward=env_cfg["goal_reward"],
                    max_steps=env_cfg["max_steps"])
    mem = ReflectionMemory(capacity=memory_capacity)

    for ep in range(1, episodes+1):
        obs = env.reset()
        done = False
        steps = 0
        reflections = 0
        total_r = 0.0
        success = 0

        while not done:
            base_probs = manhattan_policy(obs, env.size)
            # bias from memory (disabled in no_reflection)
            if mode == "no_reflection":
                bias = 0.0
                logits = np.log(base_probs + 1e-9)
            else:
                bias = mem.suggest(state_index_from_obs(obs, env.size))
                logits = np.log(base_probs + 1e-9) + bias

            # eps-greedy exploration
            if np.random.rand() < eps:
                a = np.random.randint(4)
            else:
                a = int(np.argmax(logits))

            next_obs, r, done, info = env.step(a)
            total_r += r
            steps += 1

            # Decide whether to reflect now based on mode
            reflect_now = False
            if mode == "per_step":
                reflect_now = True
            elif mode == "failure_only":
                reflect_now = False
            elif mode == "success_only":
                reflect_now = False
            elif mode == "no_reflection":
                reflect_now = False

            if reflect_now and mode != "no_reflection":
                reflections += 1
                s_idx = state_index_from_obs(obs, env.size)
                # record a simple "avoid" lesson when moving didn't change distance (heuristic)
                def d2g(ob):
                    i = int(np.argmax(ob)); x,y = divmod(i, env.size); gx,gy = env.size-1, env.size-1
                    return abs(gx-x)+abs(gy-y)
                if d2g(next_obs) >= d2g(obs):
                    mem.add(s_idx, a, "avoid_action")

            obs = next_obs

        # After episode ends: episode-level reflection for failure/success modes
        reached_goal = (env.pos == env.goal)
        if reached_goal: success = 1

        if mode == "failure_only" and not reached_goal:
            reflections += 1
            s_idx = state_index_from_obs(obs, env.size)
            mem.add(s_idx, 0, "avoid_action")
        if mode == "success_only" and reached_goal:
            reflections += 1
            gx, gy = env.size-1, env.size-1
            mem.add(gx*env.size+gy, 2, "avoid_action")
            mem.add(gx*env.size+gy, 0, "avoid_action")

        csv_writer.writerow({
            "mode": mode,
            "episode": ep,
            "success": success,
            "steps": steps,
            "return": total_r,
            "reflections": reflections
        })

def run_experiment(cfg_path:str):
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)
    out_csv = os.path.join(cfg["experiment"]["output_dir"], "reflection_timing_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode","episode","success","steps","return","reflections"])
        writer.writeheader()
        for mode in cfg["reflect"]["modes"]:
            run_mode(mode, cfg["env"], cfg["reflect"]["episodes_per_mode"],
                     cfg["reflect"]["memory_capacity"], cfg["reflect"]["exploration_eps"], writer)
    print(f"Done. Metrics saved to {out_csv}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    run_experiment(args.config)
