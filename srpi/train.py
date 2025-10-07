import os, argparse, csv, json
import numpy as np
from srpi.utils.config import load_config
from srpi.utils.misc import set_seed
from srpi.utils.logger import CSVLogger
from srpi.envs.gridworld import GridWorld, ACTIONS
from srpi.agents.policy import MLPPolicy
from srpi.lac.simple_lac import SimpleLAC

def discounted_returns(rewards, gamma):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))

def make_reflection(obs, act, next_obs, reward, done):
    # A stub "reflection" string
    base = f"Action={ACTIONS[act]} reward={reward:.2f} done={done}"
    if reward > 0: base += " reached_goal"
    elif reward < 0: base += " step_penalty"
    return base

def reflection_features(reflection:str, step:int, dist_to_goal:float):
    # Very simple numeric features derived from text metadata
    length = len(reflection)
    has_goal = 1.0 if "reached_goal" in reflection else 0.0
    has_penalty = 1.0 if "step_penalty" in reflection else 0.0
    return np.array([length/100.0, has_goal, has_penalty, step/50.0, dist_to_goal/10.0], dtype=np.float32)

def dist_to_goal_from_obs(obs_flat, size):
    idx = int(np.argmax(obs_flat))
    x, y = divmod(idx, size)
    gx, gy = size-1, size-1
    return abs(gx - x) + abs(gy - y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    exp_dir = cfg["experiment"]["output_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    set_seed(cfg["experiment"]["seed"])

    # Env
    env = GridWorld(size=cfg["env"]["size"],
                    start=tuple(cfg["env"]["start"]),
                    goal=tuple(cfg["env"]["goal"]),
                    step_penalty=cfg["env"]["step_penalty"],
                    goal_reward=cfg["env"]["goal_reward"],
                    max_steps=cfg["env"]["max_steps"])
    obs_dim = env.size * env.size
    act_dim = len(ACTIONS)

    # Agent
    policy = MLPPolicy(obs_dim, act_dim,
                       hidden=cfg["agent"]["policy_hidden"],
                       lr=cfg["train"]["lr"],
                       entropy_coef=cfg["agent"]["entropy_coef"],
                       kl_coef=cfg["agent"]["kl_coef"])

    # LAC
    lac_enabled = cfg["lac"]["enabled"]
    alpha = cfg["lac"]["alpha"]
    sigma_max = cfg["lac"]["sigma_max"]
    lac = SimpleLAC(input_dim=5, hidden=cfg["lac"]["hidden"], lr=cfg["lac"]["lr"], sigma_max=sigma_max)

    logger = CSVLogger(os.path.join(exp_dir, "metrics.csv"))
    ep_count = cfg["train"]["episodes"]
    gamma = cfg["agent"]["gamma"]

    for ep in range(1, ep_count+1):
        obs = env.reset()
        done = False

        ep_obs, ep_acts, ep_rewards, ep_advs_env = [], [], [], []
        lac_xs, lac_targets = [], []
        t = 0
        total_r = 0.0

        while not done:
            a, logp, probs, logits = policy.sample(obs)
            next_obs, r, done, info = env.step(a)
            total_r += r

            reflection = make_reflection(obs, a, next_obs, r, done)
            d2g = dist_to_goal_from_obs(next_obs, env.size)
            x = reflection_features(reflection, t, d2g)
            m, s2 = lac.predict(x)

            ep_obs.append(obs.copy())
            ep_acts.append(a)
            ep_rewards.append(r)

            obs = next_obs
            t += 1

        # compute env advantages
        rets = discounted_returns(ep_rewards, gamma)
        baseline = np.mean(rets)
        env_advs = [G - baseline for G in rets]

        # train LAC on env advantages (supervision)
        for i in range(len(env_advs)):
            # re-construct features for each step
            # (we recompute; in a real codebase you'd store them)
            # Approx dist_to_goal from stored obs->next_obs; here use obs since next not stored.
            d2g = dist_to_goal_from_obs(ep_obs[i], env.size)
            refl = "posthoc"  # placeholder
            x = reflection_features(refl, i, d2g)
            lac_xs.append(x)
            lac_targets.append(env_advs[i])
        if lac_enabled:
            lac.update(lac_xs, lac_targets)

        # blended advantage
        blended = []
        for i in range(len(env_advs)):
            d2g = dist_to_goal_from_obs(ep_obs[i], env.size)
            refl = "posthoc"
            x = reflection_features(refl, i, d2g)
            m, s2 = lac.predict(x)
            use_lac = 1.0 if (lac_enabled and s2 <= sigma_max) else 0.0
            adv = alpha * m * use_lac + (1.0 - alpha) * env_advs[i]
            blended.append(adv)

        policy.update(ep_obs, ep_acts, blended)

        if ep % cfg["train"]["log_every"] == 0:
            logger.log({
                "episode": ep,
                "return": total_r,
                "steps": t,
                "baseline": baseline,
                "mean_env_adv": float(np.mean(env_advs)),
                "mean_blended_adv": float(np.mean(blended)),
            })

        if ep % cfg["train"]["eval_every"] == 0:
            # quick greedy eval
            eval_returns = []
            for _ in range(5):
                obs = env.reset()
                done = False
                R = 0.0
                steps = 0
                while not done:
                    probs, logits = policy.policy(obs)
                    a = int(np.argmax(probs))
                    obs, r, done, _ = env.step(a)
                    R += r
                    steps += 1
                eval_returns.append(R)
            logger.log({
                "episode": ep,
                "eval_return_mean": float(np.mean(eval_returns)),
                "eval_return_std": float(np.std(eval_returns))
            })

    print(f"Done. Metrics at {os.path.join(exp_dir, 'metrics.csv')}")

if __name__ == "__main__":
    main()
