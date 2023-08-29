import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym

from util import get_logger, make_env, make_env_config, save_model, save_train_graph, save_eval_graph
from config import path_config, exp_config, ppo_config
from actor_critic import ActorCritic
from buffer import PPOBuffer

from pathlib import Path


def normalize_obs(obs, obs_max=0):
    normalized_obs = (obs - obs.mean()) / max(obs.std(), 1e-6)
    if obs_max > 0:
        normalized_obs = torch.clamp(normalized_obs, -obs_max, obs_max)
    return normalized_obs


def train(env_id):
    global_logger, writer = get_logger(env_id, path_config)

    # Managing seed for reproducible experiments
    random.seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    torch.manual_seed(exp_config.seed)

    # Set device and make environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv(make_env(env_id, exp_config, path_config, evaluation=False, idx=idx)
                                    for idx in range(exp_config.num_envs))
    env_config = make_env_config(envs)

    # Initialize agent and optimizer
    agent = ActorCritic(env_config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=ppo_config.lr, eps=1e-5)

    buffer = PPOBuffer(exp_config, envs, agent, device)

    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(seed=exp_config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(exp_config.num_envs).to(device)
    num_updates = exp_config.total_timesteps // (exp_config.num_rollout_steps * exp_config.num_envs)  # number of epochs
    save_positions = np.arange(0, num_updates // 10 + num_updates, num_updates // 10)

    for update in range(num_updates + 1):
        # Annealing the rate if instructed to do so.
        if ppo_config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * ppo_config.lr
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(buffer.num_rollout_steps):
            if ppo_config.norm_obs:
                next_obs = normalize_obs(next_obs) # obs normalization
            next_obs, next_done, infos = buffer.rollout(next_obs, next_done)
            global_step += exp_config.num_envs

            if "final_info" not in infos:
                continue
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(
                    f"num_updates: {update}/{num_updates}, global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                global_logger.train_episodic_return.append(info["episode"]["r"])
                global_logger.episodic_return_steps.append(global_step)

        last_obs, last_done = next_obs, next_done

        # Save model
        if update in save_positions:
            global_logger.save_update_steps.append(update)
            save_model(env_id, path_config, agent, update)
        global_logger.global_steps.append(global_step)

        # ! Optimizing the policy and value network
        agent.train()
        batch_size = exp_config.num_rollout_steps * exp_config.num_envs
        mb_size = ppo_config.minibatch_size
        b_inds = np.arange(batch_size)

        b_obs, b_logprobs, b_actions, b_values = buffer.get_data()
        b_advantages, b_returns = buffer.compute_adv_rets(b_values, last_obs, last_done, ppo_config)

        for epoch in range(ppo_config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds]
                if ppo_config.norm_obs:
                    mb_obs = normalize_obs(mb_obs)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)

                mb_advantages = b_advantages[mb_inds]
                if ppo_config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                ###################### Implement here : 3. PPO Loss, Value Loss ########################
                # Be careful about pytorch automatic broadcasting.
                # Policy loss part. Refer the PPO clip loss picture
                pg_loss1 = ratio * mb_advantages
                pg_loss2 = torch.clamp(ratio, 1 - ppo_config.clip_coef, 1 + ppo_config.clip_coef) * mb_advantages
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = F.mse_loss(newvalue, b_returns[mb_inds])

                #b_values[mb_inds] = newvalue
                entropy_loss = entropy.mean()

                ###################### Implement here : 4. Total Loss ########################
                # implement the total loss value by using the coefficients in ppo_config
                total_loss = pg_loss + ppo_config.vf_coef * v_loss - ppo_config.ent_coef * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)
                optimizer.step()


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        global_logger.value_loss.append(v_loss.item())
        global_logger.policy_loss.append(pg_loss.item())
        global_logger.entropy_loss.append(entropy_loss.item())
    envs.close()
    writer.close()
    print("Training is finished...")

    save_model(env_id, path_config, agent, update)
    save_train_graph(env_id, global_logger)

def evaluate(env_id, update):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, exp_config, path_config, evaluation=True, idx=0)]
    )
    env_config = make_env_config(envs)
    device = torch.device("cpu")
    agent = ActorCritic(env_config).to(device)
    ckpt_path = Path(path_config.checkpoints) / Path(env_id)
    file_name = Path(f"PPO_{update}.pt")
    file_path = str(ckpt_path / file_name)
    agent.load_state_dict(torch.load(file_path, map_location=device))
    agent.eval()
    print("Loading model is successful")

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    episodic_returns = []
    while len(episodic_returns) < exp_config.num_eval:
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue
        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue
            print(f"test_episodic_return={info['episode']['r']}")
            episodic_returns.append(info['episode']['r'])

    global_logger, _ = get_logger(env_id, path_config)
    global_logger.test_episodic_return = episodic_returns
    save_eval_graph(env_id, global_logger)



if __name__ == "__main__":
    env_id = "HalfCheetah-v4"
    evaluate(env_id, 1464)


