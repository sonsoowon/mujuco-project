import numpy as np
import torch


class PPOBuffer:
    def __init__(self, exp_config, envs, agent, device):
        self.num_rollout_steps = exp_config.num_rollout_steps
        self.num_envs = exp_config.num_envs
        self.envs = envs
        self.agent = agent
        self.device = device

        self.obs = torch.zeros((self.num_rollout_steps, self.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((self.num_rollout_steps, self.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((exp_config.num_rollout_steps, exp_config.num_envs)).to(device)
        self.rewards = torch.zeros((exp_config.num_rollout_steps, exp_config.num_envs)).to(device)
        self.dones = torch.zeros((exp_config.num_rollout_steps, exp_config.num_envs)).to(device)
        self.values = torch.zeros((exp_config.num_rollout_steps, exp_config.num_envs)).to(device)

        self.step = 0

    def rollout(self, obs, done):
        self.obs[self.step] = obs
        self.dones[self.step] = done

        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(obs)
            self.values[self.step] = value.flatten()
            self.actions[self.step] = action
            self.logprobs[self.step] = logprob

        next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        self.rewards[self.step] = torch.tensor(reward).to(self.device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

        self.step += 1

        return next_obs, next_done, infos

    def get_data(self):
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_values = self.values.reshape(-1)

        self.step = 0

        return b_obs, b_logprobs, b_actions, b_values

    def compute_adv_rets(self, b_values, last_obs, last_done, ppo_config):
        with torch.no_grad():
            values = b_values.reshape((self.num_rollout_steps, self.num_envs))
            next_value = self.agent.get_value(last_obs).reshape(1, -1)
            advantages = torch.zeros((len(self.rewards) + 1, self.num_envs)).to(self.device)
            for t in reversed(range(self.num_rollout_steps)):
                if t == self.num_rollout_steps - 1:
                    done_mask = 1.0 - last_done
                    nextvalues = next_value
                else:
                    done_mask = 1.0 - self.dones[t + 1]
                    nextvalues = values[t + 1]
                delta = self.rewards[t] + ppo_config.gamma * nextvalues * done_mask - values[t]
                advantages[t] = delta + ppo_config.gamma * ppo_config.gae_lambda * advantages[t + 1]
            returns = advantages[:-1] + values

        return advantages.reshape(-1), returns.reshape(-1)