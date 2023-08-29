from pathlib import Path
from omegaconf import OmegaConf

from dataclasses import dataclass
from typing import List
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0 as RecordVideo

from gymnasium.spaces import Discrete, Box
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

@dataclass
class GlobalLogger:
    global_steps: List
    save_update_steps: List
    episodic_return_steps: List
    train_episodic_return: List
    test_episodic_return: List
    policy_loss: List
    value_loss: List
    entropy_loss: List


def make_env(env_id, exp_config, path_config, evaluation=False, idx=0):
    mujoco_env_id = ["Swimmer-v4", "Reacher-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "HalfCheetah-v4",
                     "HumanoidStandup-v4"]

    video_path = Path(path_config.videos)
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if evaluation:
            test_path = Path(f"{env_id}/test")
            video_save_path = str(video_path / test_path)
        else:
            train_path = Path(f"{env_id}/train")
            video_save_path = str(video_path / train_path)
        if idx == 0:
            env = RecordVideo(env, video_save_path, disable_logger=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if env_id in mujoco_env_id:
            env = gym.wrappers.TimeLimit(env, exp_config.max_episode_steps)
        return env

    return thunk


def make_env_config(envs):
    env = envs.envs[0]
    print(env.observation_space)
    print(env.action_space)

    # * observation information
    if isinstance(env.observation_space, Discrete):  # if observation_space is discrete
        state_dim = env.observation_space.n

    else:  # if observation_space is continuous
        if len(env.observation_space.shape) > 1:  # Atari visual observation case
            state_dim = env.observation_space.shape
        else:  # 1D vector observation case (classic control, box2d, mujoco)
            state_dim = env.observation_space.shape[0]

    # * action_space information
    num_discretes = 0
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
        is_continuous = True
    elif isinstance(env.action_space, Discrete):
        action_dim = 1
        num_discretes = env.action_space.n
        is_continuous = False
    env_config = OmegaConf.create({"state_dim": state_dim,
                                   "action_dim": int(action_dim),
                                   "num_discretes": int(num_discretes),
                                   "is_continuous": is_continuous})
    return env_config


def save_model(env_id, path_cfg, actor_critic, update):
    ckpt_path = Path(path_cfg.checkpoints) / Path(f"{env_id}")
    if not ckpt_path.exists():
        ckpt_path.mkdir()
    model_name = Path(f"PPO_{update}.pt")
    model_path = ckpt_path / model_name
    torch.save(actor_critic.state_dict(), str(model_path))
    print(f"model saved to {model_path}")

def save_train_graph(env_id, global_logger):
    x = np.array(global_logger.episodic_return_steps)
    y = np.array(global_logger.train_episodic_return).squeeze(1)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style('darkgrid')
    sns.set_context("poster")
    sns.set(font_scale=1)
    g = sns.lineplot(x=x, y=y).set(title=f'{env_id}: train_episodic_return')
    ax.set(xlim=(0, global_logger.global_steps[-1]))
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.set(xlabel='global_step', ylabel='episodic_return')
    save_path = Path(global_logger.log_path)
    file_name = Path("train_episodic_return.jpg")
    file_path = save_path / file_name
    print(f"figure is saved to {str(file_path)}")
    plt.savefig(str(file_path), dpi=200)

def save_eval_graph(env_id, global_logger):
    y = np.array(global_logger.test_episodic_return)
    y = y.squeeze(1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlabel='episode index', ylabel='episodic_return')
    sns.set_style('darkgrid')
    sns.set_context("poster")
    sns.set(font_scale=1)
    sns.scatterplot(x=range(len(y)), y=y).set(title=f'{env_id}: test_episodic_return')
    save_path = Path(global_logger.log_path)
    file_name = Path("test_episodic_return.jpg")
    file_path = save_path / file_name
    print(f"figure is saved to {str(file_path)}")
    plt.savefig(str(file_path), dpi=200)

def get_logger(env_id, path_config):
    log_path = str(Path(path_config.logs) / Path(env_id))
    writer = SummaryWriter(log_path)
    global_logger = GlobalLogger(
        global_steps=[],
        save_update_steps=[],
        episodic_return_steps=[],
        train_episodic_return=[],
        test_episodic_return=[],
        policy_loss=[],
        value_loss=[],
        entropy_loss=[]
    )
    global_logger.log_path = log_path

    return global_logger, writer