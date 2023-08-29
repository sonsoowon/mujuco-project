from pathlib import Path
from omegaconf import OmegaConf


exp_config = OmegaConf.create({
        "seed": 0,  # environment seed
        "num_envs": 16,  # the number of environments for parallel training
        "num_eval": 10,  # the number of evaluations
        "max_episode_steps": 2048,
        # the maximum number of episode steps. Used in mujoco environments ! Don't change this value
        "num_rollout_steps": 128,  # the number of policy rollout steps
        "num_minibatches": 16,  # The number of minibatches per 1 epoch (Not mibi batch size)
        "total_timesteps": 3000000,  # total number of frames
        "print_interval": 100,  # print iverval of episodic return
        "early_stop_wating_steps": 5000,  # early stopping steps
    })

    # 그리고 ppo style value clip 시도해보기

path_config = OmegaConf.create({
    "logs": Path("./runs"),
    "videos": Path("./videos"),
    "checkpoints": Path("./checkpoints"),
})

ppo_config = OmegaConf.create({
    "anneal_lr": True,
    "anneal_clip_coef": False,
    "update_epochs": 3,  # The number of iteractions of ppo training
    "minibatch_size": 64,
    "lr": 0.0003,  # RMSProp 도전 : epsilon 0.0001
    "max_grad_norm": 2.0,
    "norm_adv": True,
    "clip_coef": 0.25,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,

    "norm_obs": True,
    "obs_max": 0,

    "value_clip": False,
    "value_clip_coef": 1,
})

