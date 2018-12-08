import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOAgent:
    buffer_attrs = [
        "states", "actions", "next_states",
        "rewards", "log_probs", "values", "dones",
    ]

    def __init__(self, env, model, tmax=128, n_epoch=10, batch_size=128,
                 gamma=0.99, gae_lambda=0.96, eps=0.10, device="cpu"):
        self.env = env
        self.model = model
        self.opt_model = optim.Adam(model.parameters(), lr=1e-4)
        self.state_dim = model.state_dim
        self.action_dim = model.action_dim
        self.tmax = tmax
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps = eps
        self.device = device

        self.rewards = None
        self.scores_by_episode = []

        self.reset()

    def to_tensor(self, x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(self.device)

    def reset(self):
        self.brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.last_states = self.to_tensor(env_info.vector_observations)

    def collect_trajectories(self):
        buffer = dict([(k, []) for k in self.buffer_attrs])

        for t in range(self.tmax):
            memory = {}

            # draw action from model
            memory["states"] = self.last_states
            pred = self.model(memory["states"])
            pred = [v.detach() for v in pred]
            memory["actions"], memory["log_probs"], _, memory["values"] = pred

            # one step forward
            actions_np = memory["actions"].cpu().numpy()
            env_info = self.env.step(actions_np)[self.brain_name]
            memory["next_states"] = self.to_tensor(env_info.vector_observations)
            memory["rewards"] = self.to_tensor(env_info.rewards)
            memory["dones"] = self.to_tensor(env_info.local_done, dtype=np.uint8)

            # stack one step memory to buffer
            for k, v in memory.items():
                buffer[k].append(v.unsqueeze(0))

            self.last_states = memory["next_states"]
            r = np.array(env_info.rewards)[None,:]
            if self.rewards is None:
                self.rewards = r
            else:
                self.rewards = np.r_[self.rewards, r]

            if memory["dones"].any():
                rewards_mean = self.rewards.sum(axis=0).mean()
                self.scores_by_episode.append(rewards_mean)
                self.rewards = None
                self.reset()

        for k, v in buffer.items():
            buffer[k] = torch.cat(v, dim=0)

        return buffer

    def calc_returns(self, rewards, values, dones, last_values):
        n_step, n_agent = rewards.shape

        # Create empty buffer
        GAE = torch.zeros_like(rewards).float().to(self.device)
        returns = torch.zeros_like(rewards).float().to(self.device)

        # Set start values
        GAE_current = torch.zeros(n_agent).float().to(self.device)
        returns_current = last_values
        values_next = last_values

        for irow in reversed(range(n_step)):
            values_current = values[irow]
            rewards_current = rewards[irow]
            gamma = self.gamma * (1. - dones[irow].float())

            # Calculate TD Error
            td_error = rewards_current + gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * self.gae_lambda * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE, returns to buffer
            GAE[irow] = GAE_current
            returns[irow] = returns_current

            values_next = values_current

        return GAE, returns

    def step(self):
        self.model.eval()

        # Collect Trajetories
        trajectories = self.collect_trajectories()

        # Calculate Score (averaged over agents)
        score = trajectories["rewards"].sum(dim=0).mean()

        # Append Values collesponding to last states
        last_values = self.model.state_values(self.last_states).detach()
        advantages, returns = self.calc_returns(trajectories["rewards"],
                                                trajectories["values"],
                                                trajectories["dones"],
                                                last_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # concat all agent 
        for k, v in trajectories.items():
            if len(v.shape) == 3:
                trajectories[k] = v.reshape([-1, v.shape[-1]])
            else:
                trajectories[k] = v.reshape([-1])
        advantages = advantages.reshape([-1])
        returns = returns.reshape([-1])

        # Mini-batch update
        self.model.train()
        n_sample = advantages.shape[0]
        n_batch = (n_sample - 1) // self.batch_size + 1
        idx = np.arange(n_sample)
        np.random.shuffle(idx)
        for k, v in trajectories.items():
            trajectories[k] = v[idx]
        advantages, returns = advantages[idx], returns[idx]

        for i_epoch in range(self.n_epoch):
            for i_batch in range(n_batch):
                idx_start = self.batch_size * i_batch
                idx_end = self.batch_size * (i_batch + 1)
                (states, actions, next_states, rewards, old_log_probs,
                 old_values, dones) = [trajectories[k][idx_start:idx_end]
                                       for k in self.buffer_attrs]
                advantages_batch = advantages[idx_start:idx_end]
                returns_batch = returns[idx_start:idx_end]

                _, log_probs, entropy, values = self.model(states, actions)
                ratio = torch.exp(log_probs - old_log_probs)
                ratio_clamped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                adv_PPO = torch.min(ratio * advantages_batch, ratio_clamped * advantages_batch)
                loss_actor = -torch.mean(adv_PPO) - 0.01 * entropy.mean()
                loss_critic = 0.5 * (returns_batch - values).pow(2).mean()
                loss = loss_actor + loss_critic

                self.opt_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
                self.opt_model.step()
                del(loss)

        self.model.eval()

        return score
