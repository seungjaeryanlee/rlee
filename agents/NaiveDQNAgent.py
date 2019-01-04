import random
import time

import torch
import wandb


class NaiveDQNAgent:
    def __init__(self, env, dqn, optimizer, device, DISCOUNT, EPSILON):
        """
        A Deep Q-Network (DQN) agent that can be trained with environments that
        have feature vectors as states and discrete values as actions.
        """
        self.env = env
        self.dqn = dqn
        self.optimizer = optimizer
        self.device = device

        self.DISCOUNT = DISCOUNT
        self.EPSILON = EPSILON

    def act(self, state, epsilon):
        """
        Return an action sampled from an epsilon-greedy policy.

        Parameters
        ----------
        state
            The state to compute the epsilon-greedy action of.
        epsilon : float
            Epsilon in epsilon-greedy policy: probability of choosing a random
            action.

        Returns
        -------
        action : int
            An integer representing a discrete action chosen by the agent.
        """
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.dqn(state.to(self.device)).cpu()
            action = q_values.max(1)[1].item()
        else:
            action = self.env.action_space.sample()

        return action

    def train(self, nb_frames):
        """
        Train the agent by interacting with the environment.

        Parameters
        ----------
        nb_frames: int
            Number of frames to train the agent.
        """
        episode_reward = 0
        loss = torch.FloatTensor([0])
        state = self.env.reset()

        for frame_idx in range(1, nb_frames + 1):
            # Start timer
            t_start = time.time()

            # Interact and save to replay buffer
            action = self.act(state, self.EPSILON)
            next_state, reward, done, _ = self.env.step(action)

            # Train agent
            self.optimizer.zero_grad()
            batch = (state, torch.LongTensor([action]), reward, next_state, done)
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()

            state = next_state
            episode_reward += reward.item()

            if done:
                print('Frame {:5d}/{:5d}\tReturn {:3.2f}\tLoss {:2.4f}'.format(
                    frame_idx + 1, nb_frames, episode_reward, loss.item()))
                wandb.log({
                    'Episode Reward': episode_reward,
                }, step=frame_idx)

                state = self.env.reset()
                episode_reward = 0

            # End timer
            t_end = time.time()

            t_delta = t_end - t_start
            fps = 1 / (t_end - t_start)

            wandb.log({
                'Reward': reward,
                'Loss': loss,
                'Time per frame': t_delta,
                'FPS': fps,
            }, step=frame_idx)

    def _compute_loss(self, batch):
        """
        Compute batch MSE loss between 1-step target Q and prediction Q.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A tuple of batches: (state_batch, action_batch, reward_batch,
            next_state_batch, done_batch).
        DISCOUNT: float
            Discount factor for computing Q-value.

        Returns
        -------
        loss : torch.FloatTensor
            MSE loss of target Q and prediction Q that can be backpropagated.
            Has shape torch.Size([1]).
        """
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # Predicted Q: Q_current(s, a)
        # q_values : torch.Size([1, self.env.action_space.n])
        # action   : torch.Size([1])
        # q_value  : torch.Size([1])
        q_values = self.dqn(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1))[0].cpu()

        # Target Q: r + gamma * max_{a'} Q_target(s', a')
        # next_q_values    : torch.Size([1, self.env.action_space.n])
        # next_q_value     : torch.Size([1])
        # expected_q_value : torch.Size([1])
        with torch.no_grad():
            # Q_target(s', a')
            next_q_values = self.dqn(next_state_batch)
            next_q_value = next_q_values.max(dim=1)[0].squeeze()
            expected_q_value = (reward_batch + self.DISCOUNT * next_q_value * (1 - done_batch)).cpu()

        assert expected_q_value.shape == q_value.shape

        # Compute MSE Loss
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        return loss
