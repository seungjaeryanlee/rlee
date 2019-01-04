import copy
import random
import time

import torch
import wandb


class DQN2015Agent:
    def __init__(self, env, dqn, optimizer, criterion, replay_buffer, epsilon_func, device,
                 DISCOUNT,
                 BATCH_SIZE,
                 MIN_REPLAY_BUFFER_SIZE,
                 TARGET_UPDATE_FREQ,
                 EVAL_FREQ):
        """
        A Deep Q-Network (DQN) agent that can be trained with environments that
        have feature vectors as states and discrete values as actions. Uses
        experience replay and target network.
        """
        self.env = env
        self.current_dqn = dqn
        self.target_dqn = copy.deepcopy(dqn)
        self.optimizer = optimizer
        self.criterion = criterion
        self.replay_buffer = replay_buffer
        self.epsilon_func = epsilon_func
        self.device = device

        self.DISCOUNT = DISCOUNT
        self.BATCH_SIZE = BATCH_SIZE
        self.MIN_REPLAY_BUFFER_SIZE = MIN_REPLAY_BUFFER_SIZE
        self.TARGET_UPDATE_FREQ = TARGET_UPDATE_FREQ
        self.EVAL_FREQ = EVAL_FREQ

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
                q_values = self.current_dqn(state.to(self.device)).cpu()
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
        episode_length = 0
        loss = torch.FloatTensor([0])
        state = self.env.reset()
        nb_episodes_until_eval = self.EVAL_FREQ

        for frame_idx in range(1, nb_frames + 1):
            # Start timer
            t_start = time.time()

            # Interact and save to replay buffer
            epsilon = self.epsilon_func(frame_idx)
            action = self.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward.item()
            episode_length += 1

            if done:
                state = self.env.reset()
                init_state_value_estimate = self.current_dqn(state.to(self.device)).max(1)[0].cpu().item()

                print('Frame {:5d}/{:5d}\tReturn {:3.2f}\tLoss {:2.4f}'.format(
                    frame_idx + 1, nb_frames, episode_reward, loss.item()))
                wandb.log({
                    'Episode Reward': episode_reward,
                    'Episode Length': episode_length,
                    'Value Estimate of Initial State': init_state_value_estimate,
                }, step=frame_idx)

                episode_reward = 0
                episode_length = 0
                nb_episodes_until_eval -= 1

            # Train DQN if the replay buffer is populated enough
            if len(self.replay_buffer) > self.MIN_REPLAY_BUFFER_SIZE:
                self.optimizer.zero_grad()
                replay_batch = self.replay_buffer.sample(self.BATCH_SIZE)
                loss = self._compute_loss(replay_batch)
                loss.backward()
                self.optimizer.step()

                wandb.log({
                    'Loss': loss,
                }, step=frame_idx)

            # Update Target DQN periodically
            if frame_idx % self.TARGET_UPDATE_FREQ == 0:
                self._update_target()

            # End timer
            t_end = time.time()

            t_delta = t_end - t_start
            fps = 1 / (t_end - t_start)

            wandb.log({
                'Epsilon': epsilon,
                'Reward': reward,
                'Time per frame': t_delta,
                'FPS': fps,
            }, step=frame_idx)

            # Evaluate agent periodically
            # Determined by episodes since environment needs to be reset
            # TODO Copy to NaiveDQN, DQN2013
            if nb_episodes_until_eval <= 0:
                eval_episode_reward = self.eval(nb_episodes=1)[0]
                wandb.log({
                    'Evaluation Episode Reward': eval_episode_reward,
                }, step=frame_idx)
                nb_episodes_until_eval = self.EVAL_FREQ

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
        state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch = self.replay_buffer.sample(self.BATCH_SIZE)
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # Predicted Q: Q_current(s, a)
        # q_values : torch.Size([BATCH_SIZE, self.env.action_space.n])
        # action   : torch.Size([BATCH_SIZE])
        # q_value  : torch.Size([BATCH_SIZE])
        q_values = self.current_dqn(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze().cpu()

        # Target Q: r + gamma * max_{a'} Q_target(s', a')
        # next_q_values    : torch.Size([BATCH_SIZE, self.env.action_space.n])
        # next_q_value     : torch.Size([BATCH_SIZE])
        # expected_q_value : torch.Size([BATCH_SIZE])
        with torch.no_grad():
            # Q_target(s', a')
            next_q_values = self.target_dqn(next_state_batch)
            next_q_value = next_q_values.max(dim=1)[0].squeeze()
            expected_q_value = (reward_batch + self.DISCOUNT * next_q_value * (1 - done_batch)).cpu()

        assert expected_q_value.shape == q_value.shape

        # Compute loss
        loss = self.criterion(q_value, expected_q_value.detach())

        return loss

    def _update_target(self):
        """
        Update weights of Target DQN with weights of current DQN.
        """
        self.target_dqn.load_state_dict(self.current_dqn.state_dict())

    def eval(self, nb_episodes=1):
        """
        Evaluate agent with greedy action selection.

        Parameters
        ----------
        nb_epsiodes : int
            Number of episodes to evaluate the agent for. Defaults to 1.

        Returns
        -------
        eval_episode_rewards : list of float
            List of cumulative rewards for each episode.
        """
        eval_episode_rewards = []

        for _ in range(nb_episodes):
            state = self.env.reset()
            done = False
            eval_episode_reward = 0

            while not done:
                action = self.act(state, epsilon=0)
                state, reward, done, info = self.env.step(action)
                eval_episode_reward += reward

            eval_episode_rewards.append(eval_episode_reward)

        return eval_episode_rewards
