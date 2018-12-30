from abc import ABC, abstractmethod

import torch
from torchvision.transforms import functional as TVF
import numpy as np


class Policy(ABC):
    @abstractmethod
    def action(self, screen, observation):
        """

        :param screen: batch, channels, height, width
        :param observation: batch, channels, height, width
        :return: an action in the embedding space, will need to be converted to the simulator space
        """
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, env):
        self.env = env

    # todo decide if action will be in embedded or simulator space
    # todo if so then embedding should be part of the policy
    def action(self, screen, observation):
        return self.env.action_space.sample()


class ActionEmbedding:
    """
    Simple one-hot embedding of the action space
    """

    def __init__(self, env):
        self.env = env

    def tensor(self, action):
        action_t = torch.zeros(self.env.action_space.n)
        action_t[action] = 1.0
        return action_t

    def numpy(self, action):
        action_n = np.zeros(self.env.action_space.n)
        action_n[action] = 1.0
        return action_n

    def embedding_to_action(self, index):
        return index

    def start_tensor(self):
        return torch.zeros(self.env.action_space.n)

    def start_numpy(self):
        return np.zeros(self.env.action_space.n)


class ToTensor(object):
    def __init__(self, action_embedding):
        self.embed_action = action_embedding

    def __call__(self, screen, observation, reward, done, info, action):
        screen_t = TVF.to_tensor(screen)
        observation_t = torch.Tensor(observation)
        reward_t = torch.Tensor([reward])
        done_t = torch.Tensor([done])
        action_t = self.embed_action.tensor(action)
        return screen_t, observation_t, reward_t, done_t, info, action_t


class Rollout:
    def __init__(self, env):
        self.env = env

    def rollout(self, policy, episode, max_timesteps=100):
        observation = self.env.reset()
        screen = self.env.render(mode='rgb_array')

        for t in range(max_timesteps):

            action = policy.action(screen, observation)
            observation, reward, done, info = self.env.step(action)
            screen = self.env.render(mode='rgb_array')

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


class RolloutGen(object):
    """
    Wrap gym in a generator object
    """

    def __init__(self, env, policy, action_embedding, populate_screen=True, render_to_window=False):
        """

        :param env: gym environment
        :param policy: policy to select actions in the environment
        :param populate_screen: populates the screen return parameter with numpy array of RGB data
        :param render_to_window: render the output to a window
        """
        self.env = env
        self.policy = policy
        self.done = True
        self.action = None
        self.populate_screen = populate_screen
        self.render_to_window = render_to_window
        self.to_tensor = ToTensor(action_embedding)

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def render(self):
        screen = None
        if self.populate_screen:
            screen = self.env.render(mode='rgb_array')
        if self.render_to_window:
            self.env.render()
        return screen

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        screen = self.render()
        screen_t, observation_t, reward_t, done_t, info, action_t = \
            self.to_tensor(screen, observation, reward, done, info, action)
        action = self.policy.action(screen_t.unsqueeze(0), observation_t.unsqueeze(0))
        return screen, observation, reward, done, info, action

    def next(self):

        if self.done:
            observation = self.env.reset()
            screen = self.render()
            reward = 0
            self.done = False
            info = {}
            screen_t, observation_t, reward_t, done, info, action = \
                self.to_tensor(screen, observation, reward, self.done, info, 0)
            self.action = self.policy.action(screen_t.unsqueeze(0), observation_t.unsqueeze(0))
            return screen, observation, reward, self.done, info, self.action

        else:
            screen, observation, reward, done, info, action = self.step(self.action)
            self.action = action
            self.done = done
            return screen, observation, reward, done, info, action


class GymSimulatorDataset(torch.utils.data.Dataset):
    def __init__(self, env, policy, length, action_embedding, output_in_numpy_format=False, render_to_window=False):
        torch.utils.data.Dataset.__init__(self)
        self.length = length
        self.count = 0
        self.policy = policy
        self.rollout = RolloutGen(env, policy, action_embedding, render_to_window=render_to_window).__iter__()
        self.output_in_numpy_format = output_in_numpy_format
        self.to_tensor = ToTensor(action_embedding=action_embedding)

    def __getitem__(self, index):
        screen, observation, reward, done, info, action = self.rollout.next()

        if not self.output_in_numpy_format:
            screen, observation, action, reward, info, done = \
                self.to_tensor(screen, observation, reward, done, info, action)

        self.count += 1

        return screen, observation, action, reward, done

    def __len__(self):
        return self.length
