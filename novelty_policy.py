import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
import gym
from gym_data import Policy, ActionEmbedding, GymSimulatorDataset


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions.

    :param tuple of (h,w)
    :returns tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = floor(((h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class AtariNoveltyNet(nn.Module):
    """Encoder."""

    def __init__(self, input_shape, action_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        nn.Module.__init__(self)

        self.e_conv1 = nn.Conv2d(3, 32, kernel_size=first_kernel, stride=first_stride)
        self.e_bn1 = nn.BatchNorm2d(32)
        output_shape = conv_output_shape(input_shape, kernel_size=first_kernel, stride=first_stride)

        self.e_conv2 = nn.Conv2d(32, 128, kernel_size=second_kernel, stride=second_stride)
        self.e_bn2 = nn.BatchNorm2d(128)
        output_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

        self.e_conv3 = nn.Conv2d(128, 128, kernel_size=second_kernel, stride=second_stride)
        self.e_bn3 = nn.BatchNorm2d(128)
        self.z_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

        self.e_squeeze1 = nn.Conv2d(128, 32, 1, 1)
        self.e_sq_bn1 = nn.BatchNorm2d(32)

        self.num_features = (32 + action_shape) * self.z_shape[0] * self.z_shape[1]
        self.linear = nn.Linear(self.num_features, 1)

    def forward(self, observation, action):
        """Forward pass.
        :param observation (batch, channels, height, width)
        :param action (batch, embedding)
        """
        encoded = F.relu(self.e_bn1(self.e_conv1(observation)))
        encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))
        encoded = F.relu(self.e_bn3(self.e_conv3(encoded)))
        encoded = F.relu(self.e_sq_bn1(self.e_squeeze1(encoded)))
        action = action.unsqueeze(-1).unsqueeze(-1)
        action_exp = action.expand(-1, -1, self.z_shape[0], self.z_shape[1])
        encoded = torch.cat((encoded, action_exp), dim=1)
        encoded = encoded.view(-1, self.num_features)
        novelty = F.sigmoid(self.linear(encoded))
        return novelty


class NoveltyPolicy(Policy):
    def __init__(self, env):
        self.env = env
        action_shape = env.action_space.n
        self.action_embed = ActionEmbedding(env)
        self.rand_net = AtariNoveltyNet((210, 160), action_shape)
        self.dist_net = AtariNoveltyNet((210, 160), action_shape)
        self.optimizer = Adam(lr=1e-3, params=self.dist_net.parameters())
        self.criterion = MSELoss()
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.rand_net = self.rand_net.to(device)
        self.dist_net = self.dist_net.to(device)
        return self

    def action(self, screen, observation):
        screen = screen.to(self.device)
        observation = observation.to(self.device)
        error = torch.zeros(env.action_space.n).to(self.device)
        batch_size = screen.size(0)

        # compute the novelty of each action in this state
        #for action_i in range(env.action_space.n):

        actions = torch.eye(env.action_space.n).to(self.device)
        #action = self.action_embed.tensor(action_i).to(self.device)

        screen_exp = screen.expand(env.action_space.n, -1, -1, -1)

        target = self.rand_net(screen_exp, actions)
        prediction = self.dist_net(screen_exp, actions)
        error = (prediction - target)
        print(error.transpose(0,1))

        #print(prediction, target)
        # select the most novel action
        error[error < 0] = error[error < 0] * -1
        #print(error)
        weighted_novelty = F.softmax(error.squeeze(), dim=0)
        #print(weighted_novelty)
        values, indices = torch.topk(weighted_novelty, 1)

        most_novel_action = indices.item()
        #print(most_novel_action)

        # train distillation network on selected action/state
        action = self.action_embed.tensor(most_novel_action).to(self.device).unsqueeze(0)
        novel_targets = self.rand_net(screen, action)
        novel_targets.detach()

        self.optimizer.zero_grad()
        prediction = self.dist_net(screen, action)
        loss = self.criterion(prediction, novel_targets)
        loss.backward()
        self.optimizer.step()

        return most_novel_action


if __name__ == '__main__':

    #env = gym.make('MontezumaRevenge-v0')
    env = gym.make('SpaceInvaders-v0')
    policy = NoveltyPolicy(env).to('cuda')
    action_embedding = ActionEmbedding(env)
    gym_dataset = GymSimulatorDataset(env, policy, 10000, action_embedding, render_to_window=True)
    gym_loader = DataLoader(gym_dataset)
    for frame in gym_loader:
        pass
