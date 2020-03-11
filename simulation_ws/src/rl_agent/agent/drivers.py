# Import simulation Env
# Import training required packags


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

torch.backends.cudnn.enabled = True

from rocket.ignite.layers import downsample3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%

class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()
        self.downs = [
                downsample3D(3, 32, [3, 4, 4], [1, 2, 2], padding=[0, 1, 1], apply_batchnorm=False),
                # (bs, 4, 128, 128, 64)
                downsample3D(32, 64, [2, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 64, 64, 128)
                downsample3D(64, 128, [1, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 4, 32, 256)
                downsample3D(128, 256, [1, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 4, 32, 256)
                downsample3D(256, 512, [1, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 4, 32, 256)
        ]
        self.chain = nn.ModuleList([*self.downs, nn.AvgPool3d([1, 4, 4]), nn.Flatten()])

    def forward(self, x):
        for layer in self.chain:
            x = layer(x)
        return x


# %%

class Target(nn.Module):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, feature_extract, lr=0.05):
        super(Target, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.lr = lr
        self.advs = []
        for i in range(action_size):
            self.advs += [nn.Linear(feature_size,
                                    feature_size)]
        self.advs = nn.ModuleList(self.advs)
        self.value = nn.Linear(feature_size,
                               feature_size)

        self.fc1 = nn.Linear(feature_size,
                             feature_size)

        self.fc2 = nn.Linear(feature_size,
                             feature_size)
        self.fc3 = nn.Linear(feature_size,
                             feature_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, mode='exp_range', base_lr=1e-8, max_lr=lr,
                                                           step_size_up=50)

    def forward(self, state):
        feature = self.feature_extract.forward(state)
        advs = [adv(feature) for adv in self.advs]
        x = torch.stack((*advs, self.value(feature)), -2)
        x = torch.sum(x, -2)
        x = nn.ELU()(x)
        x = self.fc1(x)
        x = nn.ELU()(x)
        x = self.fc2(x)
        x = nn.ELU()(x)
        x = self.fc3(x)
        x = nn.ELU()(x)
        return x

    def loss(self, inputs):
        states, _, rewards, next_states = inputs
        loss = torch.sum(nn.SmoothL1Loss(reduction='none')(
                self.forward(states),
                self.feature_extract.forward(next_states)),
                dim=1)

        return torch.sum(rewards * loss)

    def minimize(self, inputs):
        self.optimizer.zero_grad()
        loss = self.loss(inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(),10)
        self.optimizer.step()
        self.scheduler.step()


# %%

class Forward(nn.Module):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, feature_extract, lr=0.05):
        super(Forward, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.lr = lr
        self.preview = []
        for i in range(action_size):
            self.preview.append(nn.Linear(feature_size,
                                          feature_size))
        self.preview = nn.ModuleList(self.preview)
        self.fc1 = nn.Linear(feature_size,
                             feature_size)

        self.fc2 = nn.Linear(feature_size,
                             feature_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.85)

    def forward(self, state, action):
        feature = self.feature_extract.forward(state)
        preview = [pre(feature) for pre in self.preview]
        preview = torch.stack(tuple(preview), -2)
        action = torch.as_tensor(action)
        batch_size, *shape = action.size()
        action = action.reshape((batch_size, 1, *shape))
        x = torch.bmm(action, preview)
        x = torch.sum(x, -2)
        x = nn.ELU()(x)
        x = self.fc1(x)
        x = nn.ELU()(x)
        x = self.fc2(x)
        return x

    def loss(self, inputs):
        states, actions, _, next_states = inputs
        loss = nn.SmoothL1Loss(reduction='mean')(
                self.forward(states, actions),
                self.feature_extract.forward(next_states))
        return loss

    def minimize(self, inputs):
        self.optimizer.zero_grad()
        loss = self.loss(inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(),10)
        self.optimizer.step()
        self.scheduler.step()


# %%
class Inverse(nn.Module):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, feature_extract, lr=0.05):
        super(Inverse, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.lr = lr

        self.fc1 = nn.Linear(feature_size,
                             feature_size)

        self.fc2 = nn.Linear(feature_size,
                             int(feature_size / action_size))
        self.fc3 = nn.Linear(int(feature_size / action_size),
                             action_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def forward(self, state, next_state):
        # Expected features input instead of Raw image
        x = torch.sub(next_state, state)
        x = nn.ELU()(x)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        x = nn.GELU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = nn.Softmax(dim=-1)(x)
        return x

    def loss(self, inputs):
        states, actions, _, next_states = inputs
        loss = nn.BCEWithLogitsLoss(reduction='mean')(
                self.forward(self.feature_extract.forward(states),
                             self.feature_extract.forward(next_states)), actions)
        return loss

    def minimize(self, inputs):
        self.optimizer.zero_grad()
        loss = self.loss(inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(),10)
        self.optimizer.step()
        self.scheduler.step()


# %%
class IntrinsicReward(nn.Module):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, feature_extract, lr=0.05):
        super(IntrinsicReward, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.lr = lr
        self.Forward = Forward(action_size, feature_size, feature_extract, lr)
        self.Inverse = Inverse(action_size, feature_size, feature_extract, lr)
        self.Target = Target(action_size, feature_size, feature_extract, lr)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, mode='exp_range', base_lr=1e-8, max_lr=lr,
                                                           step_size_up=50)
        self.fc1 = nn.Linear(feature_size,
                             feature_size)

        self.fc2 = nn.Linear(feature_size,
                             feature_size)
        self.fc3 = nn.Linear(feature_size,
                             action_size)

    def forward(self, state):
        # Expected features input instead of Raw image
        expected_state = self.Target.forward(state)
        expected_action = self.Inverse(self.feature_extract.forward(state),expected_state)
        # next_state = self.Forward.forward(state, expected_action)
        x = self.fc1(self.feature_extract(state))
        # x = nn.GELU()(x)
        # x = self.fc2(x)
        # x = nn.GELU()(x)
        # x = self.fc3(x)
        # x = nn.ReLU()(x)

        # x = nn.Softmax(dim=1)(x)
        return expected_action

    def loss(self, inputs):
        states, actions, rewards, next_states = inputs
        curiosity = torch.mean(nn.SmoothL1Loss(reduction='none')(
                self.Forward.forward(states, actions),
                self.feature_extract.forward(next_states)), -1)
        loss = torch.sum(nn.BCEWithLogitsLoss(reduction='none')(
                self.forward(states),
                actions), dim=1)
        # print(loss.size())
        # print((rewards+curiosity).size())
        return torch.mean(loss * (rewards + curiosity))

    def minimize(self, inputs):
        self.optimizer.zero_grad()
        loss = self.loss(inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(),10)
        self.optimizer.step()
        self.scheduler.step()
        models_to_update = [self.Target, self.Forward, self.Inverse]
        for model in models_to_update:
            model.minimize(inputs)

    def cuda(self, device=None):
        super(IntrinsicReward, self).cuda()
        models_to_update = [self.Target, self.Forward, self.Inverse]
        for model in models_to_update:
            model.cuda(device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, mode='exp_range', base_lr=1e-8, max_lr=self.lr,
                                                           step_size_up=50)
        return self._apply(lambda t: t.cuda(device))
