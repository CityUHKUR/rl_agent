# Import simulation Env
# Import training required packags


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

torch.backends.cudnn.enabled = True

from rocket.ignite.layers import downsample3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%

class FeatureExtract(nn.Module):
    def __init__(self, feature_size):
        super(FeatureExtract, self).__init__()
        self.downs = [
                downsample3D(3, 32, [3, 4, 4], [1, 2, 2], padding=[0, 1, 1], apply_batchnorm=False),
                # (bs, 4, 128, 128, 64)
                downsample3D(32, 64, [2, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 64, 64, 128)
                downsample3D(64, 128, [1, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 4, 32, 256)
                downsample3D(128, 256, [1, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 4, 32, 256)
                downsample3D(256, 512, [1, 4, 4], [1, 2, 2], padding=[0, 1, 1]),  # (bs, 4, 4, 32, 256)
                downsample3D(512, 512, [1, 4, 4], [1, 1, 1], padding=[0, 0, 0], apply_batchnorm=False)
                # (bs, 4, 4, 32, 256)

        ]
        self.linears = [
                nn.Linear(feature_size, feature_size),
                nn.Linear(feature_size, feature_size),
        ]

        self.fc_out = nn.Linear(feature_size, feature_size)
        self.chain = nn.ModuleList([*self.downs,
                                    nn.Flatten(),
                                    *self.linears
                                    ])

    def forward(self, x):
        for layer in self.chain:
            if type(layer) is nn.Linear:
                x = layer(x)
                x = nn.ELU()(x)
            else:
                x = layer(x)
        return self.fc_out(x)


# %%
class ExtrinsicCritic(nn.Module):
    def __init__(self, action_size, feature_size, feature_extract, lr=0.05):
        super(ExtrinsicCritic, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.lr = lr
        self.predict = [nn.Linear(action_size, feature_size),
                        nn.Linear(feature_size, feature_size),
                        nn.Linear(feature_size, feature_size)
                        ]
        self.predict = nn.ModuleList(self.predict)
        self.fc1 = nn.Linear(2 * feature_size,
                             feature_size)
        self.fc2 = nn.Linear(feature_size,
                             feature_size)
        self.fc3 = nn.Linear(feature_size,
                             feature_size)
        self.value = nn.Linear(feature_size,
                               1)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, mode='exp_range', base_lr=1e-8, max_lr=lr,
        #                                                    step_size_up=50)

    def forward(self, state, action):
        feature = self.feature_extract.forward(state)
        batch_size, *shape = action.size()
        action_one_hot = F.one_hot(action, num_classes=self.action_size)
        x = action_one_hot.to(dtype=torch.float)
        for layer in self.predict:
            if type(layer) is nn.Linear:
                x = layer(x)
                x = nn.ELU()(x)
            else:
                x = layer(x)
        x = torch.cat((
                feature,
                x
        ),
                dim=-1
        )
        x = self.fc1(x)
        x = nn.ELU()(x)
        x = self.fc2(x)
        x = nn.ELU()(x)
        x = self.fc3(x)
        x = nn.ELU()(x)
        x = self.value(x)
        return x

    def loss(self, inputs):
        states, actions, rewards, _, _, _ = inputs
        batch_size, *shape = rewards.size()
        rewards = rewards.reshape(batch_size, 1, *shape)
        loss = nn.SmoothL1Loss(reduction='mean')(
                self.forward(states, actions.squeeze()),
                rewards)
        return loss

    def minimize(self, inputs):
        loss = self.loss(inputs)
        self.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)
        self.optimizer.step()


# %%
class IntrinsicCritic(nn.Module):
    def __init__(self, action_size, feature_size, feature_extract, Forward, lr=0.05):
        super(IntrinsicCritic, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.Forward = Forward
        self.lr = lr
        self.predict = [nn.Linear(action_size, feature_size),
                        nn.Linear(feature_size, feature_size),
                        nn.Linear(feature_size, feature_size)
                        ]
        self.predict = nn.ModuleList(self.predict)
        self.fc1 = nn.Linear(2 * feature_size,
                             feature_size)
        self.fc2 = nn.Linear(feature_size,
                             feature_size)
        self.fc3 = nn.Linear(feature_size,
                             feature_size)
        self.value = nn.Linear(feature_size,
                               1)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, mode='exp_range', base_lr=1e-8, max_lr=lr,
        #                                                    step_size_up=50)

    def forward(self, state, action):
        feature = self.feature_extract.forward(state)
        action_one_hot = F.one_hot(action, num_classes=self.action_size)
        x = action_one_hot.to(dtype=torch.float)
        for layer in self.predict:
            if type(layer) is nn.Linear:
                x = layer(x)
                x = nn.ELU()(x)
            else:
                x = layer(x)
        x = torch.cat((
                feature,
                x
        ),
                dim=-1
        )
        x = self.fc1(x)
        x = nn.ELU()(x)
        x = self.fc2(x)
        x = nn.ELU()(x)
        x = self.fc3(x)
        x = nn.ELU()(x)
        x = self.value(x)
        return x

    def loss(self, inputs):
        states, actions, _, _, next_states, _ = inputs
        curiosity = nn.SmoothL1Loss(reduction='none')(
                self.Forward.forward(states, actions.squeeze()),
                self.feature_extract.forward(next_states))
        curiosity = torch.mean(curiosity, dim=-1, keepdim=True)
        loss = nn.SmoothL1Loss(reduction='mean')(
                self.forward(states, actions),
                curiosity)
        return loss

    def minimize(self, inputs):
        loss = self.loss(inputs)
        self.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)
        self.optimizer.step()


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

        self.predict = [nn.Linear(action_size, feature_size),
                        nn.Linear(feature_size, feature_size),
                        nn.Linear(feature_size, feature_size)
                        ]
        self.predict = nn.ModuleList(self.predict)
        self.fc1 = nn.Linear(2 * feature_size,
                             feature_size)

        self.fc2 = nn.Linear(feature_size,
                             feature_size)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.85)

    def forward(self, state, action):
        feature = self.feature_extract.forward(state)
        action_one_hot = F.one_hot(action, num_classes=self.action_size)
        x = action_one_hot.to(dtype=torch.float)
        for layer in self.predict:
            if type(layer) is nn.Linear:
                x = layer(x)
                x = nn.ELU()(x)
            else:
                x = layer(x)
        x = torch.cat((
                feature,
                x
        ),
                dim=-1
        )
        # x = torch.sum(x, dim=-2)

        x = self.fc1(x)
        x = nn.ELU()(x)
        x = self.fc2(x)
        return x

    def loss(self, inputs):
        states, actions, _, _, next_states, _ = inputs
        loss = nn.SmoothL1Loss(reduction='mean')(
                self.forward(states, actions.squeeze()),
                self.feature_extract.forward(next_states))
        return loss

    def minimize(self, inputs):
        loss = self.loss(inputs)
        self.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)
        self.optimizer.step()
        # self.scheduler.step()


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
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
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
        x = nn.LogSoftmax(dim=-1)(x)
        return x

    def loss(self, inputs):
        states, actions, _, _, next_states, _ = inputs
        loss = nn.NLLLoss(reduction='mean')(
                input=self.forward(self.feature_extract.forward(states),
                                   self.feature_extract.forward(next_states)), target=actions.squeeze())
        return loss

    def minimize(self, inputs):
        loss = self.loss(inputs)
        self.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)
        self.optimizer.step()
        # self.scheduler.step()


# %%
class Actor(nn.Module):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, feature_extract, lr=0.05):
        super(Actor, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.lr = lr

        self.fc1 = nn.Linear(feature_size,
                             feature_size)

        self.fc2 = nn.Linear(feature_size,
                             int(feature_size / action_size))
        self.fc3 = nn.Linear(int(feature_size / action_size),
                             action_size)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def forward(self, state, ):
        # Expected features input instead of Raw image
        x = self.feature_extract(state)
        x = nn.GELU()(x)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        x = nn.GELU()(x)
        x = self.fc3(x)
        return x

    def policy(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)


# %%
class CuriosityAgent(nn.Module):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, feature_extract, lr=0.001, scaling_factor=0.8, epsilon=0.2,
                 entropy_limit=100):
        super(CuriosityAgent, self).__init__()
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.lr = lr
        self.epsilon = epsilon
        self.entropy_limit = entropy_limit
        self.scaling_factor = scaling_factor
        self.Forward = Forward(action_size, feature_size, feature_extract, lr)
        self.Inverse = Inverse(action_size, feature_size, feature_extract, lr)
        self.Actor = Actor(action_size, feature_size, feature_extract, lr=0.05)
        self.ExtrinsicCritic = ExtrinsicCritic(action_size, feature_size, feature_extract, lr)
        self.IntrinsicCritic = IntrinsicCritic(action_size, feature_size, self.feature_extract, self.Forward, lr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, mode='exp_range', base_lr=1e-8, max_lr=lr,
        # step_size_up=50)

    def forward(self, state):
        # Expected features input instead of Raw image
        # expected_state = self.Target.forward(state)
        # expected_action = self.Inverse(self.feature_extract.forward(state),expected_state)
        return self.Actor.forward(state)

    def policy(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)

    def loss(self, inputs):
        states, actions, rewards, reward_to_gos, next_states, logp_old = inputs
        curiosity = nn.SmoothL1Loss(reduction='none')(
                self.Forward.forward(states, actions.squeeze()),
                self.feature_extract.forward(next_states))
        curiosity = torch.mean(curiosity,
                               dim=-1)
        dynamics = torch.mean(nn.NLLLoss(reduction='none')(
                input=self.Inverse.forward(self.feature_extract.forward(states),
                                           self.feature_extract.forward(next_states)), target=actions.squeeze()),
                dim=-1)
        value = self.ExtrinsicCritic.forward(states, actions)
        advantages = reward_to_gos - value
        logp = self.policy(states).log_prob(actions)
        ratio = torch.exp(logp - logp_old)
        clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        _loss = -torch.mean(
                torch.min(ratio, clip_ratio) * (advantages +
                                                self.scaling_factor * curiosity + (1 - self.scaling_factor) * dynamics)
        )
        return _loss if _loss is not float('nan') else -1

    def minimize(self, inputs):
        loss = self.loss(inputs)
        self.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), 1)
        nn.utils.clip_grad_value_(self.parameters(), 1)
        self.optimizer.step()
        # self.scheduler.step()
        models_to_update = [self.Forward, self.Inverse, self.IntrinsicCritic, self.ExtrinsicCritic]
        for model in models_to_update:
            model.minimize(inputs)

    def cuda(self, device=None):
        super(CuriosityAgent, self).cuda()
        models_to_update = [self.feature_extract, self.Forward, self.Inverse, self.ExtrinsicCritic,
                            self.IntrinsicCritic]
        for model in models_to_update:
            model.cuda(device)
            model.optimizer = optim.Adam(model.parameters(), self.lr)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, mode='exp_range', base_lr=1e-8, max_lr=self.lr,
        #                                                    step_size_up=50)
        return self._apply(lambda t: t.cuda(device))



class Model:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")