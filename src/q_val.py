import numpy as np
import torch as th
import torch.nn as nn


class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """

    def _init_(self, input_shape, n_actions):
        super(DQNSolver, self)._init_()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Move the entire model to GPU if available
        if th.cuda.is_available():
            self.conv = self.conv.cuda()

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # Create a temporary input tensor on the same device as self.conv
        device = next(self.parameters()).device
        o = self.conv(th.zeros(1, *shape, device=device))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


def get_q_values(ddqn_q_net, observation_space, device, obs):
    assert (
        obs.shape == observation_space
    ), f"Invalid observation shape: {obs.shape}. Expected: {observation_space}"

    # Move observation tensor to the specified device
    obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).to(device)

    # Move model to the same device as the observation tensor
    ddqn_q_net = ddqn_q_net.to(device)

    with th.no_grad():
        q_values = ddqn_q_net(obs_tensor)

    # Move q_values tensor back to CPU before converting to numpy array
    q_values = q_values.cpu().numpy()

    assert isinstance(
        q_values, np.ndarray
    ), "The returned q_values is not a numpy array"
    assert q_values.shape == (
        1,
        n_actions,
    ), f"Wrong shape: (1, {n_actions}) was expected but got {q_values.shape}"

    return q_values


# Example usage:
# Assuming ddqn_q_net is an instance of DQNSolver and observation_space is a tuple representing the shape of observations
# obs = np.random.randn(*observation_space)
# device = th.device("cuda" if th.cuda.is_available() else "cpu")
# q_values = get_q_values(ddqn_q_net, observation_space, device, obs)

# Assuming ddqn_q_net is an instance of DQNSolver and observation_space is a tuple representing the shape of observations
obs_shape = (3, 84, 84)  # Example observation shape
n_actions = 5  # Example number of actions

ddqn_q_net = DQNSolver(obs_shape, n_actions)  # Create an instance of DQNSolver
observation_space = obs_shape  # Example observation space
device = th.device("cuda" if th.cuda.is_available() else "cpu")  # Determine device

# Generate a random observation
obs = np.random.randn(*observation_space)

# Get Q-values
q_values = get_q_values(ddqn_q_net, observation_space, device, obs)

print("Q-values:", q_values)
