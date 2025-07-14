import torch


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNetwork, self).__init__()
        self.linear_1 = torch.nn.Linear(state_dim, action_dim)

    def forward(self, x):
        return self.linear_1(x)

    def get_values(self, state):
        return self.forward(state)

    def get_estimate(self, state, action):
        return self.forward(state)[range(state.shape[0]), action]


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.linear_1 = torch.nn.Linear(state_dim, action_dim)

    def forward(self, x):
        return torch.nn.functional.softmax(self.linear_1(x), dim=1)

    def sample_action(self, state):
        action_probs = self.forward(state)
        action_prob_dist = torch.distributions.Categorical(action_probs)
        return action_prob_dist.sample()

    def sample_action_with_log_prob(self, state):
        action_probs = self.forward(state)
        action_prob_dist = torch.distributions.Categorical(action_probs)
        action = action_prob_dist.sample()
        log_prob = action_prob_dist.log_prob(action)
        return action, log_prob
