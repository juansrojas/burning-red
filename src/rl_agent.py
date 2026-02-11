import importlib.util
import pandas as pd
import numpy as np
import torch
import copy
import sys

from src.rl_losses import RLLosses
from src.replay_buffer import ReplayBuffer


class RLAgent:
    def __init__(self, agent_type, states, actions, policy=None, avg_reward_method=None, initial_avg_reward=0.0,
                 action_type='discrete', action_selection_rule='epsilon_greedy', policy_type='tabular',
                 policy_softmax_tau=1.0, policy_update_type='stochastic_gradient_descent', policy_loss=None,
                 value_type='tabular', value_network=None, value_update_type='stochastic_gradient_descent',
                 value_loss=None, beta_m_adam=0.9, beta_v_adam=0.999, epsilon_adam=1e-8, pytorch_device='cpu',
                 use_cvar=False, var_quantile=0.1, initial_var_reward=0.0):

        # initialize instance variables
        self.agent_type = agent_type

        self.states = states
        self.actions = actions
        self.action_type = action_type

        self.value_type = value_type
        self.policy_type = policy_type

        self.pytorch_device = pytorch_device

        self.losses = RLLosses()

        # average reward
        self.avg_reward_method = avg_reward_method
        self.avg_reward = initial_avg_reward

        # CVaR
        self.use_cvar = use_cvar
        self.var_quantile = var_quantile
        self.var_reward = initial_var_reward

        # function approximation
        self.beta_m_adam = beta_m_adam
        self.beta_v_adam = beta_v_adam
        self.epsilon_adam = epsilon_adam

        #################################
        # state and state-action values
        #################################
        # tabular case
        if self.value_type == 'tabular':
            self.values = {}
            if self.agent_type in ['td', 'ac']:
                for state in self.states:
                    self.values[state] = 0

            elif self.agent_type in ['q_learning']:
                for state in self.states:
                    self.values[state] = {}
                    for action in self.actions:
                        self.values[state][action] = 0

        # function approximation
        else:
            # value network
            self.value_network_config = value_network
            self.value_update_type = value_update_type
            self.neural_network_value = None
            self.value_loss = value_loss

        ################
        # agent policy
        ################
        self.policy = policy
        self.action_selection_rule = action_selection_rule

        # softmax
        if self.action_selection_rule == 'softmax':
            self.policy_softmax_tau = policy_softmax_tau

        # policy network
        if self.policy_type == 'nn_pytorch':
            self.policy_network_config = policy
            self.policy_update_type = policy_update_type
            self.neural_network_policy = None
            self.policy_loss = policy_loss

    @staticmethod
    def load_network_from_file(file_path, class_name):
        spec = importlib.util.spec_from_file_location("module_name", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module_name"] = module
        spec.loader.exec_module(module)
        network_class = getattr(module, class_name)
        return network_class

    def initialize_network(self, weight_type, network_config, optimizer_type, beta_m, beta_v, epsilon_adam):
        neural_network = {}

        if weight_type == 'value':
            if self.action_type == 'discrete':
                if self.agent_type in ['td', 'ac']:
                    action_dim = 1
                else:
                    action_dim = len(self.actions)
            elif self.action_type == 'continuous':
                action_dim = len(self.actions['high'])

            value_network_args = {
                'state_dim': len(self.states),
                'action_dim': action_dim,
            }

            for param in network_config.keys():
                if param not in ['file_path']:
                    value_network_args[param] = network_config[param]

            neural_network['network'] = self.load_network_from_file(network_config['file_path'], 'ValueNetwork')(**value_network_args)

        elif weight_type == 'policy':
            if self.action_type == 'discrete':
                action_dim = len(self.actions)
            elif self.action_type == 'continuous':
                action_dim = len(self.actions['high'])

            policy_network_args = {
                'state_dim': len(self.states),
                'action_dim': action_dim,
            }

            for param in network_config.keys():
                if param not in ['file_path']:
                    policy_network_args[param] = network_config[param]

            if self.action_type == 'continuous':
                policy_network_args['max_action'] = torch.FloatTensor(self.actions['high']).to(self.pytorch_device)

            neural_network['network'] = self.load_network_from_file(network_config['file_path'], 'PolicyNetwork')(**policy_network_args)

        neural_network['network'].to(self.pytorch_device)

        # set optimizer
        if optimizer_type == 'stochastic_gradient_descent':
            neural_network['optimizer'] = torch.optim.SGD(neural_network['network'].parameters())

        elif optimizer_type == 'adam':
            neural_network['optimizer'] = torch.optim.Adam(neural_network['network'].parameters(), betas=(beta_m, beta_v), eps=epsilon_adam)

        return neural_network

    def load_pytorch_networks(self):
        if self.value_type == 'nn_pytorch':
            self.neural_network_value = self.initialize_network(weight_type='value',
                                                                network_config=self.value_network_config['initialize'],
                                                                optimizer_type=self.value_update_type,
                                                                beta_m=self.beta_m_adam,
                                                                beta_v=self.beta_v_adam,
                                                                epsilon_adam=self.epsilon_adam,
                                                                )

            if 'load_weights' in self.value_network_config.keys():
                self.neural_network_value['network'].load_state_dict(torch.load(self.value_network_config['load_weights']['file_path']))

        if self.policy_type == 'nn_pytorch':
            self.neural_network_policy = self.initialize_network(weight_type='policy',
                                                                 network_config=self.policy_network_config['initialize'],
                                                                 optimizer_type=self.policy_update_type,
                                                                 beta_m=self.beta_m_adam,
                                                                 beta_v=self.beta_v_adam,
                                                                 epsilon_adam=self.epsilon_adam,
                                                                 )

            if 'load_weights' in self.policy_network_config.keys():
                self.neural_network_policy['network'].load_state_dict(torch.load(self.policy_network_config['load_weights']['file_path']))

        return

    def get_max_value(self, values):
        if self.value_type == 'tabular':
            return max(values.values())
        else:
            values_array = torch.stack(list(values.values()), dim=1).to(self.pytorch_device)
            return values_array.max(dim=1, keepdim=True)[0]

    def get_argmax_actions(self, values, return_argmax_mask=False):
        if self.value_type == 'tabular':
            max_val = self.get_max_value(values)
            return [k for k, v in values.items() if v == max_val]
        else:
            values_array = torch.stack(list(values.values()), dim=1).to(self.pytorch_device)
            argmax_mask = values_array == self.get_max_value(values)
            if return_argmax_mask:
                return argmax_mask
            else:
                argmax_actions = []
                for row in argmax_mask:
                    actions = [self.actions[i.item()] for i in row.nonzero(as_tuple=True)[0]]
                    argmax_actions.append(actions)
            return argmax_actions

    def argmax(self, values):
        # get argmax of values, breaking ties arbitrarily
        if self.value_type == 'tabular':
            return np.random.choice(self.get_argmax_actions(values))
        else:
            values_array = torch.stack(list(values.values()), dim=1).to(self.pytorch_device)
            mask = values_array == self.get_max_value(values)
            tie_breaker = torch.rand_like(values_array) * mask.float()  # generate random noise for tiebreaking
            return [self.actions[i.item()] for i in tie_breaker.argmax(dim=1)]

    def argmax_directly_from_network_values(self, network_values):
        # optimized function to get argmax from raw network values, breaking ties arbitrarily
        mask = network_values == network_values.max(dim=1, keepdim=True).values
        tie_breaker = torch.rand_like(network_values) * mask.float()
        return tie_breaker.argmax(dim=1)

    def get_tabular_value(self, state, action=None):
        if self.agent_type in ['td', 'ac']:
            return self.values[state]
        elif self.agent_type in ['q_learning']:
            return self.values[state][action]

    def get_action_index(self, state, action):
        if self.agent_type not in ['td', 'ac'] and not isinstance(action, torch.Tensor):
            action = np.array([action])  # use numpy array to handle strings

        if self.value_type == 'tabular':
            if self.agent_type in ['td', 'ac']:
                return 0
            else:
                return self.actions.index(action)

        elif self.value_type == 'nn_pytorch':
            if self.agent_type in ['td', 'ac']:
                action_index = torch.zeros(state.shape[0], dtype=torch.long).to(self.pytorch_device)
            else:
                action_index = torch.tensor([self.actions.index(a) for a in action], dtype=torch.long).to(self.pytorch_device)
            return action_index

    def query_discrete_action_values_from_network(self, state, action, values):
        action_index = self.get_action_index(state, action)
        return values[range(state.shape[0]), action_index]

    def get_value(self, state, action=None, use_target_network=False, get_all_actions=False, get_raw_discrete_action_network_values=False):
        # returns the state or state-action value

        # tabular case
        if self.value_type == 'tabular':
            if get_all_actions:
                values = {}
                for a in self.actions:
                    values[a] = self.get_tabular_value(state, a)
                return values
            else:
                return self.get_tabular_value(state, action)

        # value-function approximation
        else:
            if use_target_network:
                network = self.neural_network_target_value
            else:
                network = self.neural_network_value

            if self.action_type == 'discrete':
                network_values = network['network'].get_values(state)
                if get_raw_discrete_action_network_values:
                    return network_values
                if get_all_actions:
                    values = {}
                    for a in self.actions:
                        values[a] = self.query_discrete_action_values_from_network(state, a, network_values)
                    return values
                else:
                    return self.query_discrete_action_values_from_network(state, action, network_values)

            elif self.action_type == 'continuous':
                return network['network'].get_value(state, action)

    def get_target_value(self, state, action=None, get_all_actions=False, get_raw_discrete_action_network_values=False):
        # returns the target state or state-action value
        return self.get_value(state, action, use_target_network=True, get_all_actions=get_all_actions, get_raw_discrete_action_network_values=get_raw_discrete_action_network_values)

    def get_softmax_probabilities(self, state):
        # get the softmax policy conditional on a given state
        if self.policy_type == 'nn_pytorch':
            softmax_probs = self.neural_network_policy['network'](state)
            return {action: softmax_probs[:, i] for i, action in enumerate(self.actions)}
        else:
            values = self.get_value(state, get_all_actions=True)
            if self.value_type == 'tabular':
                preferences = torch.FloatTensor([values[a] / self.policy_softmax_tau for a in self.actions]).to(self.pytorch_device)
                softmax = torch.nn.functional.softmax(preferences, dim=0)
                return {a: softmax[i] for i, a in enumerate(self.actions)}
            else:
                value_matrix = torch.stack([values[a] for a in self.actions], dim=1)
                preferences = value_matrix / self.policy_softmax_tau
                softmax = torch.nn.functional.softmax(preferences, dim=1)
                return {a: softmax[:, i] for i, a in enumerate(self.actions)}

    def get_epsilon_greedy_probabilities(self, state, epsilon):
        # get the epsilon-greedy policy conditional on a given state
        values = self.get_value(state, get_all_actions=True)
        if self.value_type == 'tabular':
            argmax_actions = self.get_argmax_actions(values)
            action_probs = {}
            for a in self.actions:
                if a in argmax_actions:
                    action_probs[a] = torch.FloatTensor([(1 - epsilon) / len(argmax_actions) + (epsilon / len(self.actions))]).to(self.pytorch_device)
                else:
                    action_probs[a] = torch.FloatTensor([epsilon / len(self.actions)]).to(self.pytorch_device)
        else:
            # action prob = exploration + argmax bonus
            argmax_action_mask = self.get_argmax_actions(values, return_argmax_mask=True)
            probs = (torch.ones(state.shape[0], len(self.actions)).to(self.pytorch_device) * (epsilon / len(self.actions)) +
                     argmax_action_mask * (1 - epsilon) / argmax_action_mask.sum(dim=1, keepdim=True))
            action_probs = {action: probs[:, i] for i, action in enumerate(self.actions)}

        return action_probs

    def get_discrete_action_probs_from_user_defined_policy(self, policy, state):
        if self.policy_type == 'tabular':
            return {action: torch.FloatTensor([policy[state][action]]).to(self.pytorch_device) for action in self.actions}

        else:
            action_probs =  policy(state)
            return {action: action_probs[:, i] for i, action in enumerate(self.actions)}

    def get_random_discrete_action_probabilities(self, state):
        if self.policy_type == 'tabular':
            return {action: torch.FloatTensor([1 / len(self.actions)]).to(self.pytorch_device) for action in self.actions}
        else:
            probs = torch.ones(state.shape[0], len(self.actions)).to(self.pytorch_device) * 1 / len(self.actions)
            return {action: probs[:, i] for i, action in enumerate(self.actions)}

    def get_discrete_action_probs(self, state, epsilon):
        if self.action_selection_rule == 'softmax':
            return self.get_softmax_probabilities(state)

        elif self.action_selection_rule == 'epsilon_greedy':
            return self.get_epsilon_greedy_probabilities(state, epsilon)

        elif self.action_selection_rule == 'user_defined':
            return self.get_discrete_action_probs_from_user_defined_policy(self.policy, state)

        elif self.action_selection_rule == 'random':
            return self.get_random_discrete_action_probabilities(state)

    def get_discrete_action_log_prob(self, action, action_probs):
        action_prob_dist = torch.distributions.Categorical(action_probs)
        return action_prob_dist.log_prob(torch.tensor(self.actions.index(action)).to(self.pytorch_device))

    def choose_action_from_policy(self, state, epsilon=None, get_action_log_prob=False):
        # chooses action based on policy
        if self.policy_type == 'tabular':
            action_probs = {a: p.item() for a, p in self.get_discrete_action_probs(state, epsilon).items()}

            # make sure probs sum to exactly 1.0
            total = sum(action_probs.values())
            action_probs = {a: p / total for a, p in action_probs.items()}

            action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
            if get_action_log_prob:
                return action, self.get_discrete_action_log_prob(action, torch.tensor(list(action_probs.values())).to(self.pytorch_device))
            else:
                return action, None

        elif self.policy_type == 'nn_pytorch':
            if get_action_log_prob:
                action, log_prob = self.neural_network_policy['network'].sample_action_with_log_prob(state)
            else:
                action = self.neural_network_policy['network'].sample_action(state)
                log_prob = None

            if self.action_type == 'discrete':
                action = [self.actions[a.item()] for a in action]

            return action, log_prob

    def output_action(self, action):
        # format action to be sent back to the environment
        if isinstance(action, torch.Tensor):
            return action.cpu().data.numpy().flatten()
        elif self.policy_type == 'nn_pytorch' and self.action_type == 'discrete':
            return action[0]
        else:
            return action

    def value_tabular_update(self, state, action, step_size, estimate, target):
        # update tabular values
        if self.agent_type in ['td', 'ac']:
            td_error = target - estimate
            self.values[state] += step_size['value'] * td_error

        elif self.agent_type in ['q_learning']:
            td_error = target - estimate
            self.values[state][action] += step_size['value'] * td_error

        return

    def value_network_update(self, target, estimate, step_size):
        # value function approximation update:

        # calculate loss on neural network output and target
        if self.value_loss == 'mse_loss':
            loss = self.losses.mse_loss(
                estimate=estimate.reshape(-1, 1),
                target=target.reshape(-1, 1),
            )

        # set step size for learning
        self.neural_network_value['optimizer'].param_groups[0]['lr'] = step_size['value']

        # update network params
        self.neural_network_value['optimizer'].zero_grad()
        loss.backward()
        self.neural_network_value['optimizer'].step()

        return loss.item()

    def policy_network_update(self, state, action, step_size, target, estimate):
        # parameterized policy update:

        # calculate loss
        if self.policy_loss == 'ac_policy_loss':
            action_probs = self.neural_network_policy['network'](state)
            log_prob = self.get_discrete_action_log_prob(action, action_probs)
            td_error = (target - estimate).detach()
            loss = self.losses.ac_policy_loss(
                log_prob=log_prob,
                delta=td_error,
            )

        # set step size for learning
        self.neural_network_policy['optimizer'].param_groups[0]['lr'] = step_size['policy']

        # update network params
        self.neural_network_policy['optimizer'].zero_grad()
        loss.backward()
        self.neural_network_policy['optimizer'].step()

        return loss.item()

    def avg_reward_update(self, reward, target, estimate, step_size):
        # update average-reward
        if pd.isnull(self.avg_reward_method):
            return

        elif self.avg_reward_method == 'empirical':
            self.avg_reward += step_size['avg_reward'] * (reward - self.avg_reward)

        elif self.avg_reward_method == 'differential':
            self.avg_reward += step_size['avg_reward'] * (target - estimate)

    def var_update(self, raw_reward, target, estimate, step_size):
        # update VAR estimate
        if raw_reward >= self.var_reward:
            self.var_reward = self.var_reward + step_size['var'] * ((target - estimate) + self.avg_reward - self.var_reward)
        else:
            self.var_reward = self.var_reward + step_size['var'] * ((self.var_quantile / (self.var_quantile - 1)) * (target - estimate) + self.avg_reward - self.var_reward)

    def preprocess_state(self, state):
        if isinstance(state, np.ndarray):
            return torch.FloatTensor(state).to(self.pytorch_device).unsqueeze(0)
        else:
            return state

    def preprocess_action(self, action):
        if isinstance(action, np.ndarray):
            return torch.FloatTensor(action).to(self.pytorch_device).unsqueeze(0)
        else:
            return action

    def preprocess_reward(self, raw_reward):
        # get CVaR reward if needed
        if self.use_cvar:
            reward = self.var_reward - (1 / self.var_quantile) * max(0, self.var_reward - raw_reward)
        else:
            reward = raw_reward
        return reward

    def agent_start(self, init_state, epsilon):
        # start episode and choose initial action based on initial state
        with torch.no_grad():
            action, _ = self.choose_action_from_policy(state=self.preprocess_state(init_state), epsilon=epsilon)
            return self.output_action(action)

    def agent_step(self, last_state, last_action, state, raw_reward, terminal, epsilon, step_size, discount, get_action_log_prob=False):
        with torch.no_grad():
            reward = self.preprocess_reward(raw_reward)

            # select next action
            action, action_log_prob = self.choose_action_from_policy(
                state=self.preprocess_state(state),
                get_action_log_prob=get_action_log_prob,
                epsilon=epsilon,
            )

            # get target
            target = self.get_target(
                state=self.preprocess_state(state),
                reward=reward,
                terminal=terminal,
                discount=discount,
            )

        # get estimate
        estimate = self.get_estimate(
            last_state=self.preprocess_state(last_state),
            last_action=self.preprocess_action(last_action),
        )

        # perform updates and return action
        if self.value_type == 'tabular':
            # tabular:

            # update average reward
            self.avg_reward_update(reward, target, estimate, step_size)

            # update VAR estimate
            if self.use_cvar:
                self.var_update(raw_reward, target, estimate, step_size)

            # update values
            self.value_tabular_update(state=last_state,
                                      action=last_action,
                                      step_size=step_size,
                                      estimate=estimate,
                                      target=target)

            value_loss = None
            policy_loss = None

        else:
            # function approximation:

            # update average reward
            if not pd.isnull(self.avg_reward_method):
                with torch.no_grad():
                    self.avg_reward_update(
                        reward=reward,
                        target=target.item() if 'differential' in self.avg_reward_method else None,
                        estimate=estimate.item() if 'differential' in self.avg_reward_method else None,
                        step_size=step_size,
                    )

                    # update VAR estimate
                    if self.use_cvar:
                        self.var_update(raw_reward, target.item(), estimate.item(), step_size)

            # update value function
            if self.value_type == 'nn_pytorch':
                value_loss = self.value_network_update(target=target,
                                                       estimate=estimate,
                                                       step_size=step_size,
                                                       )
            else:
                value_loss = None

            # update policy
            if self.policy_type == 'nn_pytorch':
                policy_loss = self.policy_network_update(state=self.preprocess_state(last_state),
                                                         action=self.preprocess_action(last_action),
                                                         step_size=step_size,
                                                         target=target,
                                                         estimate=estimate,
                                                         )
            else:
                policy_loss = None

        return self.output_action(action), value_loss, policy_loss

    def get_estimate(self, last_state, last_action):
        # Get state or state-action value estimate
        if self.value_type == 'tabular':
            return self.get_value(last_state, last_action)

        elif self.value_type == 'nn_pytorch':
            if self.action_type == 'discrete':
                action_index = self.get_action_index(last_state, last_action)
                return self.neural_network_value['network'].get_estimate(last_state, action_index)
            elif self.action_type == 'continuous':
                return self.neural_network_value['network'].get_estimate(last_state, last_action)

    def get_target(self, state, reward, terminal, discount):
        # calculate rl target
        if self.agent_type in ['td', 'ac']:
            # get td target (which is also used in simple actor-critic)
            return self.get_td_target(state, reward, terminal, discount)

        elif self.agent_type == 'q_learning':
            # get Q-learning target
            return self.get_q_learning_target(state, reward, terminal, discount)

    def get_td_target(self, state, reward, terminal, discount):
        return reward - self.avg_reward + (1 - terminal) * discount * self.get_target_value(state, action=None)

    def get_q_learning_target(self, state, reward, terminal, discount):
        return reward - self.avg_reward + (1 - terminal) * discount * self.get_max_value(self.get_target_value(state, get_all_actions=True))
