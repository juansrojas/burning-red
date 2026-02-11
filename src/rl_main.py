import pandas as pd
import numpy as np
import wandb
import copy


class ReinforcementLearning:
    def __init__(self, experiment, agent_class, environment_class, state_representation, track_data, use_wandb, wandb_entity, wandb_project):
        # initialize instance variables
        self.agent = None
        self.environment = None

        # keep track of episode and step counts
        self.step_num = 0
        self.episode_num = 0

        # initialize agent
        self.agent = copy.deepcopy(agent_class)

        # initialize environment
        self.environment = copy.deepcopy(environment_class)

        # initialize state representation function
        self.state_representation = copy.deepcopy(state_representation)

        # set agent step
        self.agent_step = self.agent.agent_step

        # keep track of RL data
        self.track_data = track_data
        self.data = []

        # set up W&B
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.wandb = wandb.init(
                name=experiment,
                entity=wandb_entity,
                project=wandb_project,
                settings=wandb.Settings(
                    console="off",
                    quiet=True,
                    silent=True,
                    disable_code=True,
                    disable_git=True
                ),
            )

    def get_data(self):
        # return RL data
        df = pd.DataFrame(self.data)
        for col in df.columns:
            if isinstance(df[col][0], np.ndarray):
                df[col] = df[col].apply(tuple)
        return df

    def rl_start(self, seed, epsilon=None):
        # return initial state and action to start the episode

        # get initial state
        state = self.environment.env_start(seed=seed)

        # get initial action
        action = self.agent.agent_start(
            state if pd.isnull(self.state_representation) else self.state_representation.get_state_representation(state),
            epsilon=epsilon,
        )

        # update episode and step counts
        self.episode_num += 1
        self.step_num = 0

        # return initial state and action
        return state, action

    def rl_step(self, last_state, last_action, epsilon=None, step_size=None, discount=None):

        self.step_num += 1

        # get reward and new state from environment based on last state and last action taken
        (reward, state, terminal) = self.environment.env_step(last_state, last_action)

        # perform update
        action, value_loss, policy_loss = self.agent_step(
            last_state=last_state if pd.isnull(self.state_representation) else self.state_representation.get_state_representation(last_state),
            last_action=last_action,
            state=state if pd.isnull(self.state_representation) else self.state_representation.get_state_representation(state),
            raw_reward=reward,
            terminal=terminal,
            epsilon=epsilon,
            step_size=step_size,
            discount=discount,
        )

        # track data
        if self.track_data:
            self.data.append({
                'episode': self.episode_num,
                'step': self.step_num,
                'state': last_state,
                'action': last_action,
                'reward': reward,
                'next_state': state,
                'terminal': terminal,
                'value_loss': value_loss,
                'policy_loss': policy_loss,
            })

        if self.use_wandb:
            self.wandb.log({
                'episode': self.episode_num,
                'step': self.step_num,
                # 'state': last_state,
                # 'action': last_action,
                'reward': reward,
                # 'next_state': state,
                # 'terminal': terminal,
                'value_loss': value_loss,
                'policy_loss': policy_loss,
                'average_reward_estimate': self.agent.avg_reward,
            })

        return reward, state, action, terminal

    def rl_end(self):
        # end environment
        self.environment.env_end()

        # end W&B logging
        if self.use_wandb:
            self.wandb.finish()
