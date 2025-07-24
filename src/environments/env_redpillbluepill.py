import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from collections import deque


class RedPillBluePillGym(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None, dist_2_prob=0.5, history_length=100):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Original environment parameters
        self.dist_1 = {'mean': -0.7, 'stdev': 0.05}
        self.dist_2a = {'mean': -1, 'stdev': 0.05}
        self.dist_2b = {'mean': -0.2, 'stdev': 0.05}
        self.dist_2_prob = dist_2_prob

        # Gymnasium spaces
        self.action_space = spaces.Discrete(2)  # 0: red pill, 1: blue pill
        self.observation_space = spaces.Discrete(2)  # 0: red world, 1: blue world

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_width = 800
        self.window_height = self.window_width // 4
        self.history = deque(maxlen=history_length)

        self.current_state = None

    def _get_obs(self):
        return self.current_state

    def _get_info(self):
        return {'state': self.current_state}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state
        self.current_state = np.random.choice([0, 1])
        self.history.clear()
        self.history.append(self.current_state)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if action == 0:
            next_state = 0
        elif action == 1:
            next_state = 1

        if self.current_state == 0:
            reward = np.random.normal(loc=self.dist_1['mean'], scale=self.dist_1['stdev'])
        elif self.current_state == 1:
            dist = np.random.choice(['dist2a', 'dist2b'], p=[self.dist_2_prob, 1 - self.dist_2_prob])
            if dist == 'dist2a':
                reward = np.random.normal(loc=self.dist_2a['mean'], scale=self.dist_2a['stdev'])
            elif dist == 'dist2b':
                reward = np.random.normal(loc=self.dist_2b['mean'], scale=self.dist_2b['stdev'])

        reward = min(0, reward)
        self.current_state = next_state

        # Update history for rendering
        self.history.append(self.current_state)

        # Gymnasium requirements
        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.font = pygame.font.Font(None, 36)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # Calculate percentage of time in red world
        red_percent = sum(1 for x in self.history if x == 0) / len(self.history) if self.history else 0
        blue_percent = 1 - red_percent

        # Draw background bar
        bar_height = self.window_height//2
        bar_y = self.window_height//4

        # Draw red portion
        red_width = int(self.window_width * red_percent)
        pygame.draw.rect(canvas, (255, 0, 0), (0, bar_y, red_width, bar_height))

        # Draw blue portion
        blue_width = self.window_width - red_width
        pygame.draw.rect(canvas, (0, 0, 255), (red_width, bar_y, blue_width, bar_height))

        # Add percentages text
        red_text = self.font.render(f"Red World: {red_percent:.1%}", True, (0, 0, 0))
        blue_text = self.font.render(f"Blue World: {blue_percent:.1%}", True, (0, 0, 0))

        # Position text with padding from edges
        canvas.blit(red_text, (20, 10))
        canvas.blit(blue_text, (self.window_width - blue_text.get_width() - 20, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


class EnvironmentRedPillBluePill(gym.Env):
    def __init__(self, render_mode=None):
        # initialize gym environment
        self.render_mode = render_mode
        self.gym_env = RedPillBluePillGym(render_mode)

        self.state_dict = {
            0: 'redworld',
            1: 'blueworld',
        }
        self.action_dict = {
            'red_pill': 0,
            'blue_pill': 1,
        }

    def env_start(self, seed):
        # return initial state
        return self.state_dict[self.gym_env.reset(seed=seed)[0]]

    def env_step(self, state, action):
        if not pd.isnull(self.render_mode):
            self.gym_env.render()
        new_state, reward, terminal, truncated, _ = self.gym_env.step(self.action_dict[action])
        return reward, self.state_dict[new_state], (terminal or truncated)

    def env_end(self):
        self.gym_env.close()
