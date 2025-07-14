import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class PendulumGym(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None):
        # Initialize instance variables
        self.min_max_velocity = 2 * np.pi
        self.min_max_torque = 3
        self.dt = 0.05
        self.g = 9.8
        self.m = 1/3
        self.l = 3/2

        # Start state (angle, angular velocity)
        self.start_state = (np.pi, 0.0)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -self.min_max_velocity]),
            high=np.array([np.pi, self.min_max_velocity]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)  # 0: left, 1: none, 2: right

        # Rendering setup
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None

        # Current state
        self.state = None

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    @staticmethod
    def angle_normalize(theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "start_state" in options:
            self.state = options["start_state"]
        else:
            self.state = self.start_state

        return self._get_obs(), {}

    def step(self, action):
        theta, theta_dot = self.state

        # Convert discrete action to torque
        if action == 0:  # accelerate_left
            torque = -self.min_max_torque
        elif action == 1:  # dont_accelerate
            torque = 0
        elif action == 2:  # accelerate_right
            torque = self.min_max_torque
        else:
            raise ValueError("Wrong action value")

        # Calculate new state
        new_theta_dot = theta_dot + (3 * self.g / (2 * self.l) * np.sin(theta) + 3.0 / (self.m * self.l**2) * torque) * self.dt

        new_theta = theta + new_theta_dot * self.dt

        # Check velocity bounds
        if new_theta_dot <= -self.min_max_velocity or new_theta_dot >= self.min_max_velocity:
            self.state = self.start_state
        else:
            self.state = (self.angle_normalize(new_theta), new_theta_dot)

        # Calculate reward (negative cost)
        reward = -1 * self.angle_normalize(new_theta)**2

        return self._get_obs(), reward, False, False, {}

    def render(self):
        self._render_frame()

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_dim, self.screen_dim))
        canvas.fill((255, 255, 255))

        # Draw pendulum
        center = (self.screen_dim // 2, self.screen_dim // 2)
        length = 200  # pixels
        theta = self.state[0] + np.pi/2  # Adjust angle for visualization

        end_point = (
            center[0] + length * np.cos(theta),
            center[1] + length * np.sin(theta)
        )

        # Draw pivot point
        pygame.draw.circle(canvas, (0, 0, 0), center, 16)

        # Draw rod
        pygame.draw.line(canvas, (0, 255, 0), center, end_point, 8)

        # Draw bob
        pygame.draw.circle(canvas, (0, 255, 0), (int(end_point[0]), int(end_point[1])), 8)

        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


class EnvironmentPendulum:
    def __init__(self, render_mode=None):
        # initialize gym environment
        self.render_mode = render_mode
        self.gym_env = PendulumGym(render_mode)

        self.action_dict = {
            'accelerate_left': 0,
            'dont_accelerate': 1,
            'accelerate_right': 2,
        }

    def env_start(self, seed):
        # return initial state
        return self.gym_env.reset(seed=seed)[0]

    def env_step(self, state, action):
        if not pd.isnull(self.render_mode):
            self.gym_env.render()
        new_state, reward, terminal, truncated, _ = self.gym_env.step(self.action_dict[action])
        return reward, new_state, (terminal or truncated)

    def env_end(self):
        self.gym_env.close()
