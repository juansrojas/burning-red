import numpy as np

import src.tiles as tc

class StateRepresentation:
    def __init__(self, method, settings):
        self.method = method
        if self.method == 'tilecoding':
            self.iht = tc.IHT(settings['iht_size'])
            self.num_tiles = settings['num_tiles']
            self.num_tilings = settings['num_tilings']
            self.state_limits = settings['state_limits']

    def get_state_representation(self, state):
        if self.method == 'tilecoding':
            return self.tilecoding(state)

    def tilecoding(self, state):
        # Takes in a state tuple and returns a numpy array of active tiles.

        scaled_state_components = []
        for i in range(len(state)):
            # scale to the range [0, 1]
            # then multiply that range with self.num_tiles, so it scales from [0, num_tiles]
            scaled_state_components.append(((state[i] - self.state_limits[i][0]) / (self.state_limits[i][1] - self.state_limits[i][0])) * self.num_tiles)

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and scaled_state_components
        tiles = tc.tiles(self.iht, self.num_tilings, scaled_state_components)

        feature_vector = np.zeros((self.iht.size, 1))
        feature_vector[tiles] = 1

        return np.squeeze(feature_vector)
