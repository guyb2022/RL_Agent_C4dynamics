from datapoint import Datapoint
from rigidbody import Rigidbody
import numpy as np
import pandas as pd
import gym

class C4dynamicsEnv(Datapoint, Datapoint):
    # I used here inharitence to get all Datapoint/Datapoint features and function
    # also I passed the objects to the constaractor to avopid sending tons f parameters.
    # I didnt check if it is valid, I took the guess it could be done
    def __init__(self, datapoint_obj, rigidbody_obj, num_of_actions, agent_name):
        self.target_x = datapoint_obj.x
        self.missile_x = rigidbody_obj.x
        self.number_of_action = num_of_actions
        self.agent_name = agent_name
        """
        Do the same for all target/missile parameters
        """

    def _is_done(self):
        """
        Check if current_step is within bounds of the simulation
        """
        if self.current_step == len(self.data) - 1:
            return True
        return False

    def _get_observation(self):
        """
        Return the current observation.
        This function returns an observation for the agent that includes
        all the env parameters
        """
        current_row = self.data.iloc[self.current_step]
        observation = [current_row['x_target'], current_row['y_target'], ...]
        return observation

    def _get_observation_space(self):
        min_target_x, max_target_x =  np.amin(self.data['x_target']), np.amax(self.data['x_target'])
        min_target_y, max_target_y =  np.amin(self.data['y_target']), np.amax(self.data['y_target'])
        #...
        #...
        return gym.spaces.Box(low=np.array([min_target_x, min_target_y,....]),
                              high=np.array([max_target_x, max_target_y,...]))


    def _get_action_space(self):
        """
        Returns the action space for the environment.
        """
        return gym.spaces.Discrete(self.number_of_action)

    def _take_action(self,action):
        if action == 0:
            self.action_1()
        elif action == 1:
            self.action_2()
        #...
        #...
        elif action == 10:
            self.action_10()

        else:
            # some default or correction action
            self.action_11()

    def _get_reward(self):
        """
        Calculate the reward based on the distance from the target
        """
        if "missile hit target":
           return 10
        else:
            return -1

    def step(self, action):
        """
           Take the given action and return the next observation, reward, and done status.
           This function takes the given action and updates the position and balance of the
           agent accordingly. It then increments the current step and sets the done flag to
           True if the current step is equal to or greater than the total number of steps.
        """
        # Get the current observation
        observation = self._get_observation()
        self._take_action(action)
        reward = self._get_reward()
        # current step is for the dataFrame
        self.current_step += 1
        done = self._is_done()
        return observation, reward, done, {}

    def reset(self):
        """Reset the environment variables and return the initial observation."""
        return self._get_observation()