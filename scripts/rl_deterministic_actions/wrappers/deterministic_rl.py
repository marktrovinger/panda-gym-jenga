from gymnasium import Wrapper, ObservationWrapper
from gymnasium.spaces import Discrete, Tuple, MultiDiscrete
import numpy as np
from typing import Any

class DeterministicRLObservationWrapper(ObservationWrapper):
    def observation(self, observation: Any) -> np.ndarray:
        completed = super().completed
        return completed

class DeterministicRLWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task = self.env.unwrapped.task
        self.robot = self.env.unwrapped.robot
        self.action_space = MultiDiscrete([3, self.task.num_components])
        self.observation_space = Discrete(self.task.num_components,)
        self.objective_coords = []
        self.object_coords = []
        self.completed = np.zeros(self.task.num_components)
        self._setup()

    def _setup(self):
        observation = self.env.unwrapped._get_obs()
        #object_coords = []
        for i in range(self.task.num_components):
            if i < self.task.num_components - 1:
                # first 3 coords
                if i > 0:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i*3)+3])
                    self.object_coords.append(observation["achieved_goal"][i*3:(i*3)+3])
                else:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i+3)])
                    self.object_coords.append(observation["achieved_goal"][i*3:(i+3)])
            else:
                # last object
                self.objective_coords.append(observation["desired_goal"][(self.task.num_components - 1)*3:])
                self.object_coords.append(observation["achieved_goal"][(self.task.num_components - 1)*3:])
        

    def observation(self, observation: Any) -> np.ndarray:
        obs = observation
        return self.completed
    

    def _robot_action(self, action):
        # move to object
        
        current_position = np.round(self.env.unwrapped._get_obs()["observation"][0:3], decimals=3)
        action, target = action[0], action[1]

        # move to the object
        if action == 0:
            goal_position = self.object_coords[target]
            #goal_position = observation["achieved_goal"][0:3]
            if np.allclose(goal_position, current_position, 0.1):
                self.is_action_completed = True
                return current_position
            else:
                return 4.6 * (goal_position - current_position)
        elif action == 1:
            # move to the goal
            # for the stacking to work, we need to move to a higher z position first
            if not self.higher_z:
                goal_position = self.objective_coords[target] + np.array([0.0, 0.0, 0.09])
                if np.allclose(current_position, goal_position, 0.1):
                    self.higher_z = True
                    return current_position
                else: 
                    return 4.6 * (goal_position - current_position)
            else:
                goal_position = self.objective_coords[target]
                if np.allclose(current_position, goal_position, self.env.task.distance_threshold):
                    self.is_action_completed = True
                    self.completed[target] = 1
                    self.higher_z = False
                    return current_position
                else: 
                    return 4.6 * (goal_position - current_position)
        if action == 2:
            self.robot.block_gripper = True
            gripper_action = np.append(current_position, 0.0)
            #print("Closing fingers")
            self.is_action_completed = True
            return gripper_action
        else:
            self.robot.block_gripper = False
            gripper_action = np.append(current_position, 1.0)
            self.is_action_completed = True
            self.completed[target] = 1
            return gripper_action
        
    def reset(self, seed=42):
        obs, info = super().reset(seed=seed)
        obs = np.zeros(self.task.num_components)
        info = {
                "is_success": False,
                "time_taken": 0
        }
        return obs, info

    def step(self, action):
        """
        """
        self.observation_ = self.env.unwrapped._get_obs()
        reward = -1
        
        goal_action = action[0]
        goal_object = action[1]
        for i in range(100):
            # check to see if we are done with the overall objective
            # TODO: adjust for tuple structure of action space
            # TODO: adjust for change in observation space
            if self.is_action_completed and goal_action == 1:
                observation = self.completed
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                self.robot.reset()
                return observation, reward, terminated, truncated, info
            elif self.is_action_completed and goal_action != 1:
                observation = self.completed
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                #self.robot.reset()
                return observation, reward, terminated, truncated, info
            action = self._robot_action([goal_action, goal_object])
            #dont forget to add the gripper
            if goal_action < 2:
                gripper = 0.0 if self.robot.get_fingers_width() else 1.0
                action = np.append(action, gripper)
            self.robot.set_action(action)
            self.sim.step()