from typing import Optional

import numpy as np
from gymnasium.spaces import Discrete
from panda_gym.pybullet import PyBullet
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym_jenga.envs.robot.kinova import Kinova
from panda_gym_jenga.envs.tasks.jengapickandplace import JengaPickAndPlace
from panda_gym_jenga.envs.tasks.stack3 import JengaStack3
from panda_gym_jenga.envs.tasks.simplejengapickandplace import JengaSimplePickAndPlace 
from panda_gym_jenga.envs.tasks.jenga_tower import JengaTower
from panda_gym_jenga.envs.tasks.jenga_tower3 import JengaTower3
from panda_gym_jenga.envs.tasks.jenga_wall import JengaWall3



class JengaWall3Env(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 1.4, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large"
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaWall3(sim, reward_type=reward_type, object_size=object_size)
        super().__init__(
            self.robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )

class JengaWall3DeterministicEnv(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 0.75, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large",
            deterministic = False
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        #robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaWall3(sim, reward_type=reward_type, object_size=object_size, deterministic=deterministic, distance_threshold=0.07)
        self.higher_z = False
        self.object_counter = 0
        self.is_action_completed = False
        self.object_coords = []
        self.objective_coords = []
        super().__init__(
            robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )
        self._object_map()
        self._objective_map()
        

    def _objective_map(self):
        observation = self._get_obs()
        #object_coords = []
        for i in range(6):
            if i < 5:
                # first 3 coords
                if i > 0:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i*3)+3])
                else:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i+3)])
            else:
                # last object
                self.objective_coords.append(observation["desired_goal"][15:])

    def _object_map(self):
        observation = self._get_obs()
        #object_coords = []
        for i in range(6):
            if i < 5:
                # first 3 coords
                if i > 0:
                    self.object_coords.append(observation["achieved_goal"][i*3:(i*3)+3])
                else:
                    self.object_coords.append(observation["achieved_goal"][i*3:(i+3)])
            else:
                # last object
                self.object_coords.append(observation["achieved_goal"][15:])
    def _robot_action(self, action):
        # move to object
        observation = self._get_obs()
        current_position = np.round(observation["observation"][0:3], decimals=3)

        # move to the object
        if action == 0:
            goal_position = self.object_coords[self.object_counter]
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
                goal_position = self.objective_coords[self.object_counter] + np.array([0.0, 0.0, 0.09])
                if np.allclose(current_position, goal_position, 0.1):
                    self.higher_z = True
                    return current_position
                else: 
                    return 4.6 * (goal_position - current_position)
            else:
                goal_position = self.objective_coords[self.object_counter]
                if np.allclose(current_position, goal_position, self.task.distance_threshold):
                    self.is_action_completed = True
                    self.object_counter += 1
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
            return gripper_action

        

    def step(self, action: np.ndarray):
        # the actions need to be changed in order to work deterministically
        reward = -1
        
        goal_action = action
        for i in range(100):
            # check to see if we are done with the overall objective
            if self.is_action_completed and goal_action == 1:
                observation = self._get_obs()
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
                observation = self._get_obs()
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                #self.robot.reset()
                return observation, reward, terminated, truncated, info
            action = self._robot_action(goal_action)
            #dont forget to add the gripper
            if goal_action < 2:
                gripper = 0.0 if self.robot.get_fingers_width() else 1.0
                action = np.append(action, gripper)
            self.robot.set_action(action)
            self.sim.step()


class JengaTowerEnv(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 1.4, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large"
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaTower(sim, reward_type=reward_type, object_size=object_size)
        super().__init__(
            self.robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )

class JengaTower3Env(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 1.4, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large"
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaTower3(sim, reward_type=reward_type, object_size=object_size)
        super().__init__(
            self.robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )

class JengaTowerDeterministicEnv(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 0.75, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large",
            deterministic = False
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        #robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        robot = Kinova(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaTower(sim, reward_type=reward_type, object_size=object_size, deterministic=deterministic, distance_threshold=0.1)
        object_goals = [False, False, False, False]
        objective_goals = [False, False, False, False]
        self.object_counter = 0
        self.is_action_completed = False
        self.object_coords = []
        self.objective_coords = []
        super().__init__(
            robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )
        self._object_map()
        self._objective_map()
        

    def _objective_map(self):
        observation = self._get_obs()
        #object_coords = []
        for i in range(4):
            if i < 3:
                # first 3 coords
                if i > 0:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i*3)+3])
                else:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i+3)])
            else:
                # last object
                self.objective_coords.append(observation["desired_goal"][9:])

    def _object_map(self):
        observation = self._get_obs()
        #object_coords = []
        for i in range(4):
            if i < 3:
                # first 3 coords
                if i > 0:
                    self.object_coords.append(observation["achieved_goal"][i*3:(i*3)+3])
                else:
                    self.object_coords.append(observation["achieved_goal"][i*3:(i+3)])
            else:
                # last object
                self.object_coords.append(observation["achieved_goal"][9:])
    def _robot_action(self, action):
        # move to object
        observation = self._get_obs()
        current_position = np.round(observation["observation"][0:3], decimals=3)

        # move to the object
        if action == 0:
            goal_position = self.object_coords[self.object_counter]
            #goal_position = observation["achieved_goal"][0:3]
            if np.allclose(goal_position, current_position, 0.1):
                self.is_action_completed = True
                return current_position
            else:
                return 4.6 * (goal_position - current_position)
        elif action == 1:
            # move to the goal
            goal_position = self.objective_coords[self.object_counter]
            #goal_position = observation["desired_goal"][0:3]
            if np.allclose(current_position, goal_position, self.task.distance_threshold):
                self.is_action_completed = True
                self.object_counter += 1

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
            return gripper_action

        

    def step(self, action: np.ndarray):
        # the actions need to be changed in order to work deterministically
        reward = -1
        
        goal_action = action
        for i in range(100):
            # check to see if we are done with the overall objective
            if self.is_action_completed and goal_action == 1:
                observation = self._get_obs()
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
                observation = self._get_obs()
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                #self.robot.reset()
                return observation, reward, terminated, truncated, info
            action = self._robot_action(goal_action)
            #dont forget to add the gripper
            if goal_action < 2:
                gripper = 0.0 if self.robot.get_fingers_width() else 1.0
                action = np.append(action, gripper)
            self.robot.set_action(action)
            self.sim.step()

class JengaTower3DeterministicEnv(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 0.75, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large",
            deterministic = False
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        #robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaTower3(sim, reward_type=reward_type, object_size=object_size, deterministic=deterministic, distance_threshold=0.07)
        self.higher_z = False
        self.object_counter = 0
        self.is_action_completed = False
        self.object_coords = []
        self.objective_coords = []
        super().__init__(
            robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )
        self._object_map()
        self._objective_map()
        

    def _objective_map(self):
        observation = self._get_obs()
        #object_coords = []
        for i in range(6):
            if i < 5:
                # first 3 coords
                if i > 0:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i*3)+3])
                else:
                    self.objective_coords.append(observation["desired_goal"][i*3:(i+3)])
            else:
                # last object
                self.objective_coords.append(observation["desired_goal"][15:])

    def _object_map(self):
        observation = self._get_obs()
        #object_coords = []
        for i in range(6):
            if i < 5:
                # first 3 coords
                if i > 0:
                    self.object_coords.append(observation["achieved_goal"][i*3:(i*3)+3])
                else:
                    self.object_coords.append(observation["achieved_goal"][i*3:(i+3)])
            else:
                # last object
                self.object_coords.append(observation["achieved_goal"][15:])
    def _robot_action(self, action):
        # move to object
        observation = self._get_obs()
        current_position = np.round(observation["observation"][0:3], decimals=3)

        # move to the object
        if action == 0:
            goal_position = self.object_coords[self.object_counter]
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
                goal_position = self.objective_coords[self.object_counter] + np.array([0.0, 0.0, 0.09])
                if np.allclose(current_position, goal_position, 0.1):
                    self.higher_z = True
                    return current_position
                else: 
                    return 4.6 * (goal_position - current_position)
            else:
                goal_position = self.objective_coords[self.object_counter]
                if np.allclose(current_position, goal_position, self.task.distance_threshold):
                    self.is_action_completed = True
                    self.object_counter += 1
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
            return gripper_action

        

    def step(self, action: np.ndarray):
        # the actions need to be changed in order to work deterministically
        reward = -1
        
        goal_action = action
        for i in range(100):
            # check to see if we are done with the overall objective
            if self.is_action_completed and goal_action == 1:
                observation = self._get_obs()
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
                observation = self._get_obs()
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                #self.robot.reset()
                return observation, reward, terminated, truncated, info
            action = self._robot_action(goal_action)
            #dont forget to add the gripper
            if goal_action < 2:
                gripper = 0.0 if self.robot.get_fingers_width() else 1.0
                action = np.append(action, gripper)
            self.robot.set_action(action)
            self.sim.step()


class JengaPickAndPlaceEnv(RobotTaskEnv):
    """Pick and place task for Jenga blocks with the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position = None, 
            render_distance = 1.4, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
            object_size: str = "large",
            robot: str = "panda"
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        if robot == "kinova":
            robot = Kinova(sim, block_gripper=False, base_position=np.array([-0.6, -0.406, 0.0]), control_type=control_type)
        else:
            robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, -0.406, 0.0]), control_type=control_type)
        task = JengaPickAndPlace(sim, reward_type=reward_type, object_size=object_size)
        super().__init__(
            robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )

class JengaStack3Env(RobotTaskEnv):
    """Stack task with 3 blocks for the Panda robot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """
    def __init__(
            self,
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_mode: str ="rgb_array",
            render_width = 720, 
            render_height = 480, 
            render_target_position: Optional[np.ndarray] = None, 
            render_distance = 1.4, 
            render_yaw = 45, 
            render_pitch = -30, 
            render_roll = 0,
        ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, -0.406, 0.0]), control_type=control_type)
        task = JengaStack3(sim, reward_type=reward_type)
        super().__init__(
            robot, 
            task, 
            render_width, 
            render_height, 
            render_target_position, 
            render_distance, 
            render_yaw, 
            render_pitch, 
            render_roll
        )

class JengaSimplePickAndPlaceEnv(RobotTaskEnv):                                                                           
    """Pick and place task for Jenga blocks with the Panda robot.                                                   
    Args:                                                                                                           
        render_mode (str, optional): Render mode. Defaults to "rgb_array".                                          
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".                                     
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.    
            Defaults to "ee".                                                                                       
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"  
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.           
        render_width (int, optional): Image width. Defaults to 720.                                                 
        render_height (int, optional): Image height. Defaults to 480.                                               
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).                
            Defaults to [0., 0., 0.].                                                                               
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.                                 
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.                                            
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.                                       
        render_roll (int, optional): Roll of the camera. Defaults to 0.                                             
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters     
    """                                                                                                             
    def __init__(                                                                                                   
            self,                                                                                                   
            reward_type: str = "sparse",                                                                            
            control_type: str = "ee",                                                                               
            renderer: str = "Tiny",                                                                                 
            render_mode: str ="rgb_array",                                                                          
            render_width = 720,                                                                                     
            render_height = 480,                                                                                    
            render_target_position = None,                                                                          
            render_distance = 1.4,                                                                                  
            render_yaw = 45,                                                                                        
            render_pitch = -30,                                                                                     
            render_roll = 0,                                                                                        
            object_size: str = "large"                                                                              
        ) -> None:                                                                                                  
        sim = PyBullet(render_mode=render_mode, renderer=renderer)                                                  
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaSimplePickAndPlace(sim, reward_type=reward_type, object_size=object_size)                             
        super().__init__(                                                                                           
            robot,                                                                                                  
            task,                                                                                                   
            render_width,                                                                                           
            render_height,                                                                                          
            render_target_position,                                                                                 
            render_distance,                 
            render_yaw,
            render_pitch,
            render_roll
        )

class JengaSimplePickAndPlaceDeterministicEnv(RobotTaskEnv):                                                                           
    """Pick and place task for Jenga blocks with the Panda robot.                                                   
    Args:                                                                                                           
        render_mode (str, optional): Render mode. Defaults to "rgb_array".                                          
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".                                     
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.    
            Defaults to "ee".                                                                                       
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"  
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.           
        render_width (int, optional): Image width. Defaults to 720.                                                 
        render_height (int, optional): Image height. Defaults to 480.                                               
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).                
            Defaults to [0., 0., 0.].                                                                               
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.                                 
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.                                            
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.                                       
        render_roll (int, optional): Roll of the camera. Defaults to 0.                                             
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters     
    """                                                                                                             
    def __init__(                                                                                                   
            self,                                                                                                   
            reward_type: str = "sparse",                                                                            
            control_type: str = "ee",                                                                               
            renderer: str = "Tiny",                                                                                 
            render_mode: str ="rgb_array",                                                                          
            render_width = 720,                                                                                     
            render_height = 480,                                                                                    
            render_target_position = None,                                                                          
            render_distance = 1.4,                                                                                  
            render_yaw = 45,                                                                                        
            render_pitch = -30,                                                                                     
            render_roll = 0,                                                                                        
            object_size: str = "large"                                                                              
        ) -> None:                                                                                                  
        sim = PyBullet(render_mode=render_mode, renderer=renderer)                                                  
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaSimplePickAndPlace(sim, reward_type=reward_type, object_size=object_size)   
        self.is_action_completed = False  
        #self.action_space = Discrete(4,)                      
        super().__init__(                                                                                           
            robot,                                                                                                  
            task,                                                                                                   
            render_width,                                                                                           
            render_height,                                                                                          
            render_target_position,                                                                                 
            render_distance,                 
            render_yaw,
            render_pitch,
            render_roll
        )

    def _robot_action(self, action):
        # move to object
        observation = self._get_obs()
        current_position = observation["observation"][0:3]
        # move to the object
        if action == 0:
            goal_position = observation["achieved_goal"][0:3]
            if np.allclose(current_position, goal_position, self.task.distance_threshold):
                self.is_action_completed = True
                return current_position
            else:
                return 1.0 * (goal_position - current_position)
        elif action == 1:
            goal_position = observation["desired_goal"][0:3]
            if np.allclose(current_position, goal_position, self.task.distance_threshold):
                self.is_action_completed = True
                return current_position
            else: 
                return 1.0 * (goal_position - current_position)
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
            return gripper_action

        

    def step(self, action: np.ndarray):
        # the actions need to be changed in order to work deterministically
        reward = -1
        goal_action = action
        for i in range(1000):
            # check to see if we are done with the overall objective
            if self.is_action_completed and goal_action == 1:
                observation = self._get_obs()
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                return observation, reward, terminated, truncated, info
            elif self.is_action_completed and goal_action != 1:
                observation = self._get_obs()
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                return observation, reward, terminated, truncated, info
            action = self._robot_action(goal_action)
            #dont forget to add the gripper
            if goal_action < 2:
                gripper = 0.0 if self.robot.get_fingers_width() else 1.0
                action = np.append(action, gripper)
            self.robot.set_action(action)
            self.sim.step()

class JengaPickAndPlaceDeterministicEnv(RobotTaskEnv):                                                                           
    """Pick and place task for Jenga blocks with the Panda robot.                                                   
    Args:                                                                                                           
        render_mode (str, optional): Render mode. Defaults to "rgb_array".                                          
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".                                     
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.    
            Defaults to "ee".                                                                                       
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"  
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.           
        render_width (int, optional): Image width. Defaults to 720.                                                 
        render_height (int, optional): Image height. Defaults to 480.                                               
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).                
            Defaults to [0., 0., 0.].                                                                               
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.                                 
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.                                            
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.                                       
        render_roll (int, optional): Roll of the camera. Defaults to 0.                                             
        object_size (str, optional): Size of the Jenga block. Defaults to large (0.12 x 0.0381 x 0.0254) meters     
    """                                                                                                             
    def __init__(                                                                                                   
            self,                                                                                                   
            reward_type: str = "sparse",                                                                            
            control_type: str = "ee",                                                                               
            renderer: str = "Tiny",                                                                                 
            render_mode: str ="rgb_array",                                                                          
            render_width = 720,                                                                                     
            render_height = 480,                                                                                    
            render_target_position = None,                                                                          
            render_distance = 1.4,                                                                                  
            render_yaw = 45,                                                                                        
            render_pitch = -30,                                                                                     
            render_roll = 0,                                                                                        
            object_size: str = "large"                                                                              
        ) -> None:                                                                                                  
        sim = PyBullet(render_mode=render_mode, renderer=renderer)                                                  
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = JengaPickAndPlace(sim, reward_type=reward_type, object_size=object_size)   
        self.is_action_completed = False  
        #self.action_space = Discrete(4,)                      
        super().__init__(                                                                                           
            robot,                                                                                                  
            task,                                                                                                   
            render_width,                                                                                           
            render_height,                                                                                          
            render_target_position,                                                                                 
            render_distance,                 
            render_yaw,
            render_pitch,
            render_roll
        )

    def _robot_action(self, action):
        # move to object
        observation = self._get_obs()
        current_position = observation["observation"][0:3]
        # move to the object
        if action == 0:
            goal_position = observation["achieved_goal"][0:3]
            if np.allclose(current_position, goal_position, self.task.distance_threshold):
                self.is_action_completed = True
                return current_position
            else:
                return 1.0 * (goal_position - current_position)
        elif action == 1:
            goal_position = observation["desired_goal"][0:3]
            if np.allclose(current_position, goal_position, self.task.distance_threshold):
                self.is_action_completed = True
                return current_position
            else: 
                return 1.0 * (goal_position - current_position)
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
            return gripper_action

        

    def step(self, action: np.ndarray):
        # the actions need to be changed in order to work deterministically
        reward = -1
        goal_action = action
        for i in range(1000):
            # check to see if we are done with the overall objective
            if self.is_action_completed and goal_action == 1:
                observation = self._get_obs()
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                return observation, reward, terminated, truncated, info
            elif self.is_action_completed and goal_action != 1:
                observation = self._get_obs()
                reward = reward * i
                terminated = True
                truncated = False
                info = {"is_success": terminated,
                        "time_taken": i
                    }
                self.is_action_completed = False
                return observation, reward, terminated, truncated, info
            action = self._robot_action(goal_action)
            #dont forget to add the gripper
            if goal_action < 2:
                gripper = 0.0 if self.robot.get_fingers_width() else 1.0
                action = np.append(action, gripper)
            self.robot.set_action(action)
            self.sim.step()