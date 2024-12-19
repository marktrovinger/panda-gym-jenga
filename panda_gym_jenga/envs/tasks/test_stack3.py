import panda_gym
from panda_gym.pybullet import PyBullet
from panda_gym_jenga.envs.jenga_tasks import JengaStack3

sim = PyBullet(render_mode="human")
task = JengaStack3(sim)
#task = stack(sim)

task.reset()
print(task.get_obs())
print(task.get_achieved_goal())
print(task.is_success(task.get_achieved_goal(), task.get_goal()))
print(task.compute_reward(task.get_achieved_goal(), task.get_goal()))