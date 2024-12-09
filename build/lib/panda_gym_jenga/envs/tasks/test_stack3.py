from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks import stack
from stack3 import Stack3

sim = PyBullet(render_mode="human")
task = Stack3(sim)
#task = stack(sim)

task.reset()
print(task.get_obs())
print(task.get_achieved_goal())
print(task.is_success(task.get_achieved_goal(), task.get_goal()))
print(task.compute_reward(task.get_achieved_goal(), task.get_goal()))