from gymnasium import Wrapper
from gymnasium import ActionWrapper
from gymnasium.spaces import Box

class DeterministicActions(ActionWrapper):
    def __init__(self, env):
        # set the new action space
        self.action_space = Box(0, 3.0, (2,))
        super().__init__(env)

    def action(self, action):
        """Converts the actions to a deterministic form for our
        task.
        """
        return action