# Reward Structure
## Proposed Reward Structure
Similar to the robotic construction paper, I propose that we use a reward structure:
$R_{t} = -T + N_{c} + O_{c}$
where $N_{c}$ is the number of correctly placed components and the timesteps taken is subtracted from this reward. We also need to keep track of items that are correctly placed in order, which is the $O_{c}$ term.
# State and Action Spaces
## State Space
The number of states would be the number of possible configurations of the components, so in the case of 6 blocks, we would have $n! = 6! = 720$ states. 

## Action Space
The deterministic actions would be represented as a list of actions: `[0, 1, 2, 3]`, which represent move arm to object, move arm to objective, and pick, and release.
# RL Algorithms
## $Q$-learning
