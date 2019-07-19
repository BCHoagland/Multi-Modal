from gym.envs.registration import register

register(
    id='Nodeworld-v0',
    entry_point='nodeworld.envs:Nodeworld'
)
