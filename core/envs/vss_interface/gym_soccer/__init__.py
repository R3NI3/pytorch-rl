from gym.envs.registration import register

register(
    id='vss_soccer-v0',
    entry_point='core.envs.vss_interface.gym_soccer.envs:SoccerEnv'
)

register(
    id='vss_soccer-v1',
    entry_point='core.envs.vss_interface.gym_soccer.envs:SoccerEnv_v1'
)
