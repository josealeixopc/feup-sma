from .red_spy_env import RedSpyEnv
from .red_sniper_env import RedSniperEnv


class SpyVsSpyEnv:
    def __init__(self, num_grey_scouts):
        self.num_grey_scouts = num_grey_scouts

        # Init Gym environments
        self.red_spy_env = RedSpyEnv(self)
        self.red_sniper_env = RedSniperEnv(self)


