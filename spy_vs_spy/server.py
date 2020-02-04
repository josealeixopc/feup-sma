from spy_vs_spy.env.spy_vs_spy_ma_env import SpyVsSpyEnv, MultiAgentServer


if __name__ == '__main__':

    spy_vs_spy_env = SpyVsSpyEnv()

    spy_vs_spy_server = MultiAgentServer(spy_vs_spy_env, 50051)

    spy_vs_spy_server.serve()
