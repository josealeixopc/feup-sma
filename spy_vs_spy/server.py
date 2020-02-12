import argparse

from spy_vs_spy.env.spy_vs_spy_ma_env import SpyVsSpyEnv, SpyVsSpyUniquenessObservationEnv, MultiAgentServer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the SpyVsSpy server.')
    parser.add_argument('type', help='default|uniqueness')

    args = parser.parse_args()

    if args.type == 'default':
        spy_vs_spy_env = SpyVsSpyEnv()
    elif args.type == 'uniqueness':
        spy_vs_spy_env = SpyVsSpyUniquenessObservationEnv()
    else:
        raise NotImplementedError

    spy_vs_spy_server = MultiAgentServer(spy_vs_spy_env, 50051)

    spy_vs_spy_server.serve()
