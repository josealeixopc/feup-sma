from spy_vs_spy.env.spy_vs_spy_ma_env import RedSniperEnv

if __name__ == '__main__':

    env = RedSniperEnv('red-sniper', 'localhost:50051')

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            # env.render()
            print("Observation of step {}: {}".format(t, observation))
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
