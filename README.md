# feup-sma

A repository for the Multi-Agent Systems (SMA) course of FEUP's Doctoral Program in Informatics Engineering.

This repository contains the implementation of a very simple Multi-Agent Reinforcement Learning (MARL) framework based on OpenAI Gym, which allows to train agents in turn-based scenarios.

We allow different processes (that may even be in different machines) to connect to a single reinforcement learning environment using gRPC. These processes are simply OpenAI Gym environments and, hence, can be trained using available baselines.

## multi_agent_gym

There are two main classes in our framework: `MultiAgentServer` and `AgentEnv`.

### MultiAgentServer

`MultiAgentServer` contains both a `MultiAgentEnv` and a `MultiAgentServicer` object. It is a gRPC server that, once started, can receive requests containing states and actions represented by NumPy arrays (see `utils.numproto.py`) and redirect them to its `MultiAgentEnv` object.

`MultiAgentEnv` is responsible for implementing the logic part of the environment. That is, how each agent's actions affects the environment (note that the `reset` and `step` functions require an `agent_id`. For an example, check `spy_vs_spy.env`. In this particular case, the environments only moves forward after all agents have called `reset` or `step`. We achieve synchronization using Python Barriers.

### AgentEnv

`AgentEnv` is merely an extension of an OpenAI Gym enviornment. To create your own agent, you need only to extend it and implement the functions that describe the observation and action space. After that, you may connect to an already running `MultiAgentServer`.

The `reset` and `step` functions of an `AgentEnv` already deal with the conversion of actions as NumPy arrays to Proto messages, send them to the `MultiAgentServer` and convert the response (the state) back to a NumPy array.




